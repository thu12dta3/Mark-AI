import os
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from src.model.mark_arch import MarkConfig, MarkForCausalLM, PagedKVCache
from src.model.self_rag import ElevenLabsFishVoiceFormatter, SelfRAGController, SpecialTokens
MAX_ALLOWED_VRAM_GB: float = 7.5

def enforce_vram_ceiling(max_vram_gb: float=MAX_ALLOWED_VRAM_GB) -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {'total_vram_gb': 0.0, 'capped_vram_gb': max_vram_gb}
    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / 1024 ** 3
    fraction = min(0.9375, max_vram_gb / max(total_gb, 1e-06))
    torch.cuda.set_per_process_memory_fraction(fraction, device=0)
    torch.cuda.empty_cache()
    return {'total_vram_gb': round(total_gb, 2), 'capped_vram_gb': round(total_gb * fraction, 2), 'fraction': round(fraction, 4)}

class NativeIQ4NLLinear(nn.Module):

    def __init__(self, in_features: int, out_features: int, has_bias: bool=False, quant_type: str='IQ4_NL', block_size: int=32, compute_dtype: torch.dtype=torch.float16):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quant_type = quant_type.upper()
        self.block_size = block_size
        self.compute_dtype = compute_dtype
        from src.model.mark_arch import IQ4_NL_LOOKUP_VALUES
        self.register_buffer('iq4_lut', torch.tensor(IQ4_NL_LOOKUP_VALUES, dtype=compute_dtype), persistent=False)
        pad_in = (block_size - in_features % block_size) % block_size
        self.padded_in = in_features + pad_in
        n_blocks = self.padded_in // block_size
        if self.quant_type == 'Q8_0':
            self.register_buffer('qweight', torch.zeros((out_features, n_blocks, block_size), dtype=torch.int8))
        else:
            self.register_buffer('qweight', torch.zeros((out_features, n_blocks, block_size // 2), dtype=torch.uint8))
        self.register_buffer('scales', torch.ones((out_features, n_blocks, 1), dtype=torch.float16))
        if has_bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=compute_dtype), requires_grad=False)
        else:
            self.register_parameter('bias', None)

    @classmethod
    def from_linear(cls, linear: nn.Linear, quant_type: str='IQ4_NL', block_size: int=32, compute_dtype: torch.dtype=torch.float16) -> 'NativeIQ4NLLinear':
        layer = cls(linear.in_features, linear.out_features, has_bias=linear.bias is not None, quant_type=quant_type, block_size=block_size, compute_dtype=compute_dtype)
        with torch.no_grad():
            w = linear.weight.data.float()
            if layer.padded_in > layer.in_features:
                w = F.pad(w, (0, layer.padded_in - layer.in_features))
            n_blocks = layer.padded_in // block_size
            w_blocks = w.view(layer.out_features, n_blocks, block_size)
            scale = w_blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-06)
            norm_w = (w_blocks / scale).clamp(-1.0, 1.0)
            layer.scales.copy_(scale.to(torch.float16))
            if layer.quant_type == 'Q8_0':
                layer.qweight.copy_(torch.round(norm_w * 127.0).to(torch.int8))
            elif layer.quant_type == 'Q4_0':
                idx = torch.round((norm_w + 1.0) * 7.5).clamp(0, 15).to(torch.uint8)
                low = idx[..., 0::2]
                high = idx[..., 1::2]
                layer.qweight.copy_(high << 4 | low & 15)
            else:
                lut = layer.iq4_lut.float().to(w.device)
                diffs = (norm_w.unsqueeze(-1) - lut).abs()
                idx = torch.argmin(diffs, dim=-1).to(torch.uint8)
                low = idx[..., 0::2]
                high = idx[..., 1::2]
                layer.qweight.copy_(high << 4 | low & 15)
            if linear.bias is not None and layer.bias is not None:
                layer.bias.data.copy_(linear.bias.data.to(compute_dtype))
        return layer

    def dequantize_weight(self) -> torch.Tensor:
        scale = self.scales.to(self.compute_dtype)
        if self.quant_type == 'Q8_0':
            w_blocks = self.qweight.to(self.compute_dtype) / 127.0 * scale
        else:
            low = (self.qweight & 15).long()
            high = (self.qweight >> 4 & 15).long()
            idx = torch.empty((self.out_features, self.qweight.shape[1], self.block_size), dtype=torch.long, device=self.qweight.device)
            idx[..., 0::2] = low
            idx[..., 1::2] = high
            if self.quant_type == 'Q4_0':
                norm_w = idx.to(self.compute_dtype) / 7.5 - 1.0
            else:
                norm_w = self.iq4_lut[idx]
            w_blocks = norm_w * scale
        w = w_blocks.view(self.out_features, self.padded_in)[:, :self.in_features]
        return w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.dequantize_weight().to(x.dtype)
        b = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, w, b)

def quantize_mark_model_nf4(model: nn.Module, compute_dtype: torch.dtype=torch.float16, weight_quant_type: str='IQ4_NL') -> nn.Module:
    has_bnb = False
    if weight_quant_type.upper() == 'NF4':
        try:
            import bitsandbytes as bnb
            has_bnb = torch.cuda.is_available()
        except ImportError:
            has_bnb = False
    for name, module in model.named_children():
        if name in ('tok_embeddings', 'lm_head', 'norm', 'router', 'prosody_head'):
            continue
        if isinstance(module, nn.Linear):
            if has_bnb and weight_quant_type.upper() == 'NF4':
                q_linear = bnb.nn.Linear4bit(module.in_features, module.out_features, bias=module.bias is not None, compute_dtype=compute_dtype, compress_statistics=True, quant_type='nf4')
                q_linear.weight = bnb.nn.Params4bit(module.weight.data.cpu(), requires_grad=False, quant_type='nf4', compress_statistics=True)
                if module.bias is not None:
                    q_linear.bias = nn.Parameter(module.bias.data.to(compute_dtype))
                setattr(model, name, q_linear)
            else:
                q_native = NativeIQ4NLLinear.from_linear(module, quant_type=weight_quant_type if weight_quant_type.upper() != 'NF4' else 'IQ4_NL', compute_dtype=compute_dtype)
                setattr(model, name, q_native)
        else:
            quantize_mark_model_nf4(module, compute_dtype=compute_dtype, weight_quant_type=weight_quant_type)
    return model

def _is_quantized_checkpoint(sd: Dict[str, Any]) -> bool:
    for k, v in sd.items():
        if k.endswith('.qweight') or (isinstance(v, dict) and 'packed' in v):
            return True
    return False

def load_mark_for_8gb_vram(checkpoint_path: Optional[str]=None, config: Optional[MarkConfig]=None, max_context_tokens: int=8192, weight_quant_type: str='IQ4_NL', kv_cache_quant: str='IQ4_NL', variant: str='1.5b') -> Tuple[nn.Module, PagedKVCache, Dict[str, float]]:
    vram_info = enforce_vram_ceiling(MAX_ALLOWED_VRAM_GB)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    compute_dtype = torch.float16 if device.type == 'cuda' else torch.float32
    if checkpoint_path is None:
        from pathlib import Path
        repo_root = Path(__file__).resolve().parent.parent
        variant_norm = variant.lower().strip()
        candidate_names = [f'mark_{variant_norm}.pt', f'mark_engineer_{variant_norm}.pt', f'mark_{variant_norm}_iq4_nl.pt', f'mark_engineer_{variant_norm}_iq4_nl.pt', f"mark_{variant_norm.replace('.', '_')}_iq4_nl.pt", f"mark_engineer_{variant_norm.replace('.', '_')}_iq4_nl.pt", f"mark_general_{variant_norm.replace('.', '_')}_iq4_nl.pt", 'mark_engineer_302m_iq4_nl.pt', 'mark_engineer_1_5b_iq4_nl.pt', 'mark_general_1_5b_iq4_nl.pt']
        search_dirs = [repo_root / 'checkpoints', Path('checkpoints'), Path('/mnt/ssd_1/mark ai/checkpoints')]
        for d in search_dirs:
            for name in candidate_names:
                p = d / name
                if p.exists():
                    checkpoint_path = str(p)
                    break
            if checkpoint_path:
                break
    sd = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        sd = ckpt.get('model_state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
        cfg_dict = ckpt.get('config', {}) if isinstance(ckpt, dict) else {}
        if cfg_dict and config is None:
            config = MarkConfig(**cfg_dict)
    config = config or MarkConfig.create_engineer_1_5b()
    if sd is not None:
        emb_w = sd.get('tok_embeddings.weight')
        if emb_w is not None and hasattr(emb_w, 'shape'):
            config.vocab_size = emb_w.shape[0]
    model = MarkForCausalLM(config)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model = model.to(dtype=compute_dtype)
    if sd is not None:
        if _is_quantized_checkpoint(sd):
            model = quantize_mark_model_nf4(model, compute_dtype=compute_dtype, weight_quant_type=weight_quant_type)
            model.load_state_dict(sd, strict=False)
        else:
            model.load_state_dict(sd, strict=False)
            model = quantize_mark_model_nf4(model, compute_dtype=compute_dtype, weight_quant_type=weight_quant_type)
    else:
        model = quantize_mark_model_nf4(model, compute_dtype=compute_dtype, weight_quant_type=weight_quant_type)
    model = model.to(device)
    kv_cache = PagedKVCache(n_layers=config.n_layers, max_batch_size=1, max_seq_len=max_context_tokens, n_kv_heads=config.n_kv_heads, head_dim=config.head_dim, dtype=compute_dtype, device=device, use_mla=config.use_mla, kv_lora_rank=config.kv_lora_rank, qk_rope_head_dim=config.qk_rope_head_dim, quant_mode=kv_cache_quant)
    param_bytes = sum((p.numel() * p.element_size() for p in model.parameters()))
    buffer_bytes = sum((b.numel() * b.element_size() for b in model.buffers()))
    weights_mb = (param_bytes + buffer_bytes) / 1024 ** 2
    kv_mb = kv_cache.memory_allocated_mb()
    vram_info.update({'weight_quant_type': weight_quant_type, 'kv_cache_quant_type': kv_cache_quant, 'weights_allocated_mb': round(weights_mb, 2), 'mla_kv_cache_allocated_mb': round(kv_mb, 2), 'total_estimated_gb': round((weights_mb + kv_mb + 450.0) / 1024.0, 3)})
    return (model, kv_cache, vram_info)

class MarkInferenceEngine:

    def __init__(self, model: MarkForCausalLM, tokenizer: Tokenizer, kv_cache: PagedKVCache, rag_controller: Optional[SelfRAGController]=None):
        self.model = model
        self.tokenizer = tokenizer
        self.kv_cache = kv_cache
        self.rag = rag_controller or SelfRAGController()
        self.device = next(model.parameters()).device
        self.bos_id = tokenizer.token_to_id(SpecialTokens.BOS) or 1
        self.eos_id = tokenizer.token_to_id(SpecialTokens.EOS) or 2
        self.think_start_id = tokenizer.token_to_id(SpecialTokens.THINK_START) or 3
        self.retrieve_end_id = tokenizer.token_to_id(SpecialTokens.RETRIEVE_END) or 6
        self.unknown_id = tokenizer.token_to_id(SpecialTokens.UNKNOWN) or 14
        self.web_search_end_id = tokenizer.token_to_id(SpecialTokens.WEB_SEARCH_END) or 16

    @torch.inference_mode()
    def generate(self, prompt: str, max_new_tokens: int=512, temperature: float=0.2, top_p: float=0.9, force_thinking: bool=True, pixel_values: Optional[torch.Tensor]=None) -> str:
        self.kv_cache.reset()
        prompt_ids = [self.bos_id] + self.tokenizer.encode(prompt).ids
        if force_thinking and prompt_ids[-1] != self.think_start_id:
            prompt_ids.append(self.think_start_id)
        cur_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        start_pos = 0
        generated_ids = list(prompt_ids[1:])
        retrieval_hops = 0
        for step_i in range(max_new_tokens):
            logits = self.model(cur_ids, kv_cache=self.kv_cache, start_pos=start_pos, pixel_values=pixel_values if step_i == 0 else None)
            start_pos += cur_ids.shape[1]
            next_token_logits = logits[:, -1, :].float()
            if temperature <= 0.0:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                probs = F.softmax(next_token_logits / temperature, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                mask = cumulative_probs - sorted_probs > top_p
                sorted_probs[mask] = 0.0
                sorted_probs.div_(sorted_probs.sum(dim=-1, keepdim=True))
                sampled_idx = torch.multinomial(sorted_probs, num_samples=1)
                next_token = torch.gather(sorted_indices, -1, sampled_idx)
            token_id = next_token.item()
            if token_id == self.eos_id:
                break
            generated_ids.append(token_id)
            cur_ids = next_token
            if token_id == self.retrieve_end_id and retrieval_hops < self.rag.max_retrieval_hops:
                decoded_so_far = self.tokenizer.decode(generated_ids)
                query = self.rag.extract_retrieve_query(decoded_so_far)
                if query:
                    retrieval_hops += 1
                    rag_block = self.rag.format_rag_injection(query)
                    rag_ids = self.tokenizer.encode(rag_block).ids
                    generated_ids.extend(rag_ids)
                    cur_ids = torch.tensor([rag_ids], dtype=torch.long, device=self.device)
            elif token_id in (self.unknown_id, self.web_search_end_id) and retrieval_hops < self.rag.max_retrieval_hops:
                decoded_so_far = self.tokenizer.decode(generated_ids)
                web_query = self.rag.extract_web_search_query(decoded_so_far, fallback_prompt=prompt)
                if web_query:
                    retrieval_hops += 1
                    honest_notice = ''
                    if token_id == self.unknown_id and 'I need to search about this' not in decoded_so_far:
                        honest_notice = ': "I need to search about this." '
                    web_block = honest_notice + self.rag.format_web_search_injection(web_query)
                    web_ids = self.tokenizer.encode(web_block).ids
                    generated_ids.extend(web_ids)
                    cur_ids = torch.tensor([web_ids], dtype=torch.long, device=self.device)
        return self.tokenizer.decode(generated_ids)

    def generate_with_voice_payload(self, prompt: str, voice_engine: str='elevenlabs', language: str='en', max_new_tokens: int=512, pixel_values: Optional[torch.Tensor]=None) -> Dict[str, Any]:
        raw_output = self.generate(prompt=prompt, max_new_tokens=max_new_tokens, pixel_values=pixel_values)
        voice_payload = ElevenLabsFishVoiceFormatter.extract_spoken_stream(raw_output, target_engine=voice_engine, language=language)
        return {'raw_response': raw_output, 'voice_payload': voice_payload}
