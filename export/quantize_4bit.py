"""
Mark 0.1 - 8GB VRAM Local Inference Engine & Dynamic 4-Bit NF4 Quantizer
Garante execução estritamente abaixo de 7.5GB de VRAM combinando:
- Quantização 4-bit NormalFloat (NF4) via bitsandbytes (`Linear4bit`) nas projeções Transformer
- PagedKVCache pré-alocado com controle de teto de VRAM
- Loop de decodificação com interceptação nativa de `<think>...</think>` e `[RETRIEVE]...[/RETRIEVE]` (Self-RAG)
"""

import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

from src.model.mark_arch import MarkConfig, MarkForCausalLM, PagedKVCache
from src.model.self_rag import SelfRAGController, SpecialTokens


MAX_ALLOWED_VRAM_GB: float = 7.5


def enforce_vram_ceiling(max_vram_gb: float = MAX_ALLOWED_VRAM_GB) -> Dict[str, float]:
    """
    Aplica trava de segurança no alocador CUDA do PyTorch (`set_per_process_memory_fraction`)
    para garantir que o processo jamais ultrapasse 7.5GB de VRAM na GPU local de 8GB.
    """
    if not torch.cuda.is_available():
        return {"total_vram_gb": 0.0, "capped_vram_gb": max_vram_gb}

    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / (1024 ** 3)
    fraction = min(0.9375, max_vram_gb / max(total_gb, 1e-6))
    torch.cuda.set_per_process_memory_fraction(fraction, device=0)
    torch.cuda.empty_cache()
    return {
        "total_vram_gb": round(total_gb, 2),
        "capped_vram_gb": round(total_gb * fraction, 2),
        "fraction": round(fraction, 4),
    }


def quantize_mark_model_nf4(
    model: nn.Module,
    compute_dtype: torch.dtype = torch.float16,
) -> nn.Module:
    """
    Converte todas as camadas `nn.Linear` dos blocos Transformer para `bitsandbytes.nn.Linear4bit`
    usando quantização NormalFloat4 (`nf4`) + Double Quantization (`compress_statistics=True`).
    Preserva `tok_embeddings`, `norm` e `lm_head` em FP16 para zero degradação de perplexidade.
    Caso `bitsandbytes` ou GPU CUDA não estejam disponíveis no ambiente de teste, aplica fallback INT8/FP16 dinâmico.
    """
    has_bnb = False
    try:
        import bitsandbytes as bnb  # type: ignore
        has_bnb = torch.cuda.is_available()
    except ImportError:
        has_bnb = False

    for name, module in model.named_children():
        if name in ("tok_embeddings", "lm_head", "norm"):
            continue
        if isinstance(module, nn.Linear):
            if has_bnb:
                q_linear = bnb.nn.Linear4bit(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                    compute_dtype=compute_dtype,
                    compress_statistics=True,  # Double Quantization (economiza ~0.4 bits/param extra)
                    quant_type="nf4",          # NormalFloat4 ótimo para pesos gaussianos
                )
                q_linear.weight = bnb.nn.Params4bit(
                    module.weight.data.cpu(),
                    requires_grad=False,
                    quant_type="nf4",
                    compress_statistics=True,
                )
                if module.bias is not None:
                    q_linear.bias = nn.Parameter(module.bias.data.to(compute_dtype))
                setattr(model, name, q_linear)
            else:
                # Fallback em CPU ou ambiente sem bitsandbytes
                q_cpu = torch.ao.quantization.quantize_dynamic(
                    nn.Sequential(module), {nn.Linear}, dtype=torch.qint8
                )[0]
                setattr(model, name, q_cpu)
        else:
            quantize_mark_model_nf4(module, compute_dtype=compute_dtype)

    return model


def load_mark_for_8gb_vram(
    checkpoint_path: Optional[str] = None,
    config: Optional[MarkConfig] = None,
    max_context_tokens: int = 8192,
) -> Tuple[MarkForCausalLM, PagedKVCache, Dict[str, float]]:
    """
    Inicializa o modelo Mark 0.1 quantizado em 4-bit + PagedKVCache dimensionado para caber sob 7.5GB VRAM.
    Orçamento de Memória (3.0B params @ 8k-32k tokens):
    - Pesos 4-bit NF4 + Double Quant: ~1.68 GB
    - PagedKVCache (GQA 8 KV heads, 8192 tokens, FP16): ~0.50 GB (ou ~2.00 GB em 32k tokens)
    - Buffers de Ativação FlashAttention-2 + Contexto CUDA: ~0.85 GB
    - Total Alocado: ~3.03 GB (em 8k) a ~4.53 GB (em 32k) << Limite de 7.50 GB!
    """
    vram_info = enforce_vram_ceiling(MAX_ALLOWED_VRAM_GB)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    compute_dtype = torch.float16 if device.type == "cuda" else torch.float32

    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        cfg_dict = ckpt.get("config", {})
        config = MarkConfig(**cfg_dict) if cfg_dict else (config or MarkConfig())
        model = MarkForCausalLM(config)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
    else:
        config = config or MarkConfig()
        model = MarkForCausalLM(config)

    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Quantiza projeções lineares em 4-bit NF4 antes de mover para a GPU
    if device.type == "cuda":
        model = quantize_mark_model_nf4(model, compute_dtype=compute_dtype)
        model = model.to(device)
    else:
        model = model.to(dtype=compute_dtype)

    kv_cache = PagedKVCache(
        n_layers=config.n_layers,
        max_batch_size=1,
        max_seq_len=max_context_tokens,
        n_kv_heads=config.n_kv_heads,
        head_dim=config.head_dim,
        dtype=compute_dtype,
        device=device,
    )

    weights_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
    kv_mb = kv_cache.memory_allocated_mb()
    vram_info.update({
        "weights_allocated_mb": round(weights_mb, 2),
        "kv_cache_allocated_mb": round(kv_mb, 2),
        "total_estimated_gb": round((weights_mb + kv_mb + 600.0) / 1024.0, 3),
    })

    return model, kv_cache, vram_info


class MarkInferenceEngine:
    """
    Motor de Inferência Autônomo do Mark 0.1 com suporte a:
    - Geração forçada de espaço latente `<think> ... </think>`
    - Interceptação de `[RETRIEVE] query [/RETRIEVE]` para Self-RAG ativo
    """
    def __init__(
        self,
        model: MarkForCausalLM,
        tokenizer: Tokenizer,
        kv_cache: PagedKVCache,
        rag_controller: Optional[SelfRAGController] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.kv_cache = kv_cache
        self.rag = rag_controller or SelfRAGController()
        self.device = next(model.parameters()).device

        self.bos_id = tokenizer.token_to_id(SpecialTokens.BOS) or 1
        self.eos_id = tokenizer.token_to_id(SpecialTokens.EOS) or 2
        self.think_start_id = tokenizer.token_to_id(SpecialTokens.THINK_START) or 3
        self.retrieve_end_id = tokenizer.token_to_id(SpecialTokens.RETRIEVE_END) or 6

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        top_p: float = 0.9,
        force_thinking: bool = True,
    ) -> str:
        self.kv_cache.reset()
        prompt_ids = [self.bos_id] + self.tokenizer.encode(prompt).ids
        if force_thinking and prompt_ids[-1] != self.think_start_id:
            prompt_ids.append(self.think_start_id)

        cur_ids = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        start_pos = 0
        generated_ids = list(prompt_ids[1:])  # exclui BOS da decodificação textual
        retrieval_hops = 0

        for _ in range(max_new_tokens):
            logits = self.model(cur_ids, kv_cache=self.kv_cache, start_pos=start_pos)
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

            # Interceptador Self-RAG: se o modelo fechou uma tag [/RETRIEVE], pausa a GPU e busca no repositório!
            if token_id == self.retrieve_end_id and retrieval_hops < self.rag.max_retrieval_hops:
                decoded_so_far = self.tokenizer.decode(generated_ids)
                query = self.rag.extract_retrieve_query(decoded_so_far)
                if query:
                    retrieval_hops += 1
                    rag_block = self.rag.format_rag_injection(query)
                    rag_ids = self.tokenizer.encode(rag_block).ids
                    generated_ids.extend(rag_ids)
                    cur_ids = torch.tensor([rag_ids], dtype=torch.long, device=self.device)

        return self.tokenizer.decode(generated_ids)
