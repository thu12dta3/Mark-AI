#!/usr/bin/env python3
import os
import sys
import gc
import json
import tempfile
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
import numpy as np
import gguf

from src.model.mark_arch import MarkConfig, MarkModel, MarkTransformerBlock, IQ4_NL_LOOKUP_VALUES
from src.model.mark_slm_v2_302m_arch import MarkSLMV2Config, MarkSLMV2Model
from src.model.mark_engineer_v2_8b_arch import MarkEngineerV2Config, MarkEngineerV2Model
from src.model.mark_vl_2_2b_arch import MarkVL2_2BConfig, MarkVLForCausalLM
from src.model.mark_v3_arch import MarkV3Config, MarkV3Block, MarkV3ForCausalLM, get_walsh_hadamard_matrix

lut = torch.tensor(IQ4_NL_LOOKUP_VALUES, dtype=torch.float32)

def quantize_iq4_nl(arr: np.ndarray) -> np.ndarray:
    orig_shape = arr.shape
    in_features = orig_shape[-1]
    if in_features % 32 != 0:
        pad_len = ((in_features + 31) // 32) * 32 - in_features
        pad_shape = list(orig_shape)
        pad_shape[-1] = pad_len
        arr = np.concatenate([arr, np.zeros(pad_shape, dtype=arr.dtype)], axis=-1)
        orig_shape = arr.shape
        in_features = orig_shape[-1]

    n_rows = int(np.prod(orig_shape[:-1])) if len(orig_shape) > 1 else 1
    blocks_per_row = in_features // 32
    arr_2d = arr.reshape(n_rows, blocks_per_row, 32).astype(np.float32)

    kvals = np.array(gguf.quants.IQ4_NL.kvalues, dtype=np.float32)
    midpoints = (kvals[:-1] + kvals[1:]) / 2.0

    blocks = np.empty((n_rows, blocks_per_row, 18), dtype=np.uint8)

    chunk_size = 2048
    for r_start in range(0, n_rows, chunk_size):
        r_end = min(r_start + chunk_size, n_rows)
        chunk = arr_2d[r_start:r_end]

        max_val = np.max(np.abs(chunk), axis=-1, keepdims=True)
        d = (max_val / 127.0).astype(np.float16)
        d_float = d.astype(np.float32)
        d_safe = np.where(d_float == 0, 1.0, d_float)
        x_norm = chunk / d_safe

        best_idx = np.searchsorted(midpoints, x_norm).astype(np.uint8)

        low = best_idx[..., :16]
        high = best_idx[..., 16:]
        qs = (low | (high << 4)).astype(np.uint8)

        blocks[r_start:r_end, :, :2] = d.view(np.uint8).reshape(r_end - r_start, blocks_per_row, 2)
        blocks[r_start:r_end, :, 2:] = qs

    out_shape = (*orig_shape[:-1], blocks_per_row * 18)
    return blocks.reshape(out_shape)

def add_iq4_nl(writer: gguf.GGUFWriter, name: str, tensor_data: Any):
    if isinstance(tensor_data, torch.Tensor):
        arr = tensor_data.detach().cpu().float().numpy()
    else:
        arr = np.asarray(tensor_data, dtype=np.float32)
    q_bytes = quantize_iq4_nl(arr)
    writer.add_tensor(name, q_bytes, raw_dtype=gguf.GGMLQuantizationType.IQ4_NL)

def add_f32(writer: gguf.GGUFWriter, name: str, tensor_data: Any):
    if isinstance(tensor_data, torch.Tensor):
        arr = tensor_data.detach().cpu().float().numpy()
    else:
        arr = np.asarray(tensor_data, dtype=np.float32)
    writer.add_tensor(name, arr)

def unpack_tensor(val, lut=lut) -> Optional[torch.Tensor]:
    if isinstance(val, dict) and 'packed' in val:
        packed = val['packed']
        scales = val['scales'].float().unsqueeze(-1)
        orig_shape = val['orig_shape']
        low = (packed & 15).long()
        high = (packed >> 4 & 15).long()
        idx = torch.zeros((packed.shape[0], 32), dtype=torch.long)
        idx[:, 0::2] = low
        idx[:, 1::2] = high
        norm_w = lut[idx]
        w = (norm_w * scales).reshape(orig_shape)
        return w
    elif isinstance(val, torch.Tensor):
        return val
    return None

def extract_linear_weight(sd: Dict[str, Any], prefix: str, in_features: int, out_features: int, block_size: int = 32) -> torch.Tensor:
    for k in (f'{prefix}.weight', prefix):
        if k in sd:
            res = unpack_tensor(sd[k])
            if res is not None:
                return res.float()
    qw_key = f'{prefix}.qweight'
    sc_key = f'{prefix}.scales'
    if qw_key in sd and sc_key in sd:
        qw = sd[qw_key]
        sc = sd[sc_key]
        low = (qw & 15).long()
        high = (qw >> 4 & 15).long()
        idx = torch.zeros((out_features, qw.shape[1], block_size), dtype=torch.long)
        idx[..., 0::2] = low
        idx[..., 1::2] = high
        norm_w = lut[idx]
        w_blocks = norm_w * sc.float()
        return w_blocks.view(out_features, -1)[:, :in_features].float()
    raise KeyError(f'Unable to find tensor for prefix: {prefix}')

def verify_gguf_integrity(gguf_path: Path):
    print(f'[Mark GGUF Exporter] Verifying GGUF container integrity: {gguf_path}...')
    reader = gguf.GGUFReader(str(gguf_path))
    arch = reader.get_field('general.architecture')
    arch_str = bytes(arch.parts[arch.data[0]]).decode('utf-8') if arch else 'unknown'
    assert arch_str == 'deepseek2', f'Expected architecture deepseek2, got {arch_str}'
    tensor_names = [tensor.name for tensor in reader.tensors]
    assert any(('attn_kv_a_mqa.weight' in t for t in tensor_names)), 'Missing attn_kv_a_mqa.weight in GGUF!'
    assert any(('ffn_gate_exps.weight' in t for t in tensor_names)), 'Missing ffn_gate_exps.weight in GGUF!'
    is_iq4 = any(t.tensor_type == gguf.GGMLQuantizationType.IQ4_NL for t in reader.tensors)
    for tensor in reader.tensors:
        if tensor.name.endswith('_norm.weight') or tensor.name == 'output_norm.weight' or 'exp_probs_b.bias' in tensor.name:
            assert tensor.tensor_type == gguf.GGMLQuantizationType.F32, f'Expected F32 for {tensor.name}, got {tensor.tensor_type}'
        elif is_iq4:
            assert tensor.tensor_type == gguf.GGMLQuantizationType.IQ4_NL, f'Expected IQ4_NL for {tensor.name}, got {tensor.tensor_type}'
        else:
            assert tensor.tensor_type == gguf.GGMLQuantizationType.F16, f'Expected F16 for {tensor.name}, got {tensor.tensor_type}'
    status_type = "IQ4_NL" if is_iq4 else "F16"
    print(f'[Mark GGUF Exporter] Integrity Verified: architecture={arch_str}, total_tensors={len(tensor_names)}, {status_type} validated on all 2D/3D weights')
    del reader, tensor_names
    gc.collect()

def get_model_config(model_type: str) -> Any:
    m = model_type.lower().strip()
    if m in ('302m', 'engineer-302m', '300m', 'mark-engineer-302m'):
        return MarkConfig.create_engineer_300m()
    elif m in ('slm-v2-302m', 'slm_v2', 'mark-slm-v2-302m'):
        return MarkSLMV2Config()
    elif m in ('1.5b', 'engineer-1.5b', 'mark-engineer-1.5b'):
        return MarkConfig.create_engineer_1_5b()
    elif m in ('general', 'general-1.5b', 'mark-general-1.5b'):
        return MarkConfig.create_general_1_5b()
    elif m in ('vl-2.2b', 'mark-vl-2.2b', 'markvl-2.2b', 'markvl'):
        return MarkVL2_2BConfig()
    elif m in ('v2-8b', 'engineer-v2-8b', 'mark-engineer-v2-8b'):
        return MarkEngineerV2Config()
    elif m in ('slm-v3-150m', 'mark-slm-v3-150m'):
        return MarkV3Config.create_slm_v3_150m()
    elif m in ('general-v3-1.5b', 'mark-general-v3-1.5b'):
        return MarkV3Config.create_general_v3_1_5b()
    elif m in ('engineer-v3-1.5b', 'mark-engineer-v3-1.5b'):
        return MarkV3Config.create_engineer_v3_1_5b()
    elif m in ('experimental-v3-8b', 'mark-experimental-v3-8b'):
        return MarkV3Config.create_experimental_v3_8b()
    elif m in ('v3-1.1b', 'v3', 'mark-v3-1.1b', 'mark-v3', 'mark-v3-core-1.1b'):
        return MarkV3Config()
    else:
        raise ValueError(f'Unsupported model type: {model_type}')

def build_model_state_dict(model_type: str, config: Any) -> Dict[str, Any]:
    m = model_type.lower().strip()
    print(f'[Mark GGUF Exporter] Initializing weights from PyTorch architecture for {model_type}...')
    if isinstance(config, MarkV3Config) or 'v3' in m:
        model = MarkV3ForCausalLM(config)
        model.eval()
        for layer in model.model.layers:
            layer.attn.fuse_hadamard_weights_()
        sd = model.state_dict()
    elif m in ('slm-v2-302m', 'slm_v2', 'mark-slm-v2-302m'):
        model = MarkSLMV2Model(config)
        sd = model.state_dict()
    elif m in ('v2-8b', 'engineer-v2-8b', 'mark-engineer-v2-8b'):
        model = MarkEngineerV2Model(config)
        sd = model.state_dict()
    elif m in ('vl-2.2b', 'mark-vl-2.2b', 'markvl-2.2b', 'markvl'):
        model = MarkVLForCausalLM(config)
        sd = model.state_dict()
    else:
        model = MarkModel(config)
        sd = model.state_dict()
    return sd

def export_to_deepseek2_gguf(model_type: str, ckpt_path: Optional[Path], output_gguf: Path, model_name: Optional[str] = None, quant_type: Optional[str] = None):
    config = get_model_config(model_type)
    is_v3 = isinstance(config, MarkV3Config) or 'v3' in model_type.lower()
    if model_name is None:
        model_name = getattr(config, 'model_name', f'mark-{model_type}')

    if quant_type is None:
        if 'iq4_nl' in str(output_gguf).lower() or os.environ.get('MARK_GGUF_QUANT', '').lower() in ('iq4_nl', 'iq4'):
            quant_type = 'iq4_nl'
        else:
            quant_type = 'f16'

    def add_linear(w_writer: gguf.GGUFWriter, name: str, data: Any):
        if quant_type == 'iq4_nl':
            add_iq4_nl(w_writer, name, data)
        else:
            if isinstance(data, torch.Tensor):
                arr = data.detach().cpu().to(torch.float16).numpy()
            else:
                arr = np.asarray(data, dtype=np.float16)
            w_writer.add_tensor(name, arr)

    if ckpt_path is not None and Path(ckpt_path).exists() and Path(ckpt_path).is_file():
        raw_ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        candidate_sd = raw_ckpt.get('model_state_dict', raw_ckpt) if isinstance(raw_ckpt, dict) else raw_ckpt
        has_tensors = isinstance(candidate_sd, dict) and any('weight' in k for k in candidate_sd.keys())
        if has_tensors:
            print(f'[Mark GGUF Exporter] Loading state_dict from {ckpt_path}...')
            sd = candidate_sd
            emb_w = sd.get('tok_embeddings.weight') if 'tok_embeddings.weight' in sd else sd.get('model.embed_tokens.weight')
            if emb_w is not None and hasattr(emb_w, 'shape'):
                if hasattr(config, 'vocab_size'):
                    config.vocab_size = emb_w.shape[0]
        else:
            print(f'[Mark GGUF Exporter] Checkpoint at {ckpt_path} has no weight tensors. Initializing streaming layer-by-layer for {model_name}...')
            sd = None
    else:
        print(f'[Mark GGUF Exporter] Initializing streaming layer-by-layer for {model_name}...')
        sd = None

    output_gguf.parent.mkdir(parents=True, exist_ok=True)
    orig_tempdir = tempfile.tempdir
    n_layers = getattr(config, 'n_layers', getattr(config, 'num_hidden_layers', 16))
    if n_layers > 4 or getattr(config, 'dim', getattr(config, 'hidden_size', 0)) > 1024:
        spool_dir = Path('/run/media/thu12/hd 1tb/mark_checkpoints/tmp') if Path('/run/media/thu12/hd 1tb').exists() else (Path('/mnt/ssd_1/mark ai/tmp') if Path('/mnt/ssd_1').exists() else Path(tempfile.gettempdir()) / 'mark_tmp')
        spool_dir.mkdir(parents=True, exist_ok=True)
        tempfile.tempdir = str(spool_dir)
    writer = gguf.GGUFWriter(str(output_gguf), 'deepseek2', use_temp_file=True)
    print(f'[Mark GGUF Exporter] Writing DeepSeek2 metadata for {model_name} ({quant_type.upper()})...')
    writer.add_name(model_name)
    writer.add_author('thu12dta3')
    writer.add_version('1.0.0')
    writer.add_description(f'{model_name} - deepseek2 MLA + MoE Native GGUF Container (IQ4_NL)')
    writer.add_context_length(getattr(config, 'max_seq_len', getattr(config, 'max_position_embeddings', 4096)))
    writer.add_embedding_length(getattr(config, 'dim', getattr(config, 'hidden_size', 1280)))
    writer.add_block_count(getattr(config, 'n_layers', getattr(config, 'num_hidden_layers', 16)))
    writer.add_feed_forward_length(getattr(config, 'intermediate_size', 3840))
    writer.add_head_count(getattr(config, 'n_heads', getattr(config, 'num_attention_heads', 16)))
    writer.add_head_count_kv(1)
    writer.add_layer_norm_rms_eps(config.rms_norm_eps)
    writer.add_rope_freq_base(config.rope_theta)
    leading_dense = getattr(config, 'n_dense_layers', 0)
    writer.add_leading_dense_block_count(leading_dense)
    writer.add_vocab_size(config.vocab_size)
    writer.add_q_lora_rank(config.q_lora_rank)
    writer.add_kv_lora_rank(config.kv_lora_rank)
    writer.add_key_length(config.kv_lora_rank + config.qk_rope_head_dim)
    writer.add_value_length(config.kv_lora_rank)
    writer.add_key_length_mla(config.qk_nope_head_dim + config.qk_rope_head_dim)
    writer.add_value_length_mla(config.v_head_dim)
    writer.add_expert_feed_forward_length(config.moe_intermediate_size)
    writer.add_expert_count(config.n_routed_experts)
    writer.add_expert_shared_count(config.n_shared_experts)
    writer.add_expert_used_count(config.num_experts_per_tok)
    writer.add_rope_dimension_count(config.qk_rope_head_dim)
    writer.add_bos_token_id(config.bos_token_id)
    writer.add_eos_token_id(config.eos_token_id)
    writer.add_pad_token_id(config.pad_token_id)

    tok_candidates = [
        REPO_ROOT / 'models' / 'mark-v3-1.1b' / 'tokenizer.json',
        REPO_ROOT / 'models' / f'mark-{model_type}' / 'tokenizer.json',
        REPO_ROOT / 'models' / 'mark-engineer-1.5b' / 'tokenizer.json',
        REPO_ROOT / 'models' / 'mark-vl-2.2b' / 'tokenizer.json',
        REPO_ROOT / 'tokenizer' / 'tokenizer.json'
    ]
    tok_path = next((p for p in tok_candidates if p.exists()), None)
    if tok_path:
        with open(tok_path, 'r', encoding='utf-8') as f:
            tok_data = json.load(f)
        vocab = tok_data.get('model', {}).get('vocab', {})
        sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
        tokens = [k.encode('utf-8') for k, v in sorted_vocab]
        merges = tok_data.get('model', {}).get('merges', [])
        formatted_merges = [' '.join(m).encode('utf-8') if isinstance(m, list) else m.encode('utf-8') for m in merges]
        pad_vocab_size = (len(tokens) + 63) // 64 * 64
        diff = pad_vocab_size - len(tokens)
        for k in range(diff):
            tokens.append(f'<pad_align_{k}>'.encode('utf-8'))
        writer.add_tokenizer_model('gpt2')
        writer.add_token_list(tokens)
        writer.add_token_merges(formatted_merges)
        writer.add_token_scores([0.0] * len(tokens))
        writer.add_token_types([1] * len(tokens))
    else:
        pad_vocab_size = config.vocab_size

    chat_template = 'chat_template_default'
    writer.add_chat_template(chat_template)

    dim = getattr(config, 'dim', getattr(config, 'hidden_size', 1280))
    n_heads = getattr(config, 'n_heads', getattr(config, 'num_attention_heads', 16))
    qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim

    if sd is not None:
        found_layers = set()
        for k in sd.keys():
            if 'attn_norm.weight' in k:
                parts = k.split('.')
                for p in parts:
                    if p.isdigit():
                        found_layers.add(int(p))
        n_layers = len(found_layers) if found_layers else getattr(config, 'n_layers', getattr(config, 'num_hidden_layers', 16))
        emb_key = 'tok_embeddings.weight' if 'tok_embeddings.weight' in sd else 'model.embed_tokens.weight'
        w_emb = extract_linear_weight(sd, emb_key.replace('.weight', ''), dim, config.vocab_size)
        norm_key = 'norm.weight' if 'norm.weight' in sd else 'model.norm.weight'
        norm_raw = sd[norm_key]
        head_key = 'lm_head.weight'
        if head_key in sd:
            w_out = extract_linear_weight(sd, 'lm_head', dim, config.vocab_size)
        else:
            w_out = w_emb
    else:
        n_layers = getattr(config, 'n_layers', getattr(config, 'num_hidden_layers', 16))
        embed_mod = torch.nn.Embedding(config.vocab_size, dim)
        w_emb = embed_mod.weight.data.float()
        norm_raw = torch.ones(dim, dtype=torch.float32)
        if getattr(config, 'tie_word_embeddings', False):
            w_out = w_emb
        else:
            lm_head_mod = torch.nn.Linear(dim, config.vocab_size, bias=False)
            w_out = lm_head_mod.weight.data.float()

    if w_emb.shape[0] < pad_vocab_size:
        pad = torch.zeros((pad_vocab_size - w_emb.shape[0], w_emb.shape[1]), dtype=torch.float32)
        w_emb = torch.cat([w_emb, pad], dim=0)
    add_linear(writer, 'token_embd.weight', w_emb)
    del w_emb
    gc.collect()

    add_f32(writer, 'output_norm.weight', norm_raw)
    del norm_raw
    gc.collect()

    if w_out.shape[0] < pad_vocab_size:
        pad = torch.zeros((pad_vocab_size - w_out.shape[0], w_out.shape[1]), dtype=torch.float32)
        w_out = torch.cat([w_out, pad], dim=0)
    add_linear(writer, 'output.weight', w_out)
    del w_out
    gc.collect()

    for i in range(n_layers):
        b_pfx = f'blk.{i}.'
        if sd is not None:
            pfx = f'layers.{i}.' if f'layers.{i}.attn_norm.weight' in sd else f'model.layers.{i}.'
            cur_sd = sd
            block = None
        else:
            if is_v3:
                block = MarkV3Block(config, layer_idx=i)
                block.attn.fuse_hadamard_weights_()
            else:
                block = MarkTransformerBlock(config, layer_idx=i)
            cur_sd = block.state_dict()
            pfx = ''

        add_f32(writer, f'{b_pfx}attn_norm.weight', cur_sd[f'{pfx}attn_norm.weight'])

        wq_a_prefix = f'{pfx}attn.w_qa' if f'{pfx}attn.w_qa.weight' in cur_sd or f'{pfx}attn.w_qa.qweight' in cur_sd else f'{pfx}attn.wq_a'
        wq_a = extract_linear_weight(cur_sd, wq_a_prefix, dim, config.q_lora_rank)
        add_linear(writer, f'{b_pfx}attn_q_a.weight', wq_a)

        q_norm_key = next((k for k in (f'{pfx}attn.q_norm.weight', f'{pfx}attn.q_a_layernorm.weight') if k in cur_sd), None)
        add_f32(writer, f'{b_pfx}attn_q_a_norm.weight', cur_sd[q_norm_key])

        wq_b_prefix = f'{pfx}attn.w_qb' if f'{pfx}attn.w_qb.weight' in cur_sd or f'{pfx}attn.w_qb.qweight' in cur_sd else f'{pfx}attn.wq_b'
        wq_b = extract_linear_weight(cur_sd, wq_b_prefix, config.q_lora_rank, n_heads * qk_head_dim)
        add_linear(writer, f'{b_pfx}attn_q_b.weight', wq_b)

        wkv_a_prefix = f'{pfx}attn.w_kv_a' if f'{pfx}attn.w_kv_a.weight' in cur_sd or f'{pfx}attn.w_kv_a.qweight' in cur_sd else f'{pfx}attn.wkv_a'
        wkv_a = extract_linear_weight(cur_sd, wkv_a_prefix, dim, config.kv_lora_rank + config.qk_rope_head_dim)

        kv_norm_key = next((k for k in (f'{pfx}attn.kv_norm.weight', f'{pfx}attn.kv_a_layernorm.weight') if k in cur_sd), None)
        kv_norm_w = cur_sd[kv_norm_key].float()

        wkv_b_prefix = f'{pfx}attn.w_kv_b' if f'{pfx}attn.w_kv_b.weight' in cur_sd or f'{pfx}attn.w_kv_b.qweight' in cur_sd else f'{pfx}attn.wkv_b'
        wkv_b = extract_linear_weight(cur_sd, wkv_b_prefix, config.kv_lora_rank, n_heads * (config.qk_nope_head_dim + config.v_head_dim))

        if is_v3 and sd is not None:
            h_mat = get_walsh_hadamard_matrix(config.kv_lora_rank).to(device=wkv_b.device, dtype=torch.float32)
            wkv_b = wkv_b * kv_norm_w.unsqueeze(0)
            kv_norm_w = torch.ones_like(kv_norm_w)
            wkv_a[:config.kv_lora_rank, :] = h_mat @ wkv_a[:config.kv_lora_rank, :]
            wkv_b = wkv_b @ h_mat.T

        add_linear(writer, f'{b_pfx}attn_kv_a_mqa.weight', wkv_a)
        add_f32(writer, f'{b_pfx}attn_kv_a_norm.weight', kv_norm_w)

        kv_b_3d = wkv_b.view(n_heads, config.qk_nope_head_dim + config.v_head_dim, config.kv_lora_rank)
        k_b, v_b = torch.split(kv_b_3d, [config.qk_nope_head_dim, config.v_head_dim], dim=1)
        k_b = k_b.transpose(1, 2)
        add_linear(writer, f'{b_pfx}attn_k_b.weight', k_b.contiguous())
        add_linear(writer, f'{b_pfx}attn_v_b.weight', v_b.contiguous())

        wo_prefix = f'{pfx}attn.w_o' if f'{pfx}attn.w_o.weight' in cur_sd or f'{pfx}attn.w_o.qweight' in cur_sd else f'{pfx}attn.wo'
        wo = extract_linear_weight(cur_sd, wo_prefix, n_heads * config.v_head_dim, dim)
        add_linear(writer, f'{b_pfx}attn_output.weight', wo)

        add_f32(writer, f'{b_pfx}ffn_norm.weight', cur_sd[f'{pfx}ffn_norm.weight'])

        is_dense = (f'{pfx}ffn.w_gate.weight' in cur_sd or f'{pfx}ffn.w_gate.qweight' in cur_sd or f'{pfx}ffn.w_gate' in cur_sd)
        if is_dense:
            w_gate = extract_linear_weight(cur_sd, f'{pfx}ffn.w_gate', dim, config.intermediate_size)
            w_up = extract_linear_weight(cur_sd, f'{pfx}ffn.w_up', dim, config.intermediate_size)
            w_down = extract_linear_weight(cur_sd, f'{pfx}ffn.w_down', config.intermediate_size, dim)
            add_linear(writer, f'{b_pfx}ffn_gate.weight', w_gate)
            add_linear(writer, f'{b_pfx}ffn_up.weight', w_up)
            add_linear(writer, f'{b_pfx}ffn_down.weight', w_down)
        else:
            r_key = next((k for k in (f'{pfx}ffn.router.weight', f'{pfx}ffn.router_gate.weight') if k in cur_sd), None)
            if r_key:
                w_router = cur_sd[r_key]
            else:
                w_router = torch.zeros((config.n_routed_experts, dim), dtype=torch.float32)
            add_linear(writer, f'{b_pfx}ffn_gate_inp.weight', w_router)

            shexp_prefix = f'{pfx}ffn.shared_experts'
            w_sgate = extract_linear_weight(cur_sd, f'{shexp_prefix}.w_gate', dim, config.moe_intermediate_size * config.n_shared_experts)
            w_sup = extract_linear_weight(cur_sd, f'{shexp_prefix}.w_up', dim, config.moe_intermediate_size * config.n_shared_experts)
            w_sdown = extract_linear_weight(cur_sd, f'{shexp_prefix}.w_down', config.moe_intermediate_size * config.n_shared_experts, dim)
            add_linear(writer, f'{b_pfx}ffn_gate_shexp.weight', w_sgate)
            add_linear(writer, f'{b_pfx}ffn_up_shexp.weight', w_sup)
            add_linear(writer, f'{b_pfx}ffn_down_shexp.weight', w_sdown)

            gates, ups, downs = ([], [], [])
            for e in range(config.n_routed_experts):
                g = extract_linear_weight(cur_sd, f'{pfx}ffn.routed_experts.{e}.w_gate', dim, config.moe_intermediate_size)
                u = extract_linear_weight(cur_sd, f'{pfx}ffn.routed_experts.{e}.w_up', dim, config.moe_intermediate_size)
                d = extract_linear_weight(cur_sd, f'{pfx}ffn.routed_experts.{e}.w_down', config.moe_intermediate_size, dim)
                gates.append(g)
                ups.append(u)
                downs.append(d)
            add_linear(writer, f'{b_pfx}ffn_gate_exps.weight', torch.stack(gates, dim=0).contiguous())
            add_linear(writer, f'{b_pfx}ffn_up_exps.weight', torch.stack(ups, dim=0).contiguous())
            add_linear(writer, f'{b_pfx}ffn_down_exps.weight', torch.stack(downs, dim=0).contiguous())

            bias_key = next((k for k in (f'{pfx}ffn.exp_probs_b', f'{pfx}ffn.expert_bias', f'{pfx}ffn.router_bias') if k in cur_sd), None)
            if bias_key:
                add_f32(writer, f'{b_pfx}exp_probs_b.bias', cur_sd[bias_key])

        if block is not None:
            del block, cur_sd
        del wq_a, wq_b, wkv_a, wkv_b, wo
        gc.collect()
        print(f'[Mark GGUF Exporter] Processed block {i + 1}/{n_layers}...')

    print(f'[Mark GGUF Exporter] Finalizing GGUF container: {output_gguf}...')
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    verify_gguf_integrity(output_gguf)
    tempfile.tempdir = orig_tempdir
    size_mb = output_gguf.stat().st_size / (1024 * 1024)
    print(f'[Mark GGUF Exporter] Success: {output_gguf} ({size_mb:.2f} MB)')
    return output_gguf

def main():
    parser = argparse.ArgumentParser(description='Universal DeepSeek2 GGUF Exporter for Mark Models (IQ4_NL)')
    parser.add_argument('--model', type=str, default='1.5b', choices=['302m', 'slm-v2-302m', 'engineer-1.5b', '1.5b', 'general-1.5b', 'vl-2.2b', 'v2-8b', 'v3-1.1b', 'slm-v3-150m', 'general-v3-1.5b', 'engineer-v3-1.5b', 'experimental-v3-8b', 'all'])
    parser.add_argument('--ckpt', type=str, default='', help='Path to source .pt checkpoint')
    parser.add_argument('--out', type=str, default='', help='Path to destination .gguf container')
    parser.add_argument('--quant', type=str, default='iq4_nl', choices=['iq4_nl', 'f16'], help='Quantization type')
    args = parser.parse_args()

    default_paths = {
        '302m': (Path('/mnt/ssd_1/mark ai/checkpoints/mark_engineer_302m_iq4_nl.pt'), Path('/mnt/ssd_1/mark ai/checkpoints/mark_engineer_302m_iq4_nl.gguf')),
        'slm-v2-302m': (Path('/mnt/ssd_1/mark ai/checkpoints/mark_slm_v2_302m.pt'), Path('/mnt/ssd_1/mark ai/checkpoints/mark_slm_v2_302m_iq4_nl.gguf')),
        'engineer-1.5b': (Path('/mnt/ssd_1/mark ai/checkpoints/mark_engineer_1_5b_iq4_nl.pt'), Path('/mnt/ssd_1/mark ai/checkpoints/mark_engineer_1_5b_iq4_nl.gguf')),
        'general-1.5b': (Path('/mnt/ssd_1/mark ai/checkpoints/mark_general_1_5b_iq4_nl.pt'), Path('/mnt/ssd_1/mark ai/checkpoints/mark_general_1_5b_iq4_nl.gguf')),
        'vl-2.2b': (Path('/mnt/ssd_1/mark ai/checkpoints/mark_vl_2_2b/mark_vl_2_2b.pt'), Path('/mnt/ssd_1/mark ai/checkpoints/mark_vl_2_2b_iq4_nl.gguf')),
        'v2-8b': (Path('/mnt/ssd_1/mark ai/checkpoints/mark_engineer_v2_8b/mark_engineer_v2_8b_step_15000.pt'), Path('/mnt/ssd_1/mark ai/checkpoints/mark_engineer_v2_8b_iq4_nl.gguf')),
        'v3-1.1b': (Path('/mnt/ssd_1/mark ai/checkpoints/mark_v3_1_1b.pt'), Path('/mnt/ssd_1/mark ai/checkpoints/mark_v3_1_1b_iq4_nl.gguf')),
        'slm-v3-150m': (Path('/mnt/ssd_1/mark ai/checkpoints/mark_slm_v3_150m.pt'), Path('/mnt/ssd_1/mark ai/checkpoints/mark_slm_v3_150m_iq4_nl.gguf')),
        'general-v3-1.5b': (Path('/mnt/ssd_1/mark ai/checkpoints/mark_general_v3_1_5b.pt'), Path('/mnt/ssd_1/mark ai/checkpoints/mark_general_v3_1_5b_iq4_nl.gguf')),
        'engineer-v3-1.5b': (Path('/mnt/ssd_1/mark ai/checkpoints/mark_engineer_v3_1_5b.pt'), Path('/mnt/ssd_1/mark ai/checkpoints/mark_engineer_v3_1_5b_iq4_nl.gguf')),
        'experimental-v3-8b': (Path('/mnt/ssd_1/mark ai/checkpoints/mark_experimental_v3_8b.pt'), Path('/mnt/ssd_1/mark ai/checkpoints/mark_experimental_v3_8b_iq4_nl.gguf')),
    }

    models_to_export = list(default_paths.keys()) if args.model == 'all' else ['engineer-1.5b' if args.model == '1.5b' else args.model]
    for m in models_to_export:
        def_src, def_dst = default_paths.get(m, (None, Path(f'/mnt/ssd_1/mark ai/checkpoints/{m}.gguf')))
        src = Path(args.ckpt) if args.ckpt and args.model != 'all' else def_src
        dst = Path(args.out) if args.out and args.model != 'all' else def_dst
        export_to_deepseek2_gguf(m, src, dst, quant_type=args.quant)

if __name__ == '__main__':
    main()
