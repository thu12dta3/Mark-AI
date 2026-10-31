#!/usr/bin/env python3
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
import torch
import numpy as np
import gguf
from src.model.mark_arch import MarkConfig, IQ4_NL_LOOKUP_VALUES
from src.model.mark_slm_v2_302m_arch import MarkSLMV2Config
from src.model.mark_engineer_v2_8b_arch import MarkEngineerV2Config
from src.model.mark_v3_arch import MarkV3Config, get_walsh_hadamard_matrix
lut = torch.tensor(IQ4_NL_LOOKUP_VALUES, dtype=torch.float32)

def unpack_tensor(val, lut=lut) -> Optional[torch.Tensor]:
    if isinstance(val, dict) and 'packed' in val:
        packed = val['packed']
        scales = val['scales'].float().unsqueeze(-1)
        orig_shape = val['orig_shape']
        low = (packed & 15).long()
        high = (packed >> 4 & 15).long()
        idx = torch.empty((packed.shape[0], 32), dtype=torch.long)
        idx[:, 0::2] = low
        idx[:, 1::2] = high
        norm_w = lut[idx]
        w = (norm_w * scales).reshape(orig_shape)
        return w
    elif isinstance(val, torch.Tensor):
        return val
    return None

def extract_linear_weight(sd: Dict[str, Any], prefix: str, in_features: int, out_features: int, block_size: int=32) -> torch.Tensor:
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
        idx = torch.empty((out_features, qw.shape[1], block_size), dtype=torch.long)
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
    print(f'[Mark GGUF Exporter] Integrity Verified: architecture={arch_str}, total_tensors={len(tensor_names)}')

def get_model_config(model_type: str) -> Any:
    m = model_type.lower().strip()
    if m in ('302m', 'engineer-302m', '300m'):
        return MarkConfig.create_engineer_300m()
    elif m in ('slm-v2-302m', 'slm_v2'):
        return MarkSLMV2Config()
    elif m in ('1.5b', 'engineer-1.5b'):
        return MarkConfig.create_engineer_1_5b()
    elif m in ('general', 'general-1.5b'):
        return MarkConfig.create_general_1_5b()
    elif m in ('v2-8b', 'engineer-v2-8b'):
        return MarkEngineerV2Config()
    elif m in ('v3-1.1b', 'v3', 'mark-v3-core-1.1b', 'mark-v3'):
        return MarkV3Config()
    else:
        raise ValueError(f'Unsupported model type: {model_type}')

def export_to_deepseek2_gguf(model_type: str, ckpt_path: Path, output_gguf: Path, model_name: Optional[str]=None):
    if not ckpt_path.exists():
        raise FileNotFoundError(f'Source checkpoint does not exist: {ckpt_path}')
    if model_name is None:
        model_name = f'mark-{model_type}'
    print(f'[Mark GGUF Exporter] Loading state_dict from {ckpt_path}...')
    raw_ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    sd = raw_ckpt.get('model_state_dict', raw_ckpt) if isinstance(raw_ckpt, dict) else raw_ckpt
    config = get_model_config(model_type)
    is_v3 = isinstance(config, MarkV3Config) or model_type.lower().strip() in ('v3-1.1b', 'v3', 'mark-v3-core-1.1b', 'mark-v3')
    emb_w = sd.get('tok_embeddings.weight') if 'tok_embeddings.weight' in sd else sd.get('model.embed_tokens.weight')
    if emb_w is not None and hasattr(emb_w, 'shape'):
        if hasattr(config, 'vocab_size'):
            config.vocab_size = emb_w.shape[0]
    output_gguf.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(str(output_gguf), 'deepseek2')
    print('[Mark GGUF Exporter] Writing DeepSeek2 metadata...')
    writer.add_name(model_name)
    writer.add_author('thu12dta3')
    writer.add_version('1.0.0')
    writer.add_description(f'{model_name} - deepseek2 MLA + MoE Native GGUF Container')
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
    tok_candidates = [REPO_ROOT / 'models' / f'mark-{model_type}' / 'tokenizer.json', REPO_ROOT / 'models' / 'mark-v3-core-1.1b' / 'tokenizer.json', REPO_ROOT / 'models' / 'mark-engineer-1.5b' / 'tokenizer.json', REPO_ROOT / 'models' / 'mark-vl-2.2b' / 'tokenizer.json']
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
    found_layers = set()
    for k in sd.keys():
        if 'attn_norm.weight' in k:
            parts = k.split('.')
            for p in parts:
                if p.isdigit():
                    found_layers.add(int(p))
    n_layers = len(found_layers) if found_layers else getattr(config, 'n_layers', getattr(config, 'num_hidden_layers', 16))
    n_heads = getattr(config, 'n_heads', getattr(config, 'num_attention_heads', 16))
    qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
    emb_key = 'tok_embeddings.weight' if 'tok_embeddings.weight' in sd else 'model.embed_tokens.weight'
    w_emb = extract_linear_weight(sd, emb_key.replace('.weight', ''), dim, config.vocab_size).to(torch.float16)
    if w_emb.shape[0] < pad_vocab_size:
        pad = torch.zeros((pad_vocab_size - w_emb.shape[0], w_emb.shape[1]), dtype=torch.float16)
        w_emb = torch.cat([w_emb, pad], dim=0)
    writer.add_tensor('token_embd.weight', w_emb.numpy())
    norm_key = 'norm.weight' if 'norm.weight' in sd else 'model.norm.weight'
    norm_raw = sd[norm_key]
    writer.add_tensor('output_norm.weight', norm_raw.to(torch.float32).numpy())
    head_key = 'lm_head.weight'
    if head_key in sd:
        w_out = extract_linear_weight(sd, 'lm_head', dim, config.vocab_size).to(torch.float16)
    else:
        w_out = w_emb
    if w_out.shape[0] < pad_vocab_size:
        pad = torch.zeros((pad_vocab_size - w_out.shape[0], w_out.shape[1]), dtype=torch.float16)
        w_out = torch.cat([w_out, pad], dim=0)
    writer.add_tensor('output.weight', w_out.numpy())
    for i in range(n_layers):
        pfx = f'layers.{i}.' if f'layers.{i}.attn_norm.weight' in sd else f'model.layers.{i}.'
        b_pfx = f'blk.{i}.'
        writer.add_tensor(f'{b_pfx}attn_norm.weight', sd[f'{pfx}attn_norm.weight'].to(torch.float32).numpy())
        wq_a_prefix = f'{pfx}attn.w_qa' if f'{pfx}attn.w_qa.weight' in sd or f'{pfx}attn.w_qa.qweight' in sd else f'{pfx}attn.wq_a'
        wq_a = extract_linear_weight(sd, wq_a_prefix, dim, config.q_lora_rank)
        writer.add_tensor(f'{b_pfx}attn_q_a.weight', wq_a.to(torch.float16).numpy())
        q_norm_key = next((k for k in (f'{pfx}attn.q_norm.weight', f'{pfx}attn.q_a_layernorm.weight') if k in sd), None)
        writer.add_tensor(f'{b_pfx}attn_q_a_norm.weight', sd[q_norm_key].to(torch.float32).numpy())
        wq_b_prefix = f'{pfx}attn.w_qb' if f'{pfx}attn.w_qb.weight' in sd or f'{pfx}attn.w_qb.qweight' in sd else f'{pfx}attn.wq_b'
        wq_b = extract_linear_weight(sd, wq_b_prefix, config.q_lora_rank, n_heads * qk_head_dim)
        writer.add_tensor(f'{b_pfx}attn_q_b.weight', wq_b.to(torch.float16).numpy())
        wkv_a_prefix = f'{pfx}attn.w_kv_a' if f'{pfx}attn.w_kv_a.weight' in sd or f'{pfx}attn.w_kv_a.qweight' in sd else f'{pfx}attn.wkv_a'
        wkv_a = extract_linear_weight(sd, wkv_a_prefix, dim, config.kv_lora_rank + config.qk_rope_head_dim)
        kv_norm_key = next((k for k in (f'{pfx}attn.kv_norm.weight', f'{pfx}attn.kv_a_layernorm.weight') if k in sd), None)
        kv_norm_w = sd[kv_norm_key].float()
        wkv_b_prefix = f'{pfx}attn.w_kv_b' if f'{pfx}attn.w_kv_b.weight' in sd or f'{pfx}attn.w_kv_b.qweight' in sd else f'{pfx}attn.wkv_b'
        wkv_b = extract_linear_weight(sd, wkv_b_prefix, config.kv_lora_rank, n_heads * (config.qk_nope_head_dim + config.v_head_dim))
        if is_v3:
            h256 = get_walsh_hadamard_matrix(config.kv_lora_rank).to(device=wkv_b.device, dtype=torch.float32)
            wkv_b = wkv_b * kv_norm_w.unsqueeze(0)
            kv_norm_w = torch.ones_like(kv_norm_w)
            wkv_a[:config.kv_lora_rank, :] = h256 @ wkv_a[:config.kv_lora_rank, :]
            wkv_b = wkv_b @ h256.T
        writer.add_tensor(f'{b_pfx}attn_kv_a_mqa.weight', wkv_a.to(torch.float16).numpy())
        writer.add_tensor(f'{b_pfx}attn_kv_a_norm.weight', kv_norm_w.to(torch.float32).numpy())
        kv_b_3d = wkv_b.view(n_heads, config.qk_nope_head_dim + config.v_head_dim, config.kv_lora_rank)
        k_b, v_b = torch.split(kv_b_3d, [config.qk_nope_head_dim, config.v_head_dim], dim=1)
        k_b = k_b.transpose(1, 2)
        writer.add_tensor(f'{b_pfx}attn_k_b.weight', k_b.contiguous().to(torch.float16).numpy())
        writer.add_tensor(f'{b_pfx}attn_v_b.weight', v_b.contiguous().to(torch.float16).numpy())
        wo_prefix = f'{pfx}attn.w_o' if f'{pfx}attn.w_o.weight' in sd or f'{pfx}attn.w_o.qweight' in sd else f'{pfx}attn.wo'
        wo = extract_linear_weight(sd, wo_prefix, n_heads * config.v_head_dim, dim)
        writer.add_tensor(f'{b_pfx}attn_output.weight', wo.to(torch.float16).numpy())
        writer.add_tensor(f'{b_pfx}ffn_norm.weight', sd[f'{pfx}ffn_norm.weight'].to(torch.float32).numpy())
        if i < leading_dense and (f'{pfx}ffn.w_gate.weight' in sd or f'{pfx}ffn.w_gate.qweight' in sd):
            w_gate = extract_linear_weight(sd, f'{pfx}ffn.w_gate', dim, config.intermediate_size)
            w_up = extract_linear_weight(sd, f'{pfx}ffn.w_up', dim, config.intermediate_size)
            w_down = extract_linear_weight(sd, f'{pfx}ffn.w_down', config.intermediate_size, dim)
            writer.add_tensor(f'{b_pfx}ffn_gate.weight', w_gate.to(torch.float16).numpy())
            writer.add_tensor(f'{b_pfx}ffn_up.weight', w_up.to(torch.float16).numpy())
            writer.add_tensor(f'{b_pfx}ffn_down.weight', w_down.to(torch.float16).numpy())
        else:
            r_key = next((k for k in (f'{pfx}ffn.router.weight', f'{pfx}ffn.router_gate.weight') if k in sd), None)
            w_router = sd[r_key]
            writer.add_tensor(f'{b_pfx}ffn_gate_inp.weight', w_router.to(torch.float16).numpy())
            shexp_prefix = f'{pfx}ffn.shared_experts'
            w_sgate = extract_linear_weight(sd, f'{shexp_prefix}.w_gate', dim, config.moe_intermediate_size * config.n_shared_experts)
            w_sup = extract_linear_weight(sd, f'{shexp_prefix}.w_up', dim, config.moe_intermediate_size * config.n_shared_experts)
            w_sdown = extract_linear_weight(sd, f'{shexp_prefix}.w_down', config.moe_intermediate_size * config.n_shared_experts, dim)
            writer.add_tensor(f'{b_pfx}ffn_gate_shexp.weight', w_sgate.to(torch.float16).numpy())
            writer.add_tensor(f'{b_pfx}ffn_up_shexp.weight', w_sup.to(torch.float16).numpy())
            writer.add_tensor(f'{b_pfx}ffn_down_shexp.weight', w_sdown.to(torch.float16).numpy())
            gates, ups, downs = ([], [], [])
            for e in range(config.n_routed_experts):
                g = extract_linear_weight(sd, f'{pfx}ffn.routed_experts.{e}.w_gate', dim, config.moe_intermediate_size)
                u = extract_linear_weight(sd, f'{pfx}ffn.routed_experts.{e}.w_up', dim, config.moe_intermediate_size)
                d = extract_linear_weight(sd, f'{pfx}ffn.routed_experts.{e}.w_down', config.moe_intermediate_size, dim)
                gates.append(g)
                ups.append(u)
                downs.append(d)
            writer.add_tensor(f'{b_pfx}ffn_gate_exps.weight', torch.stack(gates, dim=0).contiguous().to(torch.float16).numpy())
            writer.add_tensor(f'{b_pfx}ffn_up_exps.weight', torch.stack(ups, dim=0).contiguous().to(torch.float16).numpy())
            writer.add_tensor(f'{b_pfx}ffn_down_exps.weight', torch.stack(downs, dim=0).contiguous().to(torch.float16).numpy())
            bias_key = next((k for k in (f'{pfx}ffn.exp_probs_b', f'{pfx}ffn.expert_bias', f'{pfx}ffn.router_bias') if k in sd), None)
            if bias_key:
                writer.add_tensor(f'{b_pfx}exp_probs_b.bias', sd[bias_key].to(torch.float32).numpy())
    print('[Mark GGUF Exporter] Finalizing GGUF container...')
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    verify_gguf_integrity(output_gguf)
    size_mb = output_gguf.stat().st_size / (1024 * 1024)
    print(f'[Mark GGUF Exporter] Success: {output_gguf} ({size_mb:.2f} MB)')
    return output_gguf

def main():
    parser = argparse.ArgumentParser(description='Universal DeepSeek2 GGUF Exporter for Mark Models')
    parser.add_argument('--model', type=str, default='1.5b', choices=['302m', 'slm-v2-302m', 'engineer-1.5b', '1.5b', 'general-1.5b', 'v2-8b', 'v3-1.1b', 'all'])
    parser.add_argument('--ckpt', type=str, default='', help='Path to source .pt checkpoint')
    parser.add_argument('--out', type=str, default='', help='Path to destination .gguf container')
    args = parser.parse_args()
    default_paths = {'302m': (Path('/mnt/ssd_1/mark ai/checkpoints/mark_engineer_302m_iq4_nl.pt'), Path('/mnt/ssd_1/mark ai/checkpoints/mark_engineer_302m.gguf')), 'slm-v2-302m': (Path('/mnt/ssd_1/mark ai/checkpoints/mark_slm_v2_302m.pt'), Path('/mnt/ssd_1/mark ai/checkpoints/mark_slm_v2_302m.gguf')), 'engineer-1.5b': (Path('/mnt/ssd_1/mark ai/checkpoints/mark_engineer_1_5b_iq4_nl.pt'), Path('/mnt/ssd_1/mark ai/checkpoints/mark_engineer_1_5b.gguf')), 'general-1.5b': (Path('/mnt/ssd_1/mark ai/checkpoints/mark_general_1_5b_iq4_nl.pt'), Path('/mnt/ssd_1/mark ai/checkpoints/mark_general_1_5b.gguf')), 'v2-8b': (Path('/mnt/ssd_1/mark ai/checkpoints/mark_engineer_v2_8b/mark_engineer_v2_8b_step_15000.pt'), Path('/mnt/ssd_1/mark ai/checkpoints/mark_engineer_v2_8b/mark_engineer_v2_8b.gguf')), 'v3-1.1b': (Path('/mnt/ssd_1/mark ai/checkpoints/mark_v3_core_1_1b.pt'), Path('/mnt/ssd_1/mark ai/checkpoints/mark_v3_core_1_1b.gguf'))}
    models_to_export = list(default_paths.keys()) if args.model == 'all' else ['engineer-1.5b' if args.model == '1.5b' else args.model]
    for m in models_to_export:
        def_src, def_dst = default_paths.get(m, (Path('checkpoints/dummy.pt'), Path('checkpoints/dummy.gguf')))
        src = Path(args.ckpt) if args.ckpt and args.model != 'all' else def_src
        dst = Path(args.out) if args.out and args.model != 'all' else def_dst
        if not src.exists():
            if args.ckpt:
                raise FileNotFoundError(f'Specified checkpoint not found: {src}')
            print(f'[Mark GGUF Exporter] Checkpoint not found on disk: {src}. Skipping.')
            continue
        export_to_deepseek2_gguf(m, src, dst)
if __name__ == '__main__':
    main()
