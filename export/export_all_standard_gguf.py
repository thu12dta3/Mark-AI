#!/usr/bin/env python3
"""
Mark AI - Universal Production GGUF Exporter for All Mark Models
================================================================
Generates 100% compliant, standard LLaMA GGUF containers with:
- Full BPE Tokenizer embedded (tokenizer.ggml.model, tokens, merges)
- CUDA-aligned vocabulary (padded to multiple of 64 tokens)
- FP32 Layernorms / RMSNorms (guaranteeing GGML binary_op type parity)
- Plug-and-play compatibility across Ollama, LM Studio, llama.cpp, vLLM

Strict Constraint: Zero emojis across all output and logs.
"""

import sys
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
import numpy as np
import gguf

from src.model.mark_arch import MarkConfig, IQ4_NL_LOOKUP_VALUES

lut = torch.tensor(IQ4_NL_LOOKUP_VALUES, dtype=torch.float32)


def unpack_tensor(val, lut=lut) -> torch.Tensor | None:
    if isinstance(val, dict) and "packed" in val:
        packed = val["packed"]
        scales = val["scales"].float().unsqueeze(-1)
        orig_shape = val["orig_shape"]
        low = (packed & 0x0F).long()
        high = ((packed >> 4) & 0x0F).long()
        idx = torch.empty((packed.shape[0], 32), dtype=torch.long)
        idx[:, 0::2] = low
        idx[:, 1::2] = high
        norm_w = lut[idx]
        w = (norm_w * scales).reshape(orig_shape)
        return w
    elif isinstance(val, torch.Tensor):
        return val
    return None


def dequant_linear_layer(sd, prefix: str, in_features: int, out_features: int, block_size: int = 32) -> torch.Tensor:
    # Handles both dict format and separated .qweight / .scales format
    if f"{prefix}.weight" in sd:
        v = sd[f"{prefix}.weight"]
        res = unpack_tensor(v)
        if res is not None:
            return res

    if prefix in sd:
        res = unpack_tensor(sd[prefix])
        if res is not None:
            return res

    qw_key = f"{prefix}.qweight"
    sc_key = f"{prefix}.scales"
    if qw_key in sd and sc_key in sd:
        qw = sd[qw_key]
        sc = sd[sc_key]
        low = (qw & 0x0F).long()
        high = ((qw >> 4) & 0x0F).long()
        idx = torch.empty((out_features, qw.shape[1], block_size), dtype=torch.long)
        idx[..., 0::2] = low
        idx[..., 1::2] = high
        norm_w = lut[idx]
        w_blocks = norm_w * sc.float()
        return w_blocks.view(out_features, -1)[:, :in_features]

    raise KeyError(f"Unable to find tensor for prefix {prefix}")


def export_model_gguf(
    model_type: str,
    ckpt_path: Path,
    output_gguf: Path,
    model_name: str,
):
    print(f"\n[GGUF Exporter] === Exporting {model_name} ({model_type}) ===")
    print(f"[GGUF Exporter] Source: {ckpt_path}")
    print(f"[GGUF Exporter] Destination: {output_gguf}")

    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if model_type == "302m":
        config = MarkConfig.create_engineer_300m()
    elif model_type in ("1.5b", "engineer-1.5b"):
        config = MarkConfig.create_engineer_1_5b()
    elif model_type == "general-1.5b":
        config = MarkConfig.create_general_1_5b()
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    output_gguf.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(str(output_gguf), "llama")

    # Architecture metadata
    writer.add_name(model_name)
    writer.add_author("thu12dta3")
    writer.add_version("1.0.0")
    writer.add_description(f"{model_name} - Fully Compliant Production GGUF Container")
    writer.add_context_length(config.max_seq_len)
    writer.add_embedding_length(config.dim)
    writer.add_block_count(config.n_layers)
    writer.add_feed_forward_length(config.intermediate_size)
    writer.add_head_count(config.n_heads)
    writer.add_head_count_kv(config.n_heads)
    writer.add_layer_norm_rms_eps(config.rms_norm_eps)
    writer.add_rope_freq_base(config.rope_theta)

    writer.add_bos_token_id(config.bos_token_id)
    writer.add_eos_token_id(config.eos_token_id)
    writer.add_pad_token_id(config.pad_token_id)

    # Tokenizer embedding
    tok_candidates = [
        REPO_ROOT / "models" / "mark-engineer-1.5b" / "tokenizer.json",
        REPO_ROOT / "models" / "mark-vl-2.2b" / "tokenizer.json",
    ]
    tok_path = next((p for p in tok_candidates if p.exists()), None)
    if not tok_path:
        raise FileNotFoundError("BPE tokenizer.json not found in repository models/")

    print(f"[GGUF Exporter] Embedding tokenizer from {tok_path}...")
    with open(tok_path, "r", encoding="utf-8") as f:
        tok_data = json.load(f)
    vocab = tok_data.get("model", {}).get("vocab", {})
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    tokens = [k.encode("utf-8") for k, v in sorted_vocab]
    merges = tok_data.get("model", {}).get("merges", [])
    formatted_merges = [" ".join(m).encode("utf-8") if isinstance(m, list) else m.encode("utf-8") for m in merges]

    # CUDA vector alignment padding (multiple of 64)
    pad_vocab_size = (len(tokens) + 63) // 64 * 64
    diff = pad_vocab_size - len(tokens)
    for k in range(diff):
        tokens.append(f"<pad_align_{k}>".encode("utf-8"))

    writer.add_tokenizer_model("gpt2")
    writer.add_token_list(tokens)
    writer.add_token_merges(formatted_merges)
    writer.add_token_scores([0.0] * len(tokens))
    writer.add_token_types([1] * len(tokens))

    chat_template = (
        "{% for message in messages %}"
        "{% if message['role'] == 'user' %}"
        "User: {{ message['content'] }}\n"
        "{% elif message['role'] == 'assistant' %}"
        "Assistant: {{ message['content'] }}\n"
        "{% endif %}"
        "{% endfor %}"
        "Assistant: "
    )
    writer.add_chat_template(chat_template)

    # Embedding and output norm
    emb_raw = unpack_tensor(sd["tok_embeddings.weight"])
    if emb_raw is None:
        emb_raw = dequant_linear_layer(sd, "tok_embeddings", config.dim, config.vocab_size)
    w_emb = emb_raw.to(torch.float16)
    if w_emb.shape[0] < pad_vocab_size:
        pad = torch.zeros((pad_vocab_size - w_emb.shape[0], w_emb.shape[1]), dtype=torch.float16)
        w_emb = torch.cat([w_emb, pad], dim=0)
    writer.add_tensor("token_embd.weight", w_emb.numpy())

    # Output Norm strictly FP32
    norm_raw = unpack_tensor(sd["norm.weight"])
    writer.add_tensor("output_norm.weight", norm_raw.to(torch.float32).numpy())

    # LM Head
    if "lm_head.weight" in sd:
        out_raw = unpack_tensor(sd["lm_head.weight"])
    else:
        out_raw = emb_raw
    w_out = out_raw.to(torch.float16)
    if w_out.shape[0] < pad_vocab_size:
        pad = torch.zeros((pad_vocab_size - w_out.shape[0], w_out.shape[1]), dtype=torch.float16)
        w_out = torch.cat([w_out, pad], dim=0)
    writer.add_tensor("output.weight", w_out.numpy())

    head_dim = config.dim // config.n_heads

    for i in range(config.n_layers):
        pfx = f"layers.{i}."
        b_pfx = f"blk.{i}."

        # Attention Norm (FP32)
        attn_norm = unpack_tensor(sd[f"{pfx}attn_norm.weight"])
        writer.add_tensor(f"{b_pfx}attn_norm.weight", attn_norm.to(torch.float32).numpy())

        # 1. Fold Wq
        wq_a = dequant_linear_layer(sd, f"{pfx}attn.wq_a", config.dim, config.q_lora_rank)
        wq_b = dequant_linear_layer(sd, f"{pfx}attn.wq_b", config.q_lora_rank, config.n_heads * config.qk_head_dim)
        q_norm = unpack_tensor(sd[f"{pfx}attn.q_norm.weight"])
        wq_scaled = q_norm.unsqueeze(1) * wq_a
        wq_full = torch.matmul(wq_b, wq_scaled)
        wq_dense = wq_full.view(config.n_heads, config.qk_head_dim, config.dim)[:, :head_dim, :].reshape(config.dim, config.dim)
        writer.add_tensor(f"{b_pfx}attn_q.weight", wq_dense.to(torch.float16).numpy())

        # 2. Fold Wk and Wv
        wkv_a = dequant_linear_layer(sd, f"{pfx}attn.wkv_a", config.dim, config.kv_lora_rank + config.qk_rope_head_dim)
        kv_norm = unpack_tensor(sd[f"{pfx}attn.kv_norm.weight"])
        kv_b_out_dim = config.n_heads * (config.qk_nope_head_dim + (config.v_head_dim if hasattr(config, "v_head_dim") and config.v_head_dim != 0 else head_dim))
        wkv_b = dequant_linear_layer(sd, f"{pfx}attn.wkv_b", config.kv_lora_rank, kv_b_out_dim)

        w_ckv = wkv_a[:config.kv_lora_rank]
        w_ckv_scaled = kv_norm.unsqueeze(1) * w_ckv

        v_dim_head = config.v_head_dim if hasattr(config, "v_head_dim") and config.v_head_dim != 0 else head_dim
        kv_b_3d = wkv_b.view(config.n_heads, config.qk_nope_head_dim + v_dim_head, config.kv_lora_rank)
        k_b = kv_b_3d[:, :config.qk_nope_head_dim, :]
        v_b = kv_b_3d[:, config.qk_nope_head_dim:, :]

        wk_nope = torch.matmul(k_b, w_ckv_scaled)
        wk_pe = wkv_a[config.kv_lora_rank:config.kv_lora_rank + 16].unsqueeze(0).expand(config.n_heads, -1, -1)
        if head_dim > config.qk_nope_head_dim:
            extra = head_dim - config.qk_nope_head_dim
            wk_dense = torch.cat([wk_nope, wk_pe[:, :extra, :]], dim=1).reshape(config.dim, config.dim)
        else:
            wk_dense = wk_nope[:, :head_dim, :].reshape(config.dim, config.dim)
        writer.add_tensor(f"{b_pfx}attn_k.weight", wk_dense.to(torch.float16).numpy())

        wv_dense = torch.matmul(v_b, w_ckv_scaled)[:, :head_dim, :].reshape(config.dim, config.dim)
        writer.add_tensor(f"{b_pfx}attn_v.weight", wv_dense.to(torch.float16).numpy())

        # 3. Fold Wo
        wo = dequant_linear_layer(sd, f"{pfx}attn.wo", config.n_heads * v_dim_head, config.dim)
        if wo.shape[1] > config.dim:
            wo_dense = wo.view(config.dim, config.n_heads, v_dim_head)[:, :, :head_dim].reshape(config.dim, config.dim)
        else:
            wo_dense = wo
        writer.add_tensor(f"{b_pfx}attn_output.weight", wo_dense.to(torch.float16).numpy())

        # FFN Norm (FP32)
        ffn_norm = unpack_tensor(sd[f"{pfx}ffn_norm.weight"])
        writer.add_tensor(f"{b_pfx}ffn_norm.weight", ffn_norm.to(torch.float32).numpy())

        # 4. FFN Projections
        if f"{pfx}ffn.w_gate.weight" in sd or f"{pfx}ffn.w_gate.qweight" in sd:
            # Dense SwiGLU block
            w_gate = dequant_linear_layer(sd, f"{pfx}ffn.w_gate", config.dim, config.intermediate_size)
            w_up = dequant_linear_layer(sd, f"{pfx}ffn.w_up", config.dim, config.intermediate_size)
            w_down = dequant_linear_layer(sd, f"{pfx}ffn.w_down", config.intermediate_size, config.dim)
            writer.add_tensor(f"{b_pfx}ffn_gate.weight", w_gate.to(torch.float16).numpy())
            writer.add_tensor(f"{b_pfx}ffn_up.weight", w_up.to(torch.float16).numpy())
            writer.add_tensor(f"{b_pfx}ffn_down.weight", w_down.to(torch.float16).numpy())
        else:
            # MoE folded into dense SwiGLU
            w_sgate = dequant_linear_layer(sd, f"{pfx}ffn.shared_experts.w_gate", config.dim, config.moe_intermediate_size)
            w_sup = dequant_linear_layer(sd, f"{pfx}ffn.shared_experts.w_up", config.dim, config.moe_intermediate_size)
            w_sdown = dequant_linear_layer(sd, f"{pfx}ffn.shared_experts.w_down", config.moe_intermediate_size, config.dim)

            r_gates = [dequant_linear_layer(sd, f"{pfx}ffn.routed_experts.{e}.w_gate", config.dim, config.moe_intermediate_size) for e in range(config.n_routed_experts)]
            r_ups = [dequant_linear_layer(sd, f"{pfx}ffn.routed_experts.{e}.w_up", config.dim, config.moe_intermediate_size) for e in range(config.n_routed_experts)]
            r_downs = [dequant_linear_layer(sd, f"{pfx}ffn.routed_experts.{e}.w_down", config.moe_intermediate_size, config.dim) for e in range(config.n_routed_experts)]

            avg_gate = torch.stack(r_gates, dim=0).mean(dim=0)
            avg_up = torch.stack(r_ups, dim=0).mean(dim=0)
            avg_down = torch.stack(r_downs, dim=0).mean(dim=0)

            rem = config.intermediate_size - (w_sgate.shape[0] + avg_gate.shape[0])
            if rem > 0:
                w_gate_dense = torch.cat([w_sgate, avg_gate, avg_gate[:rem]], dim=0)
                w_up_dense = torch.cat([w_sup, avg_up, avg_up[:rem]], dim=0)
                w_down_dense = torch.cat([w_sdown, avg_down, avg_down[:, :rem]], dim=1)
            else:
                w_gate_dense = torch.cat([w_sgate, avg_gate], dim=0)[:config.intermediate_size]
                w_up_dense = torch.cat([w_sup, avg_up], dim=0)[:config.intermediate_size]
                w_down_dense = torch.cat([w_sdown, avg_down], dim=1)[:, :config.intermediate_size]

            writer.add_tensor(f"{b_pfx}ffn_gate.weight", w_gate_dense.to(torch.float16).numpy())
            writer.add_tensor(f"{b_pfx}ffn_up.weight", w_up_dense.to(torch.float16).numpy())
            writer.add_tensor(f"{b_pfx}ffn_down.weight", w_down_dense.to(torch.float16).numpy())

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_mb = output_gguf.stat().st_size / (1024 * 1024)
    print(f"[GGUF Exporter] Successfully exported {output_gguf} ({size_mb:.2f} MB)")
    return output_gguf


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Export compliant standard GGUF for Mark models.")
    parser.add_argument("--model", type=str, default="302m", choices=["302m", "1.5b", "general-1.5b", "all"])
    args = parser.parse_args()

    targets = {
        "302m": (
            "302m",
            Path("/mnt/ssd_1/mark ai/checkpoints/mark_engineer_302m_iq4_nl.pt"),
            Path("/mnt/ssd_1/mark ai/checkpoints/mark_engineer_302m_iq4_nl.gguf"),
            "Mark Engineer 300M"
        ),
        "1.5b": (
            "1.5b",
            Path("/mnt/ssd_1/mark ai/checkpoints/mark_engineer_1_5b_iq4_nl.pt"),
            Path("/mnt/ssd_1/mark ai/checkpoints/mark_engineer_1_5b_iq4_nl.gguf"),
            "Mark Engineer 1.5B"
        ),
        "general-1.5b": (
            "general-1.5b",
            Path("/mnt/ssd_1/mark ai/checkpoints/mark_general_1_5b_iq4_nl.pt"),
            Path("/mnt/ssd_1/mark ai/checkpoints/mark_general_1_5b_iq4_nl.gguf"),
            "Mark General 1.5B"
        ),
    }

    selected = list(targets.keys()) if args.model == "all" else [args.model]
    for key in selected:
        m_type, src, dst, name = targets[key]
        if src.exists():
            export_model_gguf(m_type, src, dst, name)
        else:
            print(f"[Warning] Source checkpoint not found: {src}")
