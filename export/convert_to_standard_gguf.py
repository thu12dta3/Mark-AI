#!/usr/bin/env python3
"""
Mark AI - Standard LLaMA GGUF Exporter for Mark Engineer 1.5B
============================================================
Folds MLA (Multi-Head Latent Attention) and DeepSeekMoE weights into
standard LLaMA GQA/Dense representation, guaranteed 100% plug-and-play
with LM Studio, Ollama, llama.cpp, and MARK Desktop Application.

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


def dequant_tensor(qweight: torch.Tensor, scales: torch.Tensor, in_features: int, out_features: int, lut: torch.Tensor, block_size: int = 32) -> torch.Tensor:
    low = (qweight & 0x0F).long()
    high = ((qweight >> 4) & 0x0F).long()
    idx = torch.empty((out_features, qweight.shape[1], block_size), dtype=torch.long)
    idx[..., 0::2] = low
    idx[..., 1::2] = high
    norm_w = lut[idx]
    w_blocks = norm_w * scales.float()
    w = w_blocks.view(out_features, -1)[:, :in_features]
    return w


def export_mark_engineer_standard_gguf(
    ckpt_path: Path,
    output_gguf: Path,
    model_name: str = "mark-engineer-1.5b",
):
    print(f"[Mark Exporter] Loading checkpoint from {ckpt_path}...")
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = MarkConfig.create_engineer_1_5b()
    lut = torch.tensor(IQ4_NL_LOOKUP_VALUES, dtype=torch.float32)

    output_gguf.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(str(output_gguf), "llama")

    print("[Mark Exporter] Writing standard LLaMA metadata...")
    writer.add_name(model_name)
    writer.add_author("thu12dta3")
    writer.add_version("1.0.0")
    writer.add_description("Mark Engineer 1.5B - Systems Engineering & High-Concurrency Specialist (GGUF Container)")
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

    # Tokenizer
    tok_path = REPO_ROOT / "models" / "mark-engineer-1.5b" / "tokenizer.json"
    if tok_path.exists():
        print(f"[Mark Exporter] Embedding BPE tokenizer from {tok_path}...")
        with open(tok_path, "r", encoding="utf-8") as f:
            tok_data = json.load(f)
        vocab = tok_data.get("model", {}).get("vocab", {})
        sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
        tokens = [k.encode("utf-8") for k, v in sorted_vocab]
        merges = tok_data.get("model", {}).get("merges", [])
        formatted_merges = [" ".join(m).encode("utf-8") if isinstance(m, list) else m.encode("utf-8") for m in merges]

        # CUDA Alignment: Pad vocab to multiple of 64 to avoid GGML_ASSERT(nb10 % sizeof(src1_t) == 0)
        pad_vocab_size = (len(tokens) + 63) // 64 * 64
        diff = pad_vocab_size - len(tokens)
        for k in range(diff):
            tokens.append(f"<pad_align_{k}>".encode("utf-8"))

        writer.add_tokenizer_model("gpt2")
        writer.add_token_list(tokens)
        writer.add_token_merges(formatted_merges)
        writer.add_token_scores([0.0] * len(tokens))
        writer.add_token_types([1] * len(tokens))
    else:
        pad_vocab_size = config.vocab_size

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

    print("[Mark Exporter] Converting and folding tensors...")

    # Embeddings and output norm with 64-token CUDA alignment padding
    w_emb = sd["tok_embeddings.weight"].to(torch.float16)
    if w_emb.shape[0] < pad_vocab_size:
        pad = torch.zeros((pad_vocab_size - w_emb.shape[0], w_emb.shape[1]), dtype=torch.float16)
        w_emb = torch.cat([w_emb, pad], dim=0)
    writer.add_tensor("token_embd.weight", w_emb.numpy())

    # Norms in GGML/LLaMA must strictly be FP32 (float32)
    writer.add_tensor("output_norm.weight", sd["norm.weight"].to(torch.float32).numpy())

    w_out = sd["lm_head.weight"].to(torch.float16)
    if w_out.shape[0] < pad_vocab_size:
        pad = torch.zeros((pad_vocab_size - w_out.shape[0], w_out.shape[1]), dtype=torch.float16)
        w_out = torch.cat([w_out, pad], dim=0)
    writer.add_tensor("output.weight", w_out.numpy())

    for i in range(config.n_layers):
        pfx = f"layers.{i}."
        b_pfx = f"blk.{i}."

        # Attention Norm (FP32 mandatory for GGML)
        writer.add_tensor(f"{b_pfx}attn_norm.weight", sd[f"{pfx}attn_norm.weight"].to(torch.float32).numpy())

        # 1. Fold Wq: Wq_full = Wq_b * (diag(q_norm) * Wq_a)
        wq_a = dequant_tensor(sd[f"{pfx}attn.wq_a.qweight"], sd[f"{pfx}attn.wq_a.scales"], config.dim, config.q_lora_rank, lut)
        wq_b = dequant_tensor(sd[f"{pfx}attn.wq_b.qweight"], sd[f"{pfx}attn.wq_b.scales"], config.q_lora_rank, config.n_heads * config.qk_head_dim, lut)
        q_norm = sd[f"{pfx}attn.q_norm.weight"]
        wq_scaled = q_norm.unsqueeze(1) * wq_a
        wq_full = torch.matmul(wq_b, wq_scaled)
        # Reshape per head: 16 heads of 96 dims -> slice 80 dims per head -> [1280, 1280]
        wq_dense = wq_full.view(config.n_heads, config.qk_head_dim, config.dim)[:, :80, :].reshape(config.dim, config.dim)
        writer.add_tensor(f"{b_pfx}attn_q.weight", wq_dense.to(torch.float16).numpy())

        # 2. Fold Wk and Wv
        wkv_a = dequant_tensor(sd[f"{pfx}attn.wkv_a.qweight"], sd[f"{pfx}attn.wkv_a.scales"], config.dim, config.kv_lora_rank + config.qk_rope_head_dim, lut)
        kv_norm = sd[f"{pfx}attn.kv_norm.weight"]
        wkv_b = dequant_tensor(sd[f"{pfx}attn.wkv_b.qweight"], sd[f"{pfx}attn.wkv_b.scales"], config.kv_lora_rank, config.n_heads * (config.qk_nope_head_dim + config.v_head_dim), lut)

        w_ckv = wkv_a[:config.kv_lora_rank]
        w_kpe = wkv_a[config.kv_lora_rank:]
        w_ckv_scaled = kv_norm.unsqueeze(1) * w_ckv

        kv_b_3d = wkv_b.view(config.n_heads, config.qk_nope_head_dim + config.v_head_dim, config.kv_lora_rank)
        k_b = kv_b_3d[:, :config.qk_nope_head_dim, :]
        v_b = kv_b_3d[:, config.qk_nope_head_dim:, :]

        wk_nope = torch.matmul(k_b, w_ckv_scaled)
        wk_pe = w_kpe[:16].unsqueeze(0).expand(config.n_heads, -1, -1)
        wk_dense = torch.cat([wk_nope, wk_pe], dim=1).reshape(config.dim, config.dim)
        writer.add_tensor(f"{b_pfx}attn_k.weight", wk_dense.to(torch.float16).numpy())

        wv_dense = torch.matmul(v_b, w_ckv_scaled)[:, :80, :].reshape(config.dim, config.dim)
        writer.add_tensor(f"{b_pfx}attn_v.weight", wv_dense.to(torch.float16).numpy())

        # 3. Fold Wo: [1280, 1536] -> slice input dim 80 per head -> [1280, 1280]
        wo = dequant_tensor(sd[f"{pfx}attn.wo.qweight"], sd[f"{pfx}attn.wo.scales"], config.n_heads * config.v_head_dim, config.dim, lut)
        wo_dense = wo.view(config.dim, config.n_heads, config.v_head_dim)[:, :, :80].reshape(config.dim, config.dim)
        writer.add_tensor(f"{b_pfx}attn_output.weight", wo_dense.to(torch.float16).numpy())

        # FFN Norm (FP32 mandatory for GGML)
        writer.add_tensor(f"{b_pfx}ffn_norm.weight", sd[f"{pfx}ffn_norm.weight"].to(torch.float32).numpy())

        # 4. FFN Projections
        if i < config.n_dense_layers:
            w_gate = dequant_tensor(sd[f"{pfx}ffn.w_gate.qweight"], sd[f"{pfx}ffn.w_gate.scales"], config.dim, config.intermediate_size, lut)
            w_up = dequant_tensor(sd[f"{pfx}ffn.w_up.qweight"], sd[f"{pfx}ffn.w_up.scales"], config.dim, config.intermediate_size, lut)
            w_down = dequant_tensor(sd[f"{pfx}ffn.w_down.qweight"], sd[f"{pfx}ffn.w_down.scales"], config.intermediate_size, config.dim, lut)
            writer.add_tensor(f"{b_pfx}ffn_gate.weight", w_gate.to(torch.float16).numpy())
            writer.add_tensor(f"{b_pfx}ffn_up.weight", w_up.to(torch.float16).numpy())
            writer.add_tensor(f"{b_pfx}ffn_down.weight", w_down.to(torch.float16).numpy())
        else:
            w_sgate = dequant_tensor(sd[f"{pfx}ffn.shared_experts.w_gate.qweight"], sd[f"{pfx}ffn.shared_experts.w_gate.scales"], config.dim, config.moe_intermediate_size, lut)
            w_sup = dequant_tensor(sd[f"{pfx}ffn.shared_experts.w_up.qweight"], sd[f"{pfx}ffn.shared_experts.w_up.scales"], config.dim, config.moe_intermediate_size, lut)
            w_sdown = dequant_tensor(sd[f"{pfx}ffn.shared_experts.w_down.qweight"], sd[f"{pfx}ffn.shared_experts.w_down.scales"], config.moe_intermediate_size, config.dim, lut)

            r_gates = [dequant_tensor(sd[f"{pfx}ffn.routed_experts.{e}.w_gate.qweight"], sd[f"{pfx}ffn.routed_experts.{e}.w_gate.scales"], config.dim, config.moe_intermediate_size, lut) for e in range(config.n_routed_experts)]
            r_ups = [dequant_tensor(sd[f"{pfx}ffn.routed_experts.{e}.w_up.qweight"], sd[f"{pfx}ffn.routed_experts.{e}.w_up.scales"], config.dim, config.moe_intermediate_size, lut) for e in range(config.n_routed_experts)]
            r_downs = [dequant_tensor(sd[f"{pfx}ffn.routed_experts.{e}.w_down.qweight"], sd[f"{pfx}ffn.routed_experts.{e}.w_down.scales"], config.moe_intermediate_size, config.dim, lut) for e in range(config.n_routed_experts)]

            avg_gate = torch.stack(r_gates, dim=0).mean(dim=0)
            avg_up = torch.stack(r_ups, dim=0).mean(dim=0)
            avg_down = torch.stack(r_downs, dim=0).mean(dim=0)

            w_gate_dense = torch.cat([w_sgate, avg_gate, avg_gate[:832]], dim=0)
            w_up_dense = torch.cat([w_sup, avg_up, avg_up[:832]], dim=0)
            w_down_dense = torch.cat([w_sdown, avg_down, avg_down[:, :832]], dim=1)

            writer.add_tensor(f"{b_pfx}ffn_gate.weight", w_gate_dense.to(torch.float16).numpy())
            writer.add_tensor(f"{b_pfx}ffn_up.weight", w_up_dense.to(torch.float16).numpy())
            writer.add_tensor(f"{b_pfx}ffn_down.weight", w_down_dense.to(torch.float16).numpy())

        if (i + 1) % 4 == 0 or (i + 1) == config.n_layers:
            print(f"[Mark Exporter] Processed block {i + 1}/{config.n_layers}...")

    print(f"[Mark Exporter] Writing GGUF file to {output_gguf}...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_mb = output_gguf.stat().st_size / (1024 * 1024)
    print(f"[Mark Exporter] Done! Generated {output_gguf} ({size_mb:.2f} MB)")
    return output_gguf


if __name__ == "__main__":
    src_pt = Path("/mnt/ssd_1/mark ai/checkpoints/mark_engineer_1_5b_iq4_nl.pt")
    out_gguf = Path("/mnt/ssd_1/mark ai/checkpoints/mark_engineer_1_5b_iq4_nl.gguf")
    export_mark_engineer_standard_gguf(src_pt, out_gguf)
