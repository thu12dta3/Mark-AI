#!/usr/bin/env python3
"""
Mark AI - DeepSeek2 GGUF Exporter for Mark Engineer 1.5B
========================================================
Exports Mark Engineer 1.5B (MLA + DeepSeekMoE) directly into a DeepSeek2
GGUF container natively compatible with Ollama (0.40+), llama.cpp, and LM Studio.

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


def convert_mark_engineer_to_gguf(
    ckpt_path: Path,
    output_gguf: Path,
    model_name: str = "mark-engineer-1.5b",
):
    print(f"[Mark GGUF Exporter] Loading checkpoint from {ckpt_path}...")
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = MarkConfig.create_engineer_1_5b()
    lut = torch.tensor(IQ4_NL_LOOKUP_VALUES, dtype=torch.float32)

    output_gguf.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(str(output_gguf), "deepseek2")

    print("[Mark GGUF Exporter] Adding DeepSeek2 architecture metadata...")
    writer.add_name(model_name)
    writer.add_author("thu12dta3")
    writer.add_version("1.0.0")
    writer.add_description("Mark Engineer 1.5B - MLA + DeepSeekMoE + MTP Native GGUF Container")
    writer.add_context_length(config.max_seq_len)
    writer.add_embedding_length(config.dim)
    writer.add_block_count(config.n_layers)
    writer.add_feed_forward_length(config.intermediate_size)
    writer.add_head_count(config.n_heads)
    writer.add_head_count_kv(1)  # MLA MQA compressed representation
    writer.add_layer_norm_rms_eps(config.rms_norm_eps)
    writer.add_rope_freq_base(config.rope_theta)

    writer.add_leading_dense_block_count(config.n_dense_layers)
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

    # Tokenizer
    tok_path = REPO_ROOT / "models" / "mark-engineer-1.5b" / "tokenizer.json"
    if tok_path.exists():
        print(f"[Mark GGUF Exporter] Embedding BPE tokenizer from {tok_path}...")
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

    print("[Mark GGUF Exporter] Dequantizing and mapping tensors to DeepSeek2 format...")

    # Embeddings and output norm with 64-token CUDA alignment padding
    w_emb = sd["tok_embeddings.weight"].to(torch.float16)
    if w_emb.shape[0] < pad_vocab_size:
        pad = torch.zeros((pad_vocab_size - w_emb.shape[0], w_emb.shape[1]), dtype=torch.float16)
        w_emb = torch.cat([w_emb, pad], dim=0)
    writer.add_tensor("token_embd.weight", w_emb.numpy())

    writer.add_tensor("output_norm.weight", sd["norm.weight"].to(torch.float16).numpy())

    w_out = sd["lm_head.weight"].to(torch.float16)
    if w_out.shape[0] < pad_vocab_size:
        pad = torch.zeros((pad_vocab_size - w_out.shape[0], w_out.shape[1]), dtype=torch.float16)
        w_out = torch.cat([w_out, pad], dim=0)
    writer.add_tensor("output.weight", w_out.numpy())

    for i in range(config.n_layers):
        pfx = f"layers.{i}."
        b_pfx = f"blk.{i}."

        # Attention Norm
        writer.add_tensor(f"{b_pfx}attn_norm.weight", sd[f"{pfx}attn_norm.weight"].to(torch.float16).numpy())

        # MLA Attention Projections
        wq_a = dequant_tensor(sd[f"{pfx}attn.wq_a.qweight"], sd[f"{pfx}attn.wq_a.scales"], config.dim, config.q_lora_rank, lut)
        writer.add_tensor(f"{b_pfx}attn_q_a.weight", wq_a.to(torch.float16).numpy())

        q_a_norm = sd[f"{pfx}attn.q_norm.weight"].to(torch.float16).numpy()
        writer.add_tensor(f"{b_pfx}attn_q_a_norm.weight", q_a_norm)

        wq_b = dequant_tensor(sd[f"{pfx}attn.wq_b.qweight"], sd[f"{pfx}attn.wq_b.scales"], config.q_lora_rank, config.n_heads * config.qk_head_dim, lut)
        writer.add_tensor(f"{b_pfx}attn_q_b.weight", wq_b.to(torch.float16).numpy())

        wkv_a = dequant_tensor(sd[f"{pfx}attn.wkv_a.qweight"], sd[f"{pfx}attn.wkv_a.scales"], config.dim, config.kv_lora_rank + config.qk_rope_head_dim, lut)
        writer.add_tensor(f"{b_pfx}attn_kv_a_mqa.weight", wkv_a.to(torch.float16).numpy())

        kv_a_norm = sd[f"{pfx}attn.kv_norm.weight"].to(torch.float16).numpy()
        writer.add_tensor(f"{b_pfx}attn_kv_a_norm.weight", kv_a_norm)

        # Decompose wkv_b into k_b and v_b
        wkv_b = dequant_tensor(sd[f"{pfx}attn.wkv_b.qweight"], sd[f"{pfx}attn.wkv_b.scales"], config.kv_lora_rank, config.n_heads * (config.qk_nope_head_dim + config.v_head_dim), lut)
        # Shape: [n_heads, qk_nope + v_head, kv_lora_rank]
        kv_b_3d = wkv_b.view(config.n_heads, config.qk_nope_head_dim + config.v_head_dim, config.kv_lora_rank)
        k_b, v_b = torch.split(kv_b_3d, [config.qk_nope_head_dim, config.v_head_dim], dim=1)
        k_b = k_b.transpose(1, 2)  # [n_heads, kv_lora_rank, qk_nope_head_dim]
        writer.add_tensor(f"{b_pfx}attn_k_b.weight", k_b.contiguous().to(torch.float16).numpy())
        writer.add_tensor(f"{b_pfx}attn_v_b.weight", v_b.contiguous().to(torch.float16).numpy())

        # Attention Output Projection
        wo = dequant_tensor(sd[f"{pfx}attn.wo.qweight"], sd[f"{pfx}attn.wo.scales"], config.n_heads * config.v_head_dim, config.dim, lut)
        writer.add_tensor(f"{b_pfx}attn_output.weight", wo.to(torch.float16).numpy())

        # FFN Norm
        writer.add_tensor(f"{b_pfx}ffn_norm.weight", sd[f"{pfx}ffn_norm.weight"].to(torch.float16).numpy())

        if i < config.n_dense_layers:
            # Dense SwiGLU FFN
            w_gate = dequant_tensor(sd[f"{pfx}ffn.w_gate.qweight"], sd[f"{pfx}ffn.w_gate.scales"], config.dim, config.intermediate_size, lut)
            w_up = dequant_tensor(sd[f"{pfx}ffn.w_up.qweight"], sd[f"{pfx}ffn.w_up.scales"], config.dim, config.intermediate_size, lut)
            w_down = dequant_tensor(sd[f"{pfx}ffn.w_down.qweight"], sd[f"{pfx}ffn.w_down.scales"], config.intermediate_size, config.dim, lut)
            writer.add_tensor(f"{b_pfx}ffn_gate.weight", w_gate.to(torch.float16).numpy())
            writer.add_tensor(f"{b_pfx}ffn_up.weight", w_up.to(torch.float16).numpy())
            writer.add_tensor(f"{b_pfx}ffn_down.weight", w_down.to(torch.float16).numpy())
        else:
            # DeepSeekMoE
            # Router Gate
            w_router = sd[f"{pfx}ffn.router.weight"]
            writer.add_tensor(f"{b_pfx}ffn_gate_inp.weight", w_router.to(torch.float16).numpy())

            # Shared Expert
            w_sgate = dequant_tensor(sd[f"{pfx}ffn.shared_experts.w_gate.qweight"], sd[f"{pfx}ffn.shared_experts.w_gate.scales"], config.dim, config.moe_intermediate_size, lut)
            w_sup = dequant_tensor(sd[f"{pfx}ffn.shared_experts.w_up.qweight"], sd[f"{pfx}ffn.shared_experts.w_up.scales"], config.dim, config.moe_intermediate_size, lut)
            w_sdown = dequant_tensor(sd[f"{pfx}ffn.shared_experts.w_down.qweight"], sd[f"{pfx}ffn.shared_experts.w_down.scales"], config.moe_intermediate_size, config.dim, lut)
            writer.add_tensor(f"{b_pfx}ffn_gate_shexp.weight", w_sgate.to(torch.float16).numpy())
            writer.add_tensor(f"{b_pfx}ffn_up_shexp.weight", w_sup.to(torch.float16).numpy())
            writer.add_tensor(f"{b_pfx}ffn_down_shexp.weight", w_sdown.to(torch.float16).numpy())

            # Routed Experts (stacked 3D tensors: [n_experts, intermediate, dim])
            gates, ups, downs = [], [], []
            for e in range(config.n_routed_experts):
                g = dequant_tensor(sd[f"{pfx}ffn.routed_experts.{e}.w_gate.qweight"], sd[f"{pfx}ffn.routed_experts.{e}.w_gate.scales"], config.dim, config.moe_intermediate_size, lut)
                u = dequant_tensor(sd[f"{pfx}ffn.routed_experts.{e}.w_up.qweight"], sd[f"{pfx}ffn.routed_experts.{e}.w_up.scales"], config.dim, config.moe_intermediate_size, lut)
                d = dequant_tensor(sd[f"{pfx}ffn.routed_experts.{e}.w_down.qweight"], sd[f"{pfx}ffn.routed_experts.{e}.w_down.scales"], config.moe_intermediate_size, config.dim, lut)
                gates.append(g)
                ups.append(u)
                downs.append(d)

            w_gate_exps = torch.stack(gates, dim=0)
            w_up_exps = torch.stack(ups, dim=0)
            w_down_exps = torch.stack(downs, dim=0)

            writer.add_tensor(f"{b_pfx}ffn_gate_exps.weight", w_gate_exps.contiguous().to(torch.float16).numpy())
            writer.add_tensor(f"{b_pfx}ffn_up_exps.weight", w_up_exps.contiguous().to(torch.float16).numpy())
            writer.add_tensor(f"{b_pfx}ffn_down_exps.weight", w_down_exps.contiguous().to(torch.float16).numpy())

        if (i + 1) % 4 == 0 or (i + 1) == config.n_layers:
            print(f"[Mark GGUF Exporter] Processed block {i + 1}/{config.n_layers}...")

    print(f"[Mark GGUF Exporter] Writing GGUF file to {output_gguf}...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_mb = output_gguf.stat().st_size / (1024 * 1024)
    print(f"[Mark GGUF Exporter] Done! Generated {output_gguf} ({size_mb:.2f} MB)")
    return output_gguf


if __name__ == "__main__":
    src_pt = Path("/mnt/ssd_1/mark ai/checkpoints/mark_engineer_1_5b_iq4_nl.pt")
    out_gguf = Path("/mnt/ssd_1/mark ai/checkpoints/mark_engineer_1_5b_iq4_nl.gguf")
    convert_mark_engineer_to_gguf(src_pt, out_gguf)
