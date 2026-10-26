#!/usr/bin/env python3
"""
Mark AI - Checkpoint to GGUF Converter (LM Studio & llama.cpp Plug-and-Play)
===========================================================================
Converts Mark Engineer 300M (302M) weights into a GGUF container compatible
with LM Studio, Ollama, llama.cpp, and vLLM.

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


def convert_pt_to_gguf(
    ckpt_path: Path,
    output_gguf: Path,
    model_name: str = "Mark Engineer 300M",
    model_type: str = "302m",
):
    print(f"[GGUF Converter] Loading checkpoint: {ckpt_path}...")
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if model_type in ("1.5b", "engineer-1.5b"):
        config = MarkConfig.create_engineer_1_5b()
        description = "Mark Engineer 1.5B (1.502B) - MLA + DeepSeekMoE + MTP trained on A100."
    elif model_type == "general-1.5b":
        config = MarkConfig.create_engineer_1_5b()
        description = "Mark General 1.5B (1.502B) - MLA + DeepSeekMoE + MTP + Self-RAG trained on A100."
    else:
        config = MarkConfig.create_engineer_300m()
        description = "Mark Engineer 300M (302M) - MLA + DeepSeekMoE + MTP trained on A100."

    output_gguf.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(str(output_gguf), "llama")

    # Model Metadata for LM Studio
    writer.add_name(model_name)
    writer.add_author("thu12dta3")
    writer.add_version("1.0.0")
    writer.add_description(description)
    writer.add_context_length(config.max_seq_len)
    writer.add_embedding_length(config.dim)
    writer.add_block_count(config.n_layers)
    writer.add_feed_forward_length(config.intermediate_size)
    writer.add_head_count(config.n_heads)
    writer.add_head_count_kv(config.n_kv_heads)
    writer.add_layer_norm_rms_eps(config.rms_norm_eps)
    writer.add_rope_freq_base(config.rope_theta)

    # Special Tokens & Vocab
    writer.add_bos_token_id(config.bos_token_id)
    writer.add_eos_token_id(config.eos_token_id)
    writer.add_pad_token_id(config.pad_token_id)

    tok_path = REPO_ROOT / "models" / "mark-engineer-1.5b" / "tokenizer.json"
    if tok_path.exists():
        with open(tok_path, "r", encoding="utf-8") as f:
            tok_data = json.load(f)
        vocab = tok_data.get("model", {}).get("vocab", {})
        sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
        tokens = [k.encode("utf-8") for k, v in sorted_vocab]
        writer.add_tokenizer_model("gpt2")
        writer.add_token_list(tokens)
        merges = tok_data.get("model", {}).get("merges", [])
        formatted_merges = [" ".join(m).encode("utf-8") if isinstance(m, list) else m.encode("utf-8") for m in merges]
        writer.add_token_merges(formatted_merges)
        writer.add_token_scores([0.0] * len(tokens))
        writer.add_token_types([1] * len(tokens))

    # Chat Template for LM Studio
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

    print("[GGUF Converter] Mapping and dequantizing tensors to float16 GGUF precision...")
    lut = torch.tensor(IQ4_NL_LOOKUP_VALUES, dtype=torch.float32)

    tensor_count = 0
    for name, val in state_dict.items():
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
            t_data = w.to(torch.float16).numpy()
        elif isinstance(val, torch.Tensor):
            t_data = val.to(torch.float16).numpy()
        else:
            continue

        # Standard GGUF tensor naming
        gguf_name = name
        if name == "tok_embeddings.weight":
            gguf_name = "token_embd.weight"
        elif name == "norm.weight":
            gguf_name = "output_norm.weight"
        elif name == "lm_head.weight":
            gguf_name = "output.weight"
        elif name.startswith("layers."):
            parts = name.split(".")
            layer_idx = parts[1]
            sub = ".".join(parts[2:])
            if sub == "attn_norm.weight":
                gguf_name = f"blk.{layer_idx}.attn_norm.weight"
            elif sub == "ffn_norm.weight":
                gguf_name = f"blk.{layer_idx}.ffn_norm.weight"
            elif "attn." in sub:
                param = sub.replace("attn.", "")
                gguf_name = f"blk.{layer_idx}.attn_{param}"
            elif "ffn." in sub:
                param = sub.replace("ffn.", "")
                gguf_name = f"blk.{layer_idx}.ffn_{param}"
            else:
                gguf_name = f"blk.{layer_idx}.{sub}"

        writer.add_tensor(gguf_name, t_data)
        tensor_count += 1

    print(f"[GGUF Converter] Writing header and {tensor_count} tensors to {output_gguf}...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    size_mb = output_gguf.stat().st_size / (1024 * 1024)
    print(f"[GGUF Converter] Finished successfully. GGUF generated: {output_gguf} ({size_mb:.2f} MB)")
    return output_gguf


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert Mark AI checkpoints to GGUF format.")
    parser.add_argument("--model", type=str, default="302m", choices=["302m", "1.5b", "engineer-1.5b", "general-1.5b"], help="Model variant")
    parser.add_argument("--ckpt", type=str, default="", help="Path to source .pt checkpoint")
    parser.add_argument("--out", type=str, default="", help="Path to output .gguf file")
    args = parser.parse_args()

    default_paths = {
        "302m": (
            Path("/mnt/ssd_1/mark ai/checkpoints/mark_engineer_302m_iq4_nl.pt"),
            Path("/mnt/ssd_1/mark ai/checkpoints/mark_engineer_302m_iq4_nl.gguf"),
            "Mark Engineer 300M"
        ),
        "engineer-1.5b": (
            Path("/mnt/ssd_1/mark ai/checkpoints/mark_engineer_1_5b_iq4_nl.pt"),
            Path("/mnt/ssd_1/mark ai/checkpoints/mark_engineer_1_5b_iq4_nl.gguf"),
            "Mark Engineer 1.5B"
        ),
        "general-1.5b": (
            Path("/mnt/ssd_1/mark ai/checkpoints/mark_general_1_5b_iq4_nl.pt"),
            Path("/mnt/ssd_1/mark ai/checkpoints/mark_general_1_5b_iq4_nl.gguf"),
            "Mark General 1.5B"
        ),
    }

    m_key = "engineer-1.5b" if args.model in ("1.5b", "engineer-1.5b") else args.model
    def_src, def_out, def_name = default_paths.get(m_key, default_paths["302m"])

    src_pt = Path(args.ckpt) if args.ckpt else def_src
    out_gguf = Path(args.out) if args.out else def_out

    convert_pt_to_gguf(src_pt, out_gguf, model_name=def_name, model_type=m_key)
