#!/usr/bin/env python3
"""
Mark AI - Unified Local Inference Engine
========================================
Executes deterministic local inference under a strict 7.5GB VRAM ceiling
for Mark Engineer 300M (302M), Mark SLM-V2-302M, Mark Engineer 1.5B,
Mark General 1.5B, MarkVL 2.2B, Mark Engineer V2 8B, and Mark-v3-Core-1.1B.

Usage Example:
    python scripts/inference.py --model 302m --prompt "Explain the mechanics of PagedKVCache."
    python scripts/inference.py --model 1.5b --prompt "Optimize this CUDA kernel in C++."

Strict Constraint: Zero emojis across all output and logs.
"""

import sys
import time
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from src.mark import load_model, enforce_vram_ceiling, PagedKVCache


def main():
    parser = argparse.ArgumentParser(description="Unified Mark AI Inference Engine.")
    parser.add_argument(
        "--model",
        type=str,
        default="302m",
        choices=["302m", "300m", "slm-v2-302m", "1.5b", "general", "markvl-2.2b", "v2-8b", "v3-1.1b", "primal"],
        help="Mark model variant to load (default: 302m)."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Write a Rust function for asynchronous file reads without thread blocking.",
        help="Input prompt for the model."
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=64,
        help="Maximum number of new tokens to generate (default: 64)."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature (default: 0.2)."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Execution target device ('cuda', 'cpu', or 'auto')."
    )
    args = parser.parse_args()

    print("=== [Mark AI: Unified Inference Engine] ===")
    vram_stats = enforce_vram_ceiling(max_vram_gb=7.5)
    if vram_stats.get("total_vram_gb", 0) > 0:
        print(f"[VRAM] Detected GPU: {vram_stats['total_vram_gb']:.2f} GB | Hardware ceiling applied: {vram_stats['capped_vram_gb']:.2f} GB")
    else:
        print("[Device] Running on CPU.")

    print(f"[Variant] Loading Mark {args.model.upper()}...")

    # If variant is 302m and checkpoint is missing, run automated downloader with empty sys.argv
    if args.model in ("302m", "300m"):
        ckpt_iq4 = Path("checkpoints/mark_engineer_302m_iq4_nl.pt")
        ckpt_bf16 = Path("checkpoints/mark_engineer_302m_bf16.pt")
        if not ckpt_iq4.exists() and not ckpt_bf16.exists():
            print("[Mark AI] Checkpoint 302M missing from ./checkpoints. Launching setup...")
            from scripts.download_302m import main as download_main
            orig_argv = sys.argv
            sys.argv = [sys.argv[0]]
            try:
                download_main()
            finally:
                sys.argv = orig_argv

    model, config = load_model(variant=args.model, device=args.device)
    model.eval()

    # Load BPE Tokenizer
    tok_candidates = [
        REPO_ROOT / f"models/mark-{args.model}/tokenizer.json",
        REPO_ROOT / "models/mark-engineer-1.5b/tokenizer.json",
        REPO_ROOT / "models/mark-vl-2.2b/tokenizer.json",
    ]
    tok_path = next((p for p in tok_candidates if p.exists()), None)
    if tok_path:
        tokenizer = Tokenizer.from_file(str(tok_path))
    else:
        raise FileNotFoundError(f"Tokenizer not found in {tok_candidates}")

    print(f"\n[Input Prompt]\n{args.prompt}\n")
    print("[Executing Autoregressive Inference with Real BPE Tokenizer & PagedKVCache...]")

    dev = next(model.parameters()).device
    compute_dtype = torch.float16 if dev.type == "cuda" else torch.float32

    # Encode prompt
    bos_id = getattr(config, "bos_token_id", 1)
    eos_id = getattr(config, "eos_token_id", 2)
    encoded = tokenizer.encode(args.prompt).ids
    input_ids = [bos_id] + encoded
    cur_tokens = torch.tensor([input_ids], device=dev, dtype=torch.long)

    # Initialize PagedKVCache if supported
    kv_cache = None
    if hasattr(config, "n_layers") and hasattr(config, "q_lora_rank"):
        kv_cache = PagedKVCache(
            n_layers=config.n_layers,
            max_batch_size=1,
            max_seq_len=len(input_ids) + args.max_tokens + 128,
            n_kv_heads=getattr(config, "n_kv_heads", 4),
            head_dim=getattr(config, "head_dim", 64),
            dtype=compute_dtype,
            device=dev,
            use_mla=getattr(config, "use_mla", True),
            kv_lora_rank=getattr(config, "kv_lora_rank", 320),
            qk_rope_head_dim=getattr(config, "qk_rope_head_dim", 32),
            quant_mode="IQ4_NL",
        )

    generated_ids = []
    start_pos = 0
    t0 = time.time()

    with torch.no_grad():
        for step in range(args.max_tokens):
            if kv_cache is not None:
                logits = model(cur_tokens, kv_cache=kv_cache, start_pos=start_pos)
                start_pos += cur_tokens.shape[1]
            else:
                logits = model(cur_tokens)

            if isinstance(logits, tuple):
                logits = logits[0]
            elif isinstance(logits, dict):
                logits = logits["logits"]

            next_token_logits = logits[:, -1, :].float()
            if args.temperature <= 0:
                next_tok = torch.argmax(next_token_logits, dim=-1).item()
            else:
                probs = F.softmax(next_token_logits / args.temperature, dim=-1)
                next_tok = torch.multinomial(probs, num_samples=1).item()

            if next_tok == eos_id:
                break

            generated_ids.append(next_tok)
            cur_tokens = torch.tensor([[next_tok]], device=dev, dtype=torch.long)

    elapsed = max(1e-5, time.time() - t0)
    tokens_per_sec = len(generated_ids) / elapsed
    allocated_mb = torch.cuda.memory_allocated(0) / (1024**2) if dev.type == "cuda" else 0

    decoded_text = tokenizer.decode(generated_ids)
    print(f"\n[Generated Output]\n{decoded_text}\n")
    print(f"[Performance] Generated {len(generated_ids)} tokens in {elapsed:.2f}s ({tokens_per_sec:.2f} tokens/s) | Active VRAM: {allocated_mb:.1f} MB")
    print("=== [Inference Finished Successfully] ===")


if __name__ == "__main__":
    main()
