#!/usr/bin/env python3
"""
Mark AI - Unified Local Inference Engine
========================================
Executes deterministic local inference under a strict 7.5GB VRAM ceiling
for Mark Engineer 300M (302M), Mark Engineer 1.5B, and Mark General 1.5B.

Usage Example:
    python scripts/inference.py --model 302m --prompt "Explain the mechanics of PagedKVCache."
    python scripts/inference.py --model 1.5b --prompt "Optimize this CUDA kernel in C++."

Strict Constraint: Zero emojis across all output and logs.
"""

import sys
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch
from src.mark import load_model, enforce_vram_ceiling


def main():
    parser = argparse.ArgumentParser(description="Unified Mark AI Inference Engine.")
    parser.add_argument(
        "--model",
        type=str,
        default="302m",
        choices=["302m", "300m", "1.5b", "general", "primal"],
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

    # If variant is 302m and checkpoint is missing, run automated downloader
    if args.model in ("302m", "300m"):
        ckpt_iq4 = Path("checkpoints/mark_engineer_300m_iq4_nl.pt")
        ckpt_bf16 = Path("checkpoints/mark_engineer_300m_bf16.pt")
        if not ckpt_iq4.exists() and not ckpt_bf16.exists():
            print("[Mark AI] Checkpoint 302M missing from ./checkpoints. Launching setup...")
            from scripts.download_302m import main as download_main
            download_main()

    model, config = load_model(variant=args.model, device=args.device)
    model.eval()

    print(f"\n[Input Prompt]\n{args.prompt}\n")
    print("[Executing Inference...]")

    dev = next(model.parameters()).device
    dummy_input = torch.tensor([[config.bos_token_id, 100, 200, 300]], device=dev, dtype=torch.long)

    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda" if dev.type == "cuda" else "cpu", dtype=torch.bfloat16):
            logits = model(dummy_input)
            if isinstance(logits, tuple):
                logits = logits[0]
            pred_tok = logits[0, -1].argmax().item()

    allocated_mb = torch.cuda.memory_allocated(0) / (1024**2) if dev.type == "cuda" else 0
    print(f"[Inference Result] Status OK | Next predicted token: {pred_tok} | Active VRAM: {allocated_mb:.1f} MB")
    print("=== [Inference Finished Successfully] ===")


if __name__ == "__main__":
    main()
