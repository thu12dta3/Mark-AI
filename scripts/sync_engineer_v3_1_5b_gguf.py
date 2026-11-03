#!/usr/bin/env python3
"""
scripts/sync_engineer_v3_1_5b_gguf.py

GGUF deepseek2 IQ4_NL Exporter & Synchronizer for Mark-Engineer-V3-1.5B
Exports the 1.52B parameter model (18 layers, H-MLA with fused H_256, 32R/2S MoE)
into a native deepseek2 container with exactly 327 tensors:
- 236 IQ4_NL quantized weights (matrix weights, 32-element block quantization)
- 91 FP32 tensors (RMSNorm weights and router bias offsets)
"""

import os
import sys
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from export.convert_to_deepseek2_gguf import export_to_deepseek2_gguf, verify_gguf_integrity
from huggingface_hub import HfApi, hf_hub_download

REPO_ID = "miceworks/Mark-Engineer-V3-1.5B"
DEFAULT_GGUF_OUT = REPO_ROOT / "models" / "mark-engineer-v3-1.5b" / "mark_engineer_v3_1_5b_iq4_nl.gguf"


def get_hf_token() -> str:
    token = os.environ.get("HF_TOKEN")
    if not token:
        cache_path = os.path.expanduser("~/.cache/huggingface/token")
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                token = f.read().strip()
    return token or ""


def export_and_sync(
    ckpt_path: Path = None,
    output_gguf: Path = DEFAULT_GGUF_OUT,
    push_to_hub: bool = False,
    token: str = None
):
    print("=" * 80)
    print("Mark-Engineer-V3-1.5B GGUF Exporter & Synchronizer (deepseek2 IQ4_NL)")
    print("=" * 80)

    token = token or get_hf_token()

    if ckpt_path is None or not Path(ckpt_path).exists():
        print(f"[Hub] Checking for remote checkpoint in {REPO_ID}...")
        try:
            api = HfApi(token=token if token else None)
            repo_files = api.list_repo_files(REPO_ID)
            if "model.safetensors" in repo_files:
                print(f"[Hub] Found model.safetensors in {REPO_ID}. Downloading...")
                local_ckpt = hf_hub_download(repo_id=REPO_ID, filename="model.safetensors", token=token)
                ckpt_path = Path(local_ckpt)
            else:
                print(f"[Hub] No model.safetensors found on {REPO_ID}. Initializing streaming layer-by-layer...")
                ckpt_path = None
        except Exception as e:
            print(f"[Hub] Note checking remote repo: {e}. Proceeding with streaming layer-by-layer...")
            ckpt_path = None

    print(f"[GGUF Exporter] Exporting Mark-Engineer-V3-1.5B to {output_gguf} (IQ4_NL)...")
    output_gguf.parent.mkdir(parents=True, exist_ok=True)
    out_file = export_to_deepseek2_gguf(
        model_type="engineer-v3-1.5b",
        ckpt_path=ckpt_path,
        output_gguf=output_gguf,
        model_name="Mark-Engineer-V3-1.5B",
        quant_type="iq4_nl",
    )

    verify_gguf_integrity(out_file)
    size_mb = out_file.stat().st_size / (1024 * 1024)
    print(f"[GGUF Exporter] GGUF container validated: {out_file} ({size_mb:.2f} MB)")

    if push_to_hub:
        if not token:
            print("[HuggingFace] Error: HF_TOKEN not found. Skipping Hub upload.")
            return out_file

        print(f"[HuggingFace] Uploading {out_file.name} to {REPO_ID}...")
        api = HfApi(token=token)
        api.upload_file(
            path_or_fileobj=str(out_file),
            path_in_repo=out_file.name,
            repo_id=REPO_ID,
            repo_type="model",
            commit_message=f"Upload Mark-Engineer-V3-1.5B GGUF container ({out_file.name}, IQ4_NL)"
        )
        print(f"[HuggingFace] Upload successful: https://huggingface.co/{REPO_ID}")

    return out_file


def main():
    parser = argparse.ArgumentParser(description="Export Mark-Engineer-V3-1.5B to native deepseek2 GGUF (IQ4_NL)")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to model.safetensors or .pt checkpoint")
    parser.add_argument("--out", type=str, default=str(DEFAULT_GGUF_OUT), help="Output path for GGUF file")
    parser.add_argument("--push", action="store_true", help="Push exported GGUF to Hugging Face Hub")
    args = parser.parse_args()

    export_and_sync(
        ckpt_path=Path(args.ckpt) if args.ckpt else None,
        output_gguf=Path(args.out),
        push_to_hub=args.push,
    )


if __name__ == "__main__":
    main()
