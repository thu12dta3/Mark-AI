#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from export.convert_to_deepseek2_gguf import export_to_deepseek2_gguf

def convert_pt_to_gguf(ckpt_path: Path, output_gguf: Path, model_name: str = "Mark Engineer 300M", model_type: str = "302m"):
    return export_to_deepseek2_gguf(model_type=model_type, ckpt_path=ckpt_path, output_gguf=output_gguf, model_name=model_name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Mark AI checkpoints to GGUF format.")
    parser.add_argument("--model", type=str, default="302m", choices=["302m", "1.5b", "engineer-1.5b", "general-1.5b"], help="Model variant")
    parser.add_argument("--ckpt", type=str, default="", help="Path to source .pt checkpoint")
    parser.add_argument("--out", type=str, default="", help="Path to output .gguf file")
    args = parser.parse_args()

    default_paths = {
        "302m": (Path("/mnt/ssd_1/mark ai/checkpoints/mark_engineer_302m_iq4_nl.pt"), Path("/mnt/ssd_1/mark ai/checkpoints/mark_engineer_302m_iq4_nl.gguf"), "Mark Engineer 300M"),
        "engineer-1.5b": (Path("/mnt/ssd_1/mark ai/checkpoints/mark_engineer_1_5b_iq4_nl.pt"), Path("/mnt/ssd_1/mark ai/checkpoints/mark_engineer_1_5b_iq4_nl.gguf"), "Mark Engineer 1.5B"),
        "general-1.5b": (Path("/mnt/ssd_1/mark ai/checkpoints/mark_general_1_5b_iq4_nl.pt"), Path("/mnt/ssd_1/mark ai/checkpoints/mark_general_1_5b_iq4_nl.gguf"), "Mark General 1.5B"),
    }

    m_key = "engineer-1.5b" if args.model in ("1.5b", "engineer-1.5b") else args.model
    def_src, def_out, def_name = default_paths.get(m_key, default_paths["302m"])

    src_pt = Path(args.ckpt) if args.ckpt else def_src
    out_gguf = Path(args.out) if args.out else def_out

    if src_pt.exists():
        convert_pt_to_gguf(src_pt, out_gguf, model_name=def_name, model_type=m_key)
    else:
        print(f"Source checkpoint {src_pt} not found on disk. Skipping.")
