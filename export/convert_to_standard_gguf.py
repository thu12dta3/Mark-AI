#!/usr/bin/env python3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from export.convert_to_deepseek2_gguf import export_to_deepseek2_gguf

def export_mark_engineer_standard_gguf(ckpt_path: Path, output_gguf: Path, model_name: str = "mark-engineer-1.5b"):
    return export_to_deepseek2_gguf(model_type="engineer-1.5b", ckpt_path=ckpt_path, output_gguf=output_gguf, model_name=model_name)

if __name__ == "__main__":
    src_pt = Path("/mnt/ssd_1/mark ai/checkpoints/mark_engineer_1_5b_iq4_nl.pt")
    out_gguf = Path("/mnt/ssd_1/mark ai/checkpoints/mark_engineer_1_5b_iq4_nl.gguf")
    if src_pt.exists():
        export_mark_engineer_standard_gguf(src_pt, out_gguf)
    else:
        print(f"Source checkpoint {src_pt} not found on disk. Skipping.")
