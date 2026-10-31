#!/usr/bin/env python3
import os
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from export.convert_to_deepseek2_gguf import export_to_deepseek2_gguf
SSD_CHECKPOINTS = Path('/mnt/ssd_1/mark ai/checkpoints/mark_engineer_v2_8b')
OUTPUT_GGUF = SSD_CHECKPOINTS / 'mark_engineer_v2_8b_deepseek2.gguf'
CKPT_FILE = SSD_CHECKPOINTS / 'mark_engineer_v2_8b_step_15000.pt'

def export_v2_8b_gguf(ckpt_file: Path=CKPT_FILE, output_gguf: Path=OUTPUT_GGUF):
    if not ckpt_file.exists():
        raise FileNotFoundError(f'Checkpoint file does not exist: {ckpt_file}. Please ensure mark_engineer_v2_8b_step_15000.pt is mounted or downloaded before exporting.')
    print('=' * 80)
    print('MARK-ENGINEER-V2-8B: GGUF EXPORT PIPELINE (deepseek2)')
    print(f'Source Checkpoint: {ckpt_file}')
    print(f'Target Container:  {output_gguf}')
    print('=' * 80)
    export_to_deepseek2_gguf(model_type='v2-8b', ckpt_path=ckpt_file, output_gguf=output_gguf, model_name='Mark Engineer V2 8B')
if __name__ == '__main__':
    export_v2_8b_gguf()
