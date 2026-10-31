#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from export.convert_to_deepseek2_gguf import export_to_deepseek2_gguf

def export_model_gguf(model_type: str, ckpt_path: Path, output_gguf: Path, model_name: str):
    return export_to_deepseek2_gguf(model_type=model_type, ckpt_path=ckpt_path, output_gguf=output_gguf, model_name=model_name)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export compliant deepseek2 GGUF for Mark models.')
    parser.add_argument('--model', type=str, default='302m', choices=['302m', '1.5b', 'general-1.5b', 'all'])
    args = parser.parse_args()
    targets = {'302m': ('302m', Path('/mnt/ssd_1/mark ai/checkpoints/mark_engineer_302m_iq4_nl.pt'), Path('/mnt/ssd_1/mark ai/checkpoints/mark_engineer_302m_iq4_nl.gguf'), 'Mark Engineer 300M'), '1.5b': ('1.5b', Path('/mnt/ssd_1/mark ai/checkpoints/mark_engineer_1_5b_iq4_nl.pt'), Path('/mnt/ssd_1/mark ai/checkpoints/mark_engineer_1_5b_iq4_nl.gguf'), 'Mark Engineer 1.5B'), 'general-1.5b': ('general-1.5b', Path('/mnt/ssd_1/mark ai/checkpoints/mark_general_1_5b_iq4_nl.pt'), Path('/mnt/ssd_1/mark ai/checkpoints/mark_general_1_5b_iq4_nl.gguf'), 'Mark General 1.5B')}
    selected = list(targets.keys()) if args.model == 'all' else [args.model]
    for key in selected:
        m_type, src, dst, name = targets[key]
        if src.exists():
            export_model_gguf(m_type, src, dst, name)
        else:
            print(f'[Warning] Source checkpoint not found: {src}')
