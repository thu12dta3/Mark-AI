#!/usr/bin/env python3
import os
import sys
import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from src.model.mark_arch import MarkConfig, MarkModel
from export.quantize_4bit import quantize_mark_model_nf4
DEFAULT_METADATA = {'model_name': 'Mark Engineer 300M (302M)', 'version': '1.0.0', 'architecture': 'DeepSeek-V3 MLA + DeepSeekMoE + MTP', 'total_parameters': 302019848, 'active_parameters_per_token': '110M (default) to 165M (top-4)', 'hidden_dim': 1024, 'num_layers': 8, 'attention_heads': 16, 'kv_lora_rank': 256, 'routed_experts': 16, 'shared_experts': 1, 'formats': {'iq4_nl': {'filename': 'mark_engineer_302m_iq4_nl.pt', 'url': 'https://github.com/thu12dta3/Mark-AI/releases/download/v1.0.0-302m/mark_engineer_302m_iq4_nl.pt', 'size_mb': 162.99, 'vram_mb': 350.0, 'sha256': 'daf05dd0dab4fc90f620d959cb6e5731c49b3f3c4ddce0ff28765e0751075fd9', 'description': 'Converged checkpoint in native non-linear 4-bit quantization (163 MB).'}}}

def compute_sha256(filepath: Path) -> str:
    sha = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while (chunk := f.read(65536)):
            sha.update(chunk)
    return sha.hexdigest()

def resolve_local_storage(filename: str) -> Path | None:
    candidates = [Path(f'/mnt/ssd_1/mark ai/checkpoints/{filename}'), Path(f'/mnt/ssd_1/mark ai/checkpoints/300m_backup/{filename}'), Path(f'/mnt/ssd_1/mark ai/{filename}'), Path(f'./checkpoints/{filename}')]
    for c in candidates:
        if c.exists() and c.stat().st_size > 1024 * 1024:
            return c
    return None

def download_with_progress(url: str, dest_path: Path):

    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100.0, downloaded * 100.0 / total_size)
            mb_down = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(f'\r[Download] {mb_down:.1f} MB / {mb_total:.1f} MB ({percent:.1f}%)')
        else:
            mb_down = downloaded / (1024 * 1024)
            sys.stdout.write(f'\r[Download] {mb_down:.1f} MB downloaded...')
        sys.stdout.flush()
    print(f'[Remote Download] Fetching trained weights from: {url}')
    urllib.request.urlretrieve(url, dest_path, reporthook=reporthook)
    print('\n[Download] Finished successfully.')

def main():
    parser = argparse.ArgumentParser(description='Download trained checkpoint for Mark Engineer 300M (302M).')
    parser.add_argument('--dest', type=str, default='./checkpoints', help='Destination directory for model weights (default: ./checkpoints).')
    args = parser.parse_args()
    dest_dir = Path(args.dest)
    dest_dir.mkdir(parents=True, exist_ok=True)
    target_info = DEFAULT_METADATA['formats']['iq4_nl']
    target_filename = target_info['filename']
    target_path = dest_dir / target_filename
    expected_sha = target_info['sha256']
    url = target_info['url']
    print('=== [Mark Engineer 300M (302M): Checkpoint Download Utility] ===')
    print(f'[Destination] {target_path.resolve()}')
    if target_path.exists() and target_path.stat().st_size > 1024 * 1024:
        current_sha = compute_sha256(target_path)
        if current_sha == expected_sha:
            size_mb = target_path.stat().st_size / (1024 * 1024)
            print(f'[Mark AI] Checkpoint 302M already verified and present: {target_path} ({size_mb:.2f} MB)')
            print(f'[Integrity SHA256] {current_sha} (OK)')
            return
    local_source = resolve_local_storage(target_filename)
    if local_source is not None and local_source.resolve() != target_path.resolve():
        print(f'[Mark AI] Found in local storage: {local_source}')
        shutil.copy2(local_source, target_path)
        print(f'[Mark AI] Copied to {target_path} ({target_path.stat().st_size / (1024 * 1024):.2f} MB).')
    else:
        try:
            download_with_progress(url, target_path)
        except Exception as e:
            print(f'[Download Error] {e}')
            sys.exit(1)
    current_sha = compute_sha256(target_path)
    size_mb = target_path.stat().st_size / (1024 * 1024)
    print(f'\n[Final Verification] File: {target_filename} ({size_mb:.2f} MB)')
    print(f'[SHA256] {current_sha}')
    if current_sha == expected_sha:
        print('[Integrity Status] SHA256 VALIDATED SUCCESSFULLY (100% matched to production release).')
    else:
        print('[Integrity Warning] Checksum mismatch against official release.')
    meta_path = dest_dir / 'mark_engineer_300m_metadata.json'
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(DEFAULT_METADATA, f, indent=2)
    print('\n=== [Model Ready for Inference] ===')
    print('Run inference with:')
    print('  python scripts/inference.py --model 302m --prompt "Write a concurrent lock-free ring buffer in C++20."')
if __name__ == '__main__':
    main()
