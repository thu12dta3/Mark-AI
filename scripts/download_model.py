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
CATALOG = {'302m': {'model_name': 'Mark Engineer 300M (302M)', 'version': '1.0.0', 'total_parameters': 302019848, 'filename': 'mark_engineer_302m_iq4_nl.gguf', 'url': 'https://github.com/thu12dta3/Mark-AI/releases/download/v1.0.0-302m/mark_engineer_302m_iq4_nl.gguf', 'size_mb': 576.07, 'vram_mb': 350.0, 'sha256': '070657c1a6910a6f770a871ed73e7bb529bf6eb2183e4c4bb816fa8a7864f128', 'description': 'Mark Engineer 302M in GGUF format for LM Studio and Ollama.'}, 'engineer-1.5b': {'model_name': 'Mark Engineer 1.5B (Conversational & Systems)', 'version': '1.0.0', 'total_parameters': 1501680656, 'filename': 'mark_engineer_1_5b_iq4_nl.gguf', 'url': 'https://github.com/thu12dta3/Mark-AI/releases/download/v1.0.0-engineer-1.5b/mark_engineer_1_5b_iq4_nl.gguf', 'size_mb': 2853.90, 'vram_mb': 1250.0, 'sha256': '2e84c661c3b4a00165a3a21608effe25cb9c87e6b469cc709d793858667d7a16', 'description': 'Mark Engineer 1.5B in deepseek2 GGUF format for LM Studio and Ollama.'}, 'general-1.5b': {'model_name': 'Mark General 1.5B (General Knowledge & STEM)', 'version': '1.0.0', 'total_parameters': 1501680656, 'filename': 'mark_general_1_5b_iq4_nl.gguf', 'url': 'https://github.com/thu12dta3/Mark-AI/releases/download/v1.0.0-general-1.5b/mark_general_1_5b_iq4_nl.gguf', 'size_mb': 2854.02, 'vram_mb': 1250.0, 'sha256': '6f6c7d8ca470829325b6a1e8162689f41746a7a2817f520ef8cb7a1fef7c4087', 'description': 'Mark General 1.5B in deepseek2 GGUF format for LM Studio and Ollama.'}, 'markvl-2.2b': {'model_name': 'MarkVL 2.2B (Dense Visual Perception Engine)', 'version': '1.0.0', 'total_parameters': 2222269696, 'filename': 'mark_vl_2_2b.gguf', 'url': 'https://github.com/thu12dta3/Mark-AI/releases/download/v1.0.0-markvl-2.2b/mark_vl_2_2b_q4_k_m.gguf.part-00', 'size_mb': 2380.88, 'vram_mb': 1850.0, 'sha256': '910b75cfe9c9a1bc418e72c4b0258248f171445d65d044d894fba341f0a69cbf', 'description': 'MarkVL 2.2B Multimodal Vision-Language Engine in GGUF format for Ollama and llama.cpp.'}}

def compute_sha256(filepath: Path) -> str:
    sha = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while (chunk := f.read(65536)):
            sha.update(chunk)
    return sha.hexdigest()

def resolve_local_storage(filename: str) -> Path | None:
    candidates = [Path(f'/mnt/ssd_1/mark ai/checkpoints/{filename}'), Path(f'/mnt/ssd_1/mark ai/checkpoints/mark_vl_2_2b/{filename}'), Path(f'/mnt/ssd_1/mark ai/checkpoints/mark_vl_2_2b/mark_vl_2_2b_q4_k_m.gguf'), Path(f'/mnt/ssd_1/mark ai/checkpoints/300m_backup/{filename}'), Path(f'/mnt/ssd_1/mark ai/{filename}'), Path(f'./checkpoints/{filename}')]
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

def download_single_model(model_key: str, dest_dir: Path):
    if model_key not in CATALOG:
        print(f'[Error] Unknown model key: {model_key}. Available: {list(CATALOG.keys())}')
        return
    info = CATALOG[model_key]
    target_filename = info['filename']
    target_path = dest_dir / target_filename
    expected_sha = info['sha256']
    url = info['url']
    print(f"\n=== [Downloading: {info['model_name']}] ===")
    print(f'[Target File] {target_path.resolve()}')
    print(f"[Expected Size] ~{info['size_mb']:.1f} MB")
    if target_path.exists() and target_path.stat().st_size > 1024 * 1024:
        current_sha = compute_sha256(target_path)
        if current_sha == expected_sha:
            size_mb = target_path.stat().st_size / (1024 * 1024)
            print(f'[Mark AI] Verified and present: {target_path} ({size_mb:.2f} MB)')
            return
    local_source = resolve_local_storage(target_filename)
    if local_source is not None and local_source.resolve() != target_path.resolve():
        print(f'[Mark AI] Found in local storage: {local_source}')
        shutil.copy2(local_source, target_path)
        print(f'[Mark AI] Copied to {target_path} ({target_path.stat().st_size / (1024 * 1024):.2f} MB).')
    else:
        try:
            if 'parts' in info:
                base_url = info['base_url']
                parts = info['parts']
                part_files = []
                for p in parts:
                    p_url = f'{base_url}/{p}'
                    p_path = target_dir / p
                    if not p_path.exists():
                        download_with_progress(p_url, p_path)
                    part_files.append(p_path)
                print(f'[Reassembly] Concatenating multipart archives into {target_path}...')
                with open(target_path, 'wb') as out_f:
                    for pf in part_files:
                        with open(pf, 'rb') as in_f:
                            shutil.copyfileobj(in_f, out_f)
                        pf.unlink()
            else:
                download_with_progress(url, target_path)
        except Exception as e:
            print(f'[Download Error] {e}')
            return
    current_sha = compute_sha256(target_path)
    size_mb = target_path.stat().st_size / (1024 * 1024)
    print(f'[Verification] File: {target_filename} ({size_mb:.2f} MB)')
    print(f'[SHA256] {current_sha}')

def main():
    parser = argparse.ArgumentParser(description='Download trained checkpoints for Mark AI models.')
    parser.add_argument('--model', type=str, default='engineer-1.5b', choices=['302m', '1.5b', 'engineer-1.5b', 'general-1.5b', 'markvl-2.2b', 'all'], help='Model checkpoint variant to download (default: engineer-1.5b).')
    parser.add_argument('--dest', type=str, default='./checkpoints', help='Destination directory for model weights (default: ./checkpoints).')
    args = parser.parse_args()
    dest_dir = Path(args.dest)
    dest_dir.mkdir(parents=True, exist_ok=True)
    if args.model in ('1.5b', 'engineer-1.5b'):
        download_single_model('engineer-1.5b', dest_dir)
    elif args.model == 'general-1.5b':
        download_single_model('general-1.5b', dest_dir)
    elif args.model == '302m':
        download_single_model('302m', dest_dir)
    elif args.model == 'markvl-2.2b':
        download_single_model('markvl-2.2b', dest_dir)
    elif args.model == 'all':
        for k in CATALOG.keys():
            download_single_model(k, dest_dir)
if __name__ == '__main__':
    main()
