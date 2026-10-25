#!/usr/bin/env python3
"""
Mark Engineer 300M (302M) - Download & Setup Script
===================================================
Script oficial para obtencao dos pesos treinados do Mark Engineer 300M
(302.02M parametros totais, ~110M a 165M ativos por token).

Download Direto (1 Arquivo Solo - Estilo Hugging Face):
    curl -L -O https://github.com/thu12dta3/Mark-AI/releases/download/v1.0.0-302m/mark_engineer_302m_iq4_nl.pt

Ou via Python:
    python download_302m.py
    python download_302m.py --dest ./checkpoints

Regra Estrita: Zero emojis em qualquer texto, log ou print.
"""

import os
import sys
import argparse
import hashlib
import json
import shutil
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from src.model.mark_arch import MarkConfig, MarkModel
from export.quantize_4bit import quantize_mark_model_nf4


DEFAULT_METADATA = {
    "model_name": "Mark Engineer 300M (302M)",
    "version": "1.0.0",
    "architecture": "DeepSeek-V3 MLA + DeepSeekMoE + MTP",
    "total_parameters": 302019848,
    "active_parameters_per_token": "110M (padrao) a 165M (top-4)",
    "hidden_dim": 1024,
    "num_layers": 8,
    "attention_heads": 16,
    "kv_lora_rank": 256,
    "routed_experts": 16,
    "shared_experts": 1,
    "formats": {
        "iq4_nl": {
            "filename": "mark_engineer_302m_iq4_nl.pt",
            "url": "https://github.com/thu12dta3/Mark-AI/releases/download/v1.0.0-302m/mark_engineer_302m_iq4_nl.pt",
            "size_mb": 162.99,
            "vram_mb": 350.0,
            "sha256": "daf05dd0dab4fc90f620d959cb6e5731c49b3f3c4ddce0ff28765e0751075fd9",
            "description": "Pesos convergidos em quantizacao 4-bit nao-linear nativa (163 MB)."
        }
    }
}


def compute_sha256(filepath: Path) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            sha.update(chunk)
    return sha.hexdigest()


def resolve_local_storage(filename: str) -> Path | None:
    """Busca se o arquivo ja existe no SSD SATA secundario ou no diretorio atual."""
    candidates = [
        Path(f"/mnt/ssd_1/mark ai/checkpoints/{filename}"),
        Path(f"/mnt/ssd_1/mark ai/checkpoints/300m_backup/{filename}"),
        Path(f"/mnt/ssd_1/mark ai/{filename}"),
        Path(f"./checkpoints/{filename}"),
    ]
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
            sys.stdout.write(f"\r[Download] {mb_down:.1f} MB / {mb_total:.1f} MB ({percent:.1f}%)")
        else:
            mb_down = downloaded / (1024 * 1024)
            sys.stdout.write(f"\r[Download] {mb_down:.1f} MB baixados...")
        sys.stdout.flush()

    print(f"[Download Remoto] Baixando pesos treinados de: {url}")
    urllib.request.urlretrieve(url, dest_path, reporthook=reporthook)
    print("\n[Download] Concluido com sucesso.")


def main():
    parser = argparse.ArgumentParser(description="Download do modelo treinado Mark Engineer 300M (302M).")
    parser.add_argument(
        "--dest",
        type=str,
        default="./checkpoints",
        help="Diretorio de destino para os pesos do modelo (padrao: ./checkpoints)."
    )
    args = parser.parse_args()

    dest_dir = Path(args.dest)
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    target_info = DEFAULT_METADATA["formats"]["iq4_nl"]
    target_filename = target_info["filename"]
    target_path = dest_dir / target_filename
    expected_sha = target_info["sha256"]
    url = target_info["url"]

    print("=== [Mark Engineer 300M (302M): Download de Pesos Treinados] ===")
    print(f"[Destino] {target_path.resolve()}")

    if target_path.exists() and target_path.stat().st_size > 1024 * 1024:
        current_sha = compute_sha256(target_path)
        if current_sha == expected_sha:
            size_mb = target_path.stat().st_size / (1024 * 1024)
            print(f"[Mark AI] Checkpoint 302M ja presente e integro: {target_path} ({size_mb:.2f} MB)")
            print(f"[Integridade SHA256] {current_sha} (OK)")
            return

    # 1. Verifica no SSD SATA secundario
    local_source = resolve_local_storage(target_filename)
    if local_source is not None and local_source.resolve() != target_path.resolve():
        print(f"[Mark AI] Encontrado no armazenamento local: {local_source}")
        shutil.copy2(local_source, target_path)
        print(f"[Mark AI] Copiado para {target_path} ({target_path.stat().st_size / (1024*1024):.2f} MB).")
    else:
        # 2. Download direto via HTTPS do release oficial
        try:
            download_with_progress(url, target_path)
        except Exception as e:
            print(f"[Erro no Download] {e}")
            sys.exit(1)

    # Verificacao de integridade
    current_sha = compute_sha256(target_path)
    size_mb = target_path.stat().st_size / (1024 * 1024)
    print(f"\n[Verificacao Final] Arquivo: {target_filename} ({size_mb:.2f} MB)")
    print(f"[SHA256] {current_sha}")
    if current_sha == expected_sha:
        print("[Status de Integridade] SHA256 VALIDADO COM SUCESSO (100% compativel com o treino de producao).")
    else:
        print("[Aviso de Integridade] Hash divergente do release oficial.")

    # Grava metadados
    meta_path = dest_dir / "mark_engineer_300m_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(DEFAULT_METADATA, f, indent=2)

    print("\n=== [Modelo Pronto para Inferencia] ===")
    print("Execute inferencia imediata com:")
    print("  python inference.py --model 302m --prompt \"Escreva um lock-free ring buffer concorrente em C++20.\"")


if __name__ == "__main__":
    main()
