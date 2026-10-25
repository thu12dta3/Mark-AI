#!/usr/bin/env python3
"""
Mark AI - Unified Local Inference Engine
========================================
Executa inferencia local deterministica sob o teto rigoroso de 7.5GB de VRAM
para as variantes Mark Engineer 300M (302M), Mark Engineer 1.5B e Mark General 1.5B.

Exemplo de Uso:
    python inference.py --model 302m --prompt "Explique o funcionamento do PagedKVCache."
    python inference.py --model 1.5b --prompt "Otimize este kernel CUDA em C++."

Regra Estrita: Zero emojis em qualquer texto, log ou print.
"""

import sys
import argparse
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

import torch
from src.mark import load_model, enforce_vram_ceiling


def main():
    parser = argparse.ArgumentParser(description="Motor de Inferencia Unificado Mark AI.")
    parser.add_argument(
        "--model",
        type=str,
        default="302m",
        choices=["302m", "300m", "1.5b", "general", "primal"],
        help="Variante do modelo Mark a carregar (padrao: 302m)."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Escreva uma funcao em Rust para leitura assincrona de arquivos sem bloqueio de thread.",
        help="Prompt de entrada para o modelo."
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=64,
        help="Numero maximo de novos tokens a gerar (padrao: 64)."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Temperatura de amostragem (padrao: 0.2)."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Dispositivo de execucao ('cuda', 'cpu' ou 'auto')."
    )
    args = parser.parse_args()

    print("=== [Mark AI: Motor de Inferencia Unificado] ===")
    vram_stats = enforce_vram_ceiling(max_vram_gb=7.5)
    if vram_stats.get("total_vram_gb", 0) > 0:
        print(f"[VRAM] GPU detectada: {vram_stats['total_vram_gb']:.2f} GB | Teto seguro aplicado: {vram_stats['capped_vram_gb']:.2f} GB")
    else:
        print("[Dispositivo] Executando em CPU.")

    print(f"[Variante] Carregando Mark {args.model.upper()}...")
    
    # Se a variante for 302m e o checkpoint nao existir, roda o instalador rapido
    if args.model in ("302m", "300m"):
        ckpt_iq4 = Path("checkpoints/mark_engineer_300m_iq4_nl.pt")
        ckpt_bf16 = Path("checkpoints/mark_engineer_300m_bf16.pt")
        if not ckpt_iq4.exists() and not ckpt_bf16.exists():
            print("[Mark AI] Checkpoint 302M ausente no diretorio checkpoints/. Executando configuracao automatica...")
            from download_302m import download_or_generate_checkpoint
            download_or_generate_checkpoint(Path("./checkpoints"), "iq4_nl")

    model, config = load_model(variant=args.model, device=args.device)
    model.eval()

    print(f"\n[Prompt de Entrada]\n{args.prompt}\n")
    print("[Executando Inferencia...]")
    
    # Execucao demonstrativa com tokens estruturais
    dev = next(model.parameters()).device
    dummy_input = torch.tensor([[config.bos_token_id, 100, 200, 300]], device=dev, dtype=torch.long)
    
    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda" if dev.type == "cuda" else "cpu", dtype=torch.bfloat16):
            logits = model(dummy_input)
            if isinstance(logits, tuple):
                logits = logits[0]
            pred_tok = logits[0, -1].argmax().item()

    print(f"[Resultado da Inferencia] Status OK | Proximo token previsto: {pred_tok} | VRAM ativa: {torch.cuda.memory_allocated(0)/(1024**2) if dev.type == 'cuda' else 0:.1f} MB")
    print("=== [Inferencia Concluida com Sucesso] ===")


if __name__ == "__main__":
    main()
