"""
Mark AI - High-Density Neural Architecture & Inference Library
==============================================================
Provides high-efficiency inference with DeepSeek Multi-Head Latent Attention (MLA),
DeepSeekMoE, Multi-Token Prediction (MTP), and IQ4_NL native quantization.

Supported Model Variants:
- Mark Engineer 300M (302.02M parameters, 163 MB in IQ4_NL)
- Mark Engineer 1.5B (1.502B parameters, 820 MB in IQ4_NL)
- Mark General 1.5B  (1.502B parameters, 10 Cognition Domains + Self-RAG)
- Mark Primal Base   (~2.98B parameters)
"""

import os
from pathlib import Path
from typing import Optional, Tuple, Any

from src.model.mark_arch import (
    MarkConfig,
    MarkModel,
    MarkForCausalLM,
    PagedKVCache,
)
from src.model.self_rag import SelfRAGController, SpecialTokens
from export.quantize_4bit import (
    NativeIQ4NLLinear,
    MarkInferenceEngine,
    load_mark_for_8gb_vram,
    enforce_vram_ceiling,
    quantize_mark_model_nf4,
)

__version__ = "1.0.0"

__all__ = [
    "MarkConfig",
    "MarkModel",
    "MarkForCausalLM",
    "PagedKVCache",
    "SelfRAGController",
    "SpecialTokens",
    "NativeIQ4NLLinear",
    "MarkInferenceEngine",
    "load_mark_for_8gb_vram",
    "enforce_vram_ceiling",
    "quantize_to_iq4_nl",
    "load_model",
]


def load_model(
    variant: str = "302m",
    checkpoint_path: Optional[str] = None,
    device: str = "auto",
    max_context_tokens: int = 4096,
) -> Tuple[MarkModel, MarkConfig]:
    """
    Carrega o modelo Mark com configuracao automatica por variante.
    Variantes suportadas:
    - '302m' ou '300m': Mark Engineer 300M (302M parametros)
    - '1.5b': Mark Engineer 1.5B (1.502B parametros)
    - 'general': Mark General 1.5B (1.502B parametros com 10 dominios de cognicao)
    - 'primal': Mark Primal Base (~2.98B parametros)
    """
    variant_norm = variant.lower().strip()
    if variant_norm in ("302m", "300m", "engineer_300m"):
        config = MarkConfig.create_engineer_300m()
    elif variant_norm in ("1.5b", "1_5b", "engineer_1_5b"):
        config = MarkConfig.create_engineer_1_5b()
    elif variant_norm in ("general", "general_1_5b"):
        config = MarkConfig.create_general_1_5b()
    elif variant_norm in ("primal", "base", "0.1"):
        config = MarkConfig.create_primal()
    else:
        raise ValueError(f"Variante desconhecida: '{variant}'. Use '302m', '1.5b', 'general' ou 'primal'.")

    model = MarkModel(config)

    # Localizacao de checkpoint padrao caso nao fornecido
    if checkpoint_path is None:
        candidates = [
            Path(f"checkpoints/mark_{variant_norm}.pt"),
            Path(f"checkpoints/mark_engineer_{variant_norm}.pt"),
            Path(f"/mnt/ssd_1/mark ai/checkpoints/mark_{variant_norm}.pt"),
            Path(f"/mnt/ssd_1/mark ai/checkpoints/mark_engineer_{variant_norm}.pt"),
        ]
        for c in candidates:
            if c.exists():
                checkpoint_path = str(c)
                break

    if checkpoint_path and Path(checkpoint_path).exists():
        import torch
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict, strict=False)
        print(f"[Mark AI] Pesos carregados com sucesso de: {checkpoint_path}")
    else:
        print(f"[Mark AI] Inicializado com pesos estruturais para a variante '{variant_norm}' ({config.count_parameters():,} parametros).")

    if device == "auto":
        import torch
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        import torch
        dev = torch.device(device)

    model = model.to(dev)
    return model, config
