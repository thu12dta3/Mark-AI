"""
Source architecture file for Mark-SLM-3M (< 3M parameter ceiling).
Defines Mark3MConfig and Mark3MForCausalLM.
"""

import sys
from pathlib import Path

models_dir = Path(__file__).resolve().parent.parent.parent / "models" / "mark-slm-3m"
if str(models_dir) not in sys.path:
    sys.path.insert(0, str(models_dir))

from configuration_mark_3m import Mark3MConfig
from modeling_mark_3m import (
    RMSNorm,
    Mark3MYaRNRotaryEmbedding,
    Mark3MMLA,
    Mark3MSwiGLU,
    Mark3MMoE,
    Mark3MBlock,
    Mark3MModel,
    Mark3MForCausalLM,
    get_walsh_hadamard_matrix,
)

__all__ = [
    "Mark3MConfig",
    "RMSNorm",
    "Mark3MYaRNRotaryEmbedding",
    "Mark3MMLA",
    "Mark3MSwiGLU",
    "Mark3MMoE",
    "Mark3MBlock",
    "Mark3MModel",
    "Mark3MForCausalLM",
    "get_walsh_hadamard_matrix",
]
