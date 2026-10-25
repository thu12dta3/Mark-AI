"""
Mark 0.1 - Root Entrypoint for Neural Architecture (`model.py`)
Re-exporta a topologia completa definida em `src/model/mark_arch.py` e `src/model/self_rag.py`.
"""

from src.model.mark_arch import (
    MarkConfig,
    RMSNorm,
    precompute_freqs_cis,
    apply_rotary_emb,
    PagedKVCache,
    MarkSelfAttention,
    MarkSwiGLUFFN,
    MarkTransformerBlock,
    MarkModel,
    MarkForCausalLM,
)
from src.model.self_rag import SpecialTokens, SelfRAGController

__all__ = [
    "MarkConfig",
    "RMSNorm",
    "precompute_freqs_cis",
    "apply_rotary_emb",
    "PagedKVCache",
    "MarkSelfAttention",
    "MarkSwiGLUFFN",
    "MarkTransformerBlock",
    "MarkModel",
    "MarkForCausalLM",
    "SpecialTokens",
    "SelfRAGController",
]

if __name__ == "__main__":
    cfg = MarkConfig()
    print(f"[Mark 0.1] Total Active Parameters: {cfg.count_parameters():,} (~{cfg.count_parameters()/1e9:.2f}B)")
    print(f"[Mark 0.1] 4-bit NF4 Weights Est.: {(cfg.count_parameters() * 0.55) / (1024**3):.2f} GB")
    kv_8k_mb = (2 * cfg.n_layers * cfg.n_kv_heads * cfg.head_dim * 8192 * 2) / (1024**2)
    kv_32k_mb = (2 * cfg.n_layers * cfg.n_kv_heads * cfg.head_dim * 32768 * 2) / (1024**2)
    print(f"[Mark 0.1] KV Cache @ 8k tokens (FP16, GQA 8 heads):  {kv_8k_mb:.1f} MB")
    print(f"[Mark 0.1] KV Cache @ 32k tokens (FP16, GQA 8 heads): {kv_32k_mb:.1f} MB")
