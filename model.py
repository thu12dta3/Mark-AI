"""
Mark 0.1 - Root Entrypoint for Neural Architecture (`model.py`)
Re-exporta a topologia DeepSeek-V3/R1 (MLA, DeepSeekMoE, MTP), Visão Multimodal (PixelShuffle 2x2)
e Adaptador de Voz (ElevenLabs / Fish Speech).
"""

from src.model.mark_arch import (
    MarkConfig,
    RMSNorm,
    precompute_freqs_cis,
    apply_rotary_emb,
    PagedKVCache,
    DeepSeekMLAAttention,
    MarkSelfAttention,
    MarkSwiGLUFFN,
    DeepSeekMoELayer,
    MarkVisionEncoder,
    DeepSeekMTPModule,
    MarkTransformerBlock,
    MarkModel,
    MarkForCausalLM,
)
from src.model.self_rag import (
    SpecialTokens,
    ElevenLabsFishVoiceFormatter,
    SelfRAGController,
)

__all__ = [
    "MarkConfig",
    "RMSNorm",
    "precompute_freqs_cis",
    "apply_rotary_emb",
    "PagedKVCache",
    "DeepSeekMLAAttention",
    "MarkSelfAttention",
    "MarkSwiGLUFFN",
    "DeepSeekMoELayer",
    "MarkVisionEncoder",
    "DeepSeekMTPModule",
    "MarkTransformerBlock",
    "MarkModel",
    "MarkForCausalLM",
    "SpecialTokens",
    "ElevenLabsFishVoiceFormatter",
    "SelfRAGController",
]

if __name__ == "__main__":
    cfg = MarkConfig()
    print(f"[Mark 0.1] Total Parameters (MoE + MLA + MTP + Vision): {cfg.count_parameters():,} (~{cfg.count_parameters()/1e9:.2f}B)")
    print(f"[Mark 0.1] 4-bit NF4 Weights Est.: {(cfg.count_parameters() * 0.55) / (1024**3):.2f} GB")
    mla_8k_mb = (cfg.n_layers * 8192 * cfg.mla_cache_dim * 2) / (1024**2)
    mla_32k_mb = (cfg.n_layers * 32768 * cfg.mla_cache_dim * 2) / (1024**2)
    print(f"[Mark 0.1] DeepSeek MLA Latent KV Cache @ 8k tokens:  {mla_8k_mb:.1f} MB")
    print(f"[Mark 0.1] DeepSeek MLA Latent KV Cache @ 32k tokens: {mla_32k_mb:.1f} MB")
