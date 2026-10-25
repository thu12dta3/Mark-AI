"""
Mark 0.1 - Root Entrypoint for 8GB VRAM Local Inference (`inference_8gb.py`)
Re-exporta `export/quantize_4bit.py` para carregamento com quantização 4-bit NF4 e PagedKVCache sob 7.5GB VRAM.
"""

from export.quantize_4bit import (
    MAX_ALLOWED_VRAM_GB,
    enforce_vram_ceiling,
    quantize_mark_model_nf4,
    load_mark_for_8gb_vram,
    MarkInferenceEngine,
)

__all__ = [
    "MAX_ALLOWED_VRAM_GB",
    "enforce_vram_ceiling",
    "quantize_mark_model_nf4",
    "load_mark_for_8gb_vram",
    "MarkInferenceEngine",
]

if __name__ == "__main__":
    import glob
    from tokenizers import Tokenizer
    from src.model.mark_arch import MarkConfig

    ckpts = sorted(glob.glob("checkpoints/mark_0_1_step_*.pt"))
    ckpt_path = ckpts[-1] if ckpts else None

    tokenizer = Tokenizer.from_file("data/tokenizer/mark_bpe_tokenizer.json")
    model, kv_cache, vram_stats = load_mark_for_8gb_vram(
        checkpoint_path=ckpt_path,
        config=MarkConfig(
            vocab_size=tokenizer.get_vocab_size(),
            dim=512,
            n_layers=6,
            n_heads=8,
            n_kv_heads=2,
            intermediate_size=1536,
            max_seq_len=1024,
        ) if not ckpt_path else None,
        max_context_tokens=1024,
    )
    print(f"[Inference 8GB] Orçamento e Alocação de Memória: {vram_stats}")

    engine = MarkInferenceEngine(model, tokenizer, kv_cache)
    prompt = "Corrija o vazamento de descritores no relay QUIC do daemon Rust."
    output = engine.generate(prompt, max_new_tokens=48, temperature=0.0)
    print(f"[Inference 8GB] Saída gerada:\n{output}")
