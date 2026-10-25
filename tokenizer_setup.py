"""
Mark 0.1 - Root Entrypoint for Tokenizer & Synthetic Distillation Prep (`tokenizer_setup.py`)
Re-exporta e executa o pipeline de `src/tokenizer/build_bpe.py`.
"""

from src.tokenizer.build_bpe import (
    DOMAIN_CODE_SEED_CORPUS,
    iter_repository_files,
    build_mark_bpe_tokenizer,
    generate_synthetic_reasoning_dataset,
)

__all__ = [
    "DOMAIN_CODE_SEED_CORPUS",
    "iter_repository_files",
    "build_mark_bpe_tokenizer",
    "generate_synthetic_reasoning_dataset",
]

if __name__ == "__main__":
    jsonl_path = generate_synthetic_reasoning_dataset()
    tokenizer = build_mark_bpe_tokenizer()
    sample = (
        "<think>\nVerificando concorrência em src-rust/mark_core/src/daemon.rs\n</think>\n"
        "[RETRIEVE] src-rust/mark_core/src/daemon.rs [/RETRIEVE]"
    )
    encoded = tokenizer.encode(sample)
    print(f"[Tokenizer Setup] Synthetic DPO dataset salvo em: {jsonl_path}")
    print(f"[Tokenizer Setup] Vocabulário treinado: {tokenizer.get_vocab_size()} tokens")
    print(f"[Tokenizer Setup] Tokens de controle verificados: {encoded.tokens}")
