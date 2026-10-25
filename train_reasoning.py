"""
Mark 0.1 - Root Entrypoint for Reasoning & Self-RAG Cloud Training (`train_reasoning.py`)
Re-exporta `src/training/train_cloud.py` e `src/training/reasoning_loss.py`.
"""

from src.training.reasoning_loss import MarkReasoningAndSelfRAGLoss
from src.training.train_cloud import (
    MarkReasoningDataset,
    collate_reasoning_batch,
    GoogleDriveCheckpointManager,
    get_cosine_lr,
    train_mark_a100,
)

__all__ = [
    "MarkReasoningAndSelfRAGLoss",
    "MarkReasoningDataset",
    "collate_reasoning_batch",
    "GoogleDriveCheckpointManager",
    "get_cosine_lr",
    "train_mark_a100",
]

if __name__ == "__main__":
    import os
    from tokenizer_setup import build_mark_bpe_tokenizer, generate_synthetic_reasoning_dataset

    if not os.path.exists("data/synthetic_dpo/mark_reasoning_train.jsonl"):
        generate_synthetic_reasoning_dataset()
    if not os.path.exists("data/tokenizer/mark_bpe_tokenizer.json"):
        build_mark_bpe_tokenizer()

    print("[Train Reasoning] Executando validação de treinamento end-to-end (<think> + Self-RAG + ORPO)...")
    train_mark_a100(
        max_steps=10,
        save_every_n_steps=5,
        batch_size=2,
        grad_accum_steps=1,
        use_full_3b_config=False,
    )
