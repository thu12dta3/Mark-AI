"""
Mark 0.1 - Cloud A100 Training Pipeline (BF16 Mixed Precision + FlashAttention-2 + Google Drive Checkpointing)
Focado em ensinar raciocínio latente (<think>...</think>) e Self-RAG ([RETRIEVE], [EVALUATE], [ACCEPT], [REJECT]).
"""

import glob
import json
import math
import os
import time
from dataclasses import asdict
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer

from src.model.mark_arch import MarkConfig, MarkForCausalLM
from src.model.self_rag import SpecialTokens
from src.training.reasoning_loss import MarkReasoningAndSelfRAGLoss


class MarkReasoningDataset(Dataset):
    """
    Carrega amostras sintéticas destiladas (Prompt + <think> + [RETRIEVE] + <|rag_context|> + Resposta)
    e mascara:
    - O prompt do usuário (label = -100)
    - O conteúdo bruto retornado pelo RAG entre <|rag_context|> e <|/rag_context|> (label = -100)
      para que o modelo aprenda a EMITIR [RETRIEVE] e AVALIAR com [EVALUATE] [ACCEPT]/[REJECT],
      sem desperdiçar capacidade decorando textos externos.
    """
    def __init__(
        self,
        jsonl_path: str,
        tokenizer: Tokenizer,
        max_seq_len: int = 8192,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples: List[Dict[str, str]] = []

        if os.path.exists(jsonl_path):
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.samples.append(json.loads(line))

        self.pad_id = tokenizer.token_to_id(SpecialTokens.PAD) or 0
        self.bos_id = tokenizer.token_to_id(SpecialTokens.BOS) or 1
        self.eos_id = tokenizer.token_to_id(SpecialTokens.EOS) or 2
        self.ctx_start_id = tokenizer.token_to_id(SpecialTokens.CONTEXT_START)
        self.ctx_end_id = tokenizer.token_to_id(SpecialTokens.CONTEXT_END)

    def __len__(self) -> int:
        return len(self.samples)

    def _encode_pair(self, prompt: str, response: str) -> Dict[str, torch.Tensor]:
        prompt_ids = [self.bos_id] + self.tokenizer.encode(prompt).ids
        resp_ids = self.tokenizer.encode(response).ids + [self.eos_id]

        input_ids = (prompt_ids + resp_ids)[: self.max_seq_len]
        labels = ([-100] * len(prompt_ids) + resp_ids)[: self.max_seq_len]

        # Mascara blocos de contexto injetados pelo RAG externo (<|rag_context|> ... <|/rag_context|>)
        in_rag_ctx = False
        for i, tok_id in enumerate(labels):
            if tok_id == self.ctx_start_id:
                in_rag_ctx = True
            if in_rag_ctx:
                labels[i] = -100
            if tok_id == self.ctx_end_id:
                in_rag_ctx = False

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.samples[idx]
        chosen_pack = self._encode_pair(item["prompt"], item["chosen"])
        rejected_pack = self._encode_pair(item["prompt"], item.get("rejected", ""))
        return {
            "input_ids": chosen_pack["input_ids"],
            "labels": chosen_pack["labels"],
            "rejected_input_ids": rejected_pack["input_ids"],
            "rejected_labels": rejected_pack["labels"],
        }


def collate_reasoning_batch(batch: List[Dict[str, torch.Tensor]], pad_id: int = 0) -> Dict[str, torch.Tensor]:
    def _pad_list(tensors: List[torch.Tensor], pad_val: int) -> Tuple[torch.Tensor, torch.Tensor]:
        max_len = max(t.size(0) for t in tensors)
        padded = torch.full((len(tensors), max_len), pad_val, dtype=torch.long)
        mask = torch.zeros((len(tensors), max_len), dtype=torch.long)
        for i, t in enumerate(tensors):
            padded[i, : t.size(0)] = t
            mask[i, : t.size(0)] = 1
        return padded, mask

    input_ids, attention_mask = _pad_list([b["input_ids"] for b in batch], pad_id)
    labels, _ = _pad_list([b["labels"] for b in batch], -100)
    rej_ids, rej_mask = _pad_list([b["rejected_input_ids"] for b in batch], pad_id)
    rej_labels, _ = _pad_list([b["rejected_labels"] for b in batch], -100)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "rejected_input_ids": rej_ids,
        "rejected_attention_mask": rej_mask,
        "rejected_labels": rej_labels,
    }


class GoogleDriveCheckpointManager:
    """
    Gerencia salvamento resiliente e retomada automática de checkpoints no Google Drive (`/content/drive/MyDrive/Mark-AI/checkpoints`)
    para evitar perda de progresso em desconexões do Google Colab Pro / A100.
    """
    def __init__(self, checkpoint_dir: str = "/content/drive/MyDrive/Mark-AI/checkpoints", keep_last_n: int = 3):
        # Fallback local caso não esteja montado no Colab ainda
        if not os.path.exists("/content/drive") and checkpoint_dir.startswith("/content/drive"):
            checkpoint_dir = "checkpoints"
        self.checkpoint_dir = checkpoint_dir
        self.keep_last_n = keep_last_n
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def save(
        self,
        model: MarkForCausalLM,
        optimizer: torch.optim.Optimizer,
        step: int,
        loss: float,
        config: MarkConfig,
    ) -> str:
        ckpt_path = os.path.join(self.checkpoint_dir, f"mark_0_1_step_{step:06d}.pt")
        payload = {
            "step": step,
            "loss": loss,
            "config": asdict(config),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        torch.save(payload, ckpt_path)

        # Rotaciona checkpoints antigos para não lotar o Google Drive
        all_ckpts = sorted(glob.glob(os.path.join(self.checkpoint_dir, "mark_0_1_step_*.pt")))
        if len(all_ckpts) > self.keep_last_n:
            for old_ckpt in all_ckpts[: -self.keep_last_n]:
                try:
                    os.remove(old_ckpt)
                except OSError:
                    pass
        return ckpt_path

    def load_latest(
        self,
        model: MarkForCausalLM,
        optimizer: Optional[torch.optim.Optimizer] = None,
        device: torch.device = torch.device("cpu"),
    ) -> int:
        all_ckpts = sorted(glob.glob(os.path.join(self.checkpoint_dir, "mark_0_1_step_*.pt")))
        if not all_ckpts:
            return 0
        latest = all_ckpts[-1]
        ckpt = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if optimizer is not None and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        return int(ckpt.get("step", 0))


def get_cosine_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / max(1, warmup_steps)
    if step >= max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def train_mark_a100(
    dataset_path: str = "data/synthetic_dpo/mark_reasoning_train.jsonl",
    tokenizer_path: str = "data/tokenizer/mark_bpe_tokenizer.json",
    checkpoint_dir: str = "/content/drive/MyDrive/Mark-AI/checkpoints",
    max_steps: int = 1000,
    save_every_n_steps: int = 100,
    batch_size: int = 4,
    grad_accum_steps: int = 8,
    max_lr: float = 3e-4,
    use_full_3b_config: bool = True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    pt_dtype = torch.bfloat16 if use_bf16 else torch.float32

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    tokenizer = Tokenizer.from_file(tokenizer_path)

    if use_full_3b_config:
        config = MarkConfig(vocab_size=max(49152, tokenizer.get_vocab_size()))
    else:
        # Configuração compacta para validação rápida em CPU/CI
        config = MarkConfig(
            vocab_size=tokenizer.get_vocab_size(),
            dim=512,
            n_layers=6,
            n_heads=8,
            n_kv_heads=2,
            intermediate_size=1536,
            max_seq_len=1024,
        )

    model = MarkForCausalLM(config).to(device=device, dtype=pt_dtype if use_bf16 else torch.float32)
    model.enable_gradient_checkpointing()

    # Fused AdamW nativo para GPUs Ampere (A100)
    use_fused = device.type == "cuda"
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=max_lr,
        betas=(0.9, 0.95),
        weight_decay=0.1,
        fused=use_fused,
    )

    criterion = MarkReasoningAndSelfRAGLoss(config)
    ckpt_manager = GoogleDriveCheckpointManager(checkpoint_dir=checkpoint_dir)
    start_step = ckpt_manager.load_latest(model, optimizer, device=device)

    dataset = MarkReasoningDataset(dataset_path, tokenizer, max_seq_len=config.max_seq_len)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_reasoning_batch(b, pad_id=config.pad_token_id),
    )

    model.train()
    step = start_step
    data_iter = iter(loader)

    while step < max_steps:
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        last_metrics: Dict[str, float] = {}

        for _ in range(grad_accum_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            rej_ids = batch["rejected_input_ids"].to(device)
            rej_mask = batch["rejected_attention_mask"].to(device)
            rej_labels = batch["rejected_labels"].to(device)

            with torch.amp.autocast(device_type=device.type, dtype=pt_dtype, enabled=use_bf16):
                chosen_logits = model(input_ids, attention_mask=attention_mask)
                rejected_logits = model(rej_ids, attention_mask=rej_mask)
                loss, metrics = criterion(
                    chosen_logits,
                    labels,
                    rejected_logits=rejected_logits,
                    rejected_labels=rej_labels,
                )
                scaled_loss = loss / grad_accum_steps

            scaled_loss.backward()
            accum_loss += scaled_loss.item()
            last_metrics = metrics

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        lr = get_cosine_lr(step, warmup_steps=max(10, max_steps // 10), max_steps=max_steps, max_lr=max_lr, min_lr=max_lr * 0.1)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.step()
        step += 1

        if step % 10 == 0 or step == 1:
            print(
                f"[Step {step:05d}/{max_steps:05d}] "
                f"Loss: {accum_loss:.4f} | "
                f"CE: {last_metrics.get('loss_weighted_ce', 0):.4f} | "
                f"ORPO: {last_metrics.get('loss_orpo', 0):.4f} | "
                f"P(<think>): {last_metrics.get('think_initiation_prob', 0):.3f} | "
                f"LR: {lr:.2e}"
            )

        if step % save_every_n_steps == 0 or step == max_steps:
            saved_path = ckpt_manager.save(model, optimizer, step, accum_loss, config)
            print(f"[Checkpoint] Salvo com segurança em: {saved_path}")

    return model
