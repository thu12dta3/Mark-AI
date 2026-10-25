"""
Mark 0.1 - High-Performance Cloud A100 Training Pipeline
Integra:
- Precisão mista `bfloat16` + TensorFloat-32 (`tf32`)
- DeepSeek Multi-Head Latent Attention (MLA) + DeepSeekMoE Aux-Loss-Free + Multi-Token Prediction (MTP)
- Mascaramento seletivo de `<|rag_context|>` para ensinar Self-RAG real sem decorar textos estáticos
- Checkpointing rotativo persistente com verificação de sanidade de gradientes (zero NaN/Inf)
"""

import glob
import json
import math
import os
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer

from src.model.mark_arch import MarkConfig, MarkForCausalLM
from src.model.self_rag import SpecialTokens
from src.training.reasoning_loss import MarkReasoningAndSelfRAGLoss


class MarkReasoningDataset(Dataset):
    """
    Carrega amostras sintéticas poliglotas e multimodais (Prompt + <think> + [RETRIEVE] + <|rag_context|> + Resposta)
    e mascara com -100 os tokens de prompt e os blocos brutos retornados pelo RAG externo.
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


LOCAL_SATA_SSD_DIR = "/mnt/ssd_1/mark ai/checkpoints"


class GoogleDriveCheckpointManager:
    """
    Gerenciador de persistência rotativa com suporte a:
    1. Nuvem (Google Drive: `/content/drive/MyDrive/Mark-AI/checkpoints` ou fallback `/content/checkpoints`).
    2. Local rígido: grava única e estritamente no SSD SATA de 256GB sem Windows (`/mnt/ssd_1/mark ai/checkpoints`).
    """
    def __init__(
        self,
        checkpoint_dir: str = "/content/drive/MyDrive/Mark-AI/checkpoints",
        local_ssd_dir: Optional[str] = None,
        keep_last_n: int = 3,
    ):
        if not os.path.exists("/content/drive") and checkpoint_dir.startswith("/content/drive"):
            checkpoint_dir = "/content/checkpoints"
        self.checkpoint_dir = checkpoint_dir

        # Destino local estrito no SSD SATA secundário de 256GB
        if local_ssd_dir is None and os.path.exists("/mnt/ssd_1/mark ai"):
            local_ssd_dir = LOCAL_SATA_SSD_DIR
        self.local_ssd_dir = local_ssd_dir

        self.keep_last_n = keep_last_n
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        if self.local_ssd_dir and os.path.exists(os.path.dirname(self.local_ssd_dir)):
            try:
                os.makedirs(self.local_ssd_dir, exist_ok=True)
            except OSError:
                pass

    def save(
        self,
        model: MarkForCausalLM,
        optimizer: torch.optim.Optimizer,
        step: int,
        loss: float,
        config: MarkConfig,
    ) -> str:
        filename = f"mark_0_1_step_{step:06d}.pt"
        ckpt_path = os.path.join(self.checkpoint_dir, f"mark_0_1_step_{step:06d}.pt")
        payload = {
            "step": step,
            "loss": loss,
            "config": asdict(config),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        torch.save(payload, ckpt_path)

        # Se o SSD SATA local estiver montado, espelha para /mnt/ssd_1/mark ai/checkpoints
        if self.local_ssd_dir and os.path.exists(self.local_ssd_dir):
            try:
                import shutil
                local_path = os.path.join(self.local_ssd_dir, filename)
                shutil.copyfile(ckpt_path, local_path)
            except Exception:
                pass

        all_ckpts = sorted(glob.glob(os.path.join(self.checkpoint_dir, "mark_0_1_step_*.pt")))
        if len(all_ckpts) > self.keep_last_n:
            for old_ckpt in all_ckpts[: -self.keep_last_n]:
                try:
                    os.remove(old_ckpt)
                except OSError:
                    pass

        if self.local_ssd_dir and os.path.exists(self.local_ssd_dir):
            local_ckpts = sorted(glob.glob(os.path.join(self.local_ssd_dir, "mark_0_1_step_*.pt")))
            if len(local_ckpts) > self.keep_last_n:
                for old_local in local_ckpts[: -self.keep_last_n]:
                    try:
                        os.remove(old_local)
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
        try:
            ckpt = torch.load(latest, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
            if optimizer is not None and "optimizer_state_dict" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            return int(ckpt.get("step", 0))
        except Exception:
            return 0


def get_cosine_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / max(1, warmup_steps)
    if step >= max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def train_mark_a100(
    dataset_path: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    checkpoint_dir: str = "/content/drive/MyDrive/Mark-AI/checkpoints",
    max_steps: int = 1000,
    save_every_n_steps: int = 100,
    batch_size: int = 4,
    grad_accum_steps: int = 8,
    max_lr: float = 3e-4,
    use_full_3b_config: bool = True,
):
    repo_root = Path(__file__).resolve().parent.parent.parent

    if dataset_path is None:
        dataset_path = str(repo_root / "data" / "synthetic_dpo" / "mark_reasoning_train.jsonl")
    elif not os.path.isabs(dataset_path):
        dataset_path = str((repo_root / dataset_path).resolve())

    if tokenizer_path is None:
        tokenizer_path = str(repo_root / "data" / "tokenizer" / "mark_bpe_tokenizer.json")
    elif not os.path.isabs(tokenizer_path):
        tokenizer_path = str((repo_root / tokenizer_path).resolve())

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
        config = MarkConfig(
            vocab_size=tokenizer.get_vocab_size(),
            dim=512,
            n_layers=6,
            n_heads=8,
            q_lora_rank=128,
            kv_lora_rank=128,
            qk_nope_head_dim=32,
            qk_rope_head_dim=32,
            v_head_dim=64,
            n_dense_layers=2,
            n_routed_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=512,
            intermediate_size=1536,
            max_seq_len=1024,
            vision_enabled=False,
        )

    model = MarkForCausalLM(config).to(device=device, dtype=pt_dtype if use_bf16 else torch.float32)
    model.enable_gradient_checkpointing()

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
                chosen_out = model(input_ids, attention_mask=attention_mask, return_mtp=True)
                if isinstance(chosen_out, tuple):
                    chosen_logits, mtp_logits = chosen_out
                else:
                    chosen_logits, mtp_logits = chosen_out, None

                rejected_logits = model(rej_ids, attention_mask=rej_mask, return_mtp=False)
                loss, metrics = criterion(
                    chosen_logits,
                    labels,
                    rejected_logits=rejected_logits,
                    rejected_labels=rej_labels,
                    mtp_logits=mtp_logits,
                )
                router_z_loss = model.get_router_z_loss()
                loss = loss + router_z_loss
                metrics["loss_router_z"] = float(router_z_loss.item())
                scaled_loss = loss / grad_accum_steps

            scaled_loss.backward()
            accum_loss += scaled_loss.item()
            last_metrics = metrics

        # Proteção contra gradientes não-finitos (regra de segurança de TRAINING.md do MARK-PUBLIC)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        if not torch.isfinite(grad_norm):
            print(f"[Step {step:05d}] Aviso: Gradiente não-finito detectado; pulando atualização de pesos.")
            optimizer.zero_grad(set_to_none=True)
            continue

        lr = get_cosine_lr(
            step,
            warmup_steps=max(10, max_steps // 10),
            max_steps=max_steps,
            max_lr=max_lr,
            min_lr=max_lr * 0.1,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.step()
        step += 1

        if step % 10 == 0 or step == 1:
            print(
                f"[Step {step:05d}/{max_steps:05d}] "
                f"Loss: {accum_loss:.4f} | "
                f"CE: {last_metrics.get('loss_weighted_ce', 0):.4f} | "
                f"MTP: {last_metrics.get('loss_mtp', 0):.4f} | "
                f"GRPO/ORPO: {last_metrics.get('loss_orpo', 0):.4f} | "
                f"P(<think>): {last_metrics.get('think_initiation_prob', 0):.3f} | "
                f"LR: {lr:.2e}"
            )

        if step % save_every_n_steps == 0 or step == max_steps:
            saved_path = ckpt_manager.save(model, optimizer, step, accum_loss, config)
            print(f"[Checkpoint] Salvo com segurança em: {saved_path}")

    return model
