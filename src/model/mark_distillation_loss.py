"""
Mark-Engineer-V2-8B: Frontier Distillation & Multi-Objective Loss Formulation
=============================================================================
This module implements the exact loss formulation designed for Mark-Engineer-V2-8B:
1. Label-Smoothed Cross-Entropy Loss for ground-truth next-token prediction.
2. Top-K Forward KL Divergence Distillation (Temperature T=2.0) against frontier teacher logits
   (DeepSeek-V3/V4, Arcee packed formats, Qwen-Max, Claude/GPT-5 synthetic traces).
3. Soft-Capping / Tanh logit regularization to prevent neuron saturation.
4. Multi-Token Prediction (MTP Depth 1) future-token loss.
5. MoE Router Z-Loss and auxiliary load balancing integration.
"""

from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class MarkEngineerDistillationLoss(nn.Module):
    """
    Composite distillation loss module for Mark-Engineer-V2-8B.
    Guarantees numerical stability in bfloat16/float16 by upcasting softmax
    and KL divergence reductions to float32.
    """
    def __init__(
        self,
        temperature: float = 2.0,
        kd_weight: float = 0.45,
        ce_weight: float = 0.55,
        label_smoothing: float = 0.02,
        mtp_weight: float = 0.30,
        router_z_weight: float = 1.0,
        logit_soft_cap: float = 30.0,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.temperature = float(temperature)
        self.kd_weight = float(kd_weight)
        self.ce_weight = float(ce_weight)
        self.label_smoothing = float(label_smoothing)
        self.mtp_weight = float(mtp_weight)
        self.router_z_weight = float(router_z_weight)
        self.logit_soft_cap = float(logit_soft_cap)
        self.ignore_index = int(ignore_index)

    def _soft_cap_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Applies tanh soft-capping to prevent logit explosion and neuron saturation:
        z' = C * tanh(z / C)
        """
        if self.logit_soft_cap > 0.0:
            return self.logit_soft_cap * torch.tanh(logits / self.logit_soft_cap)
        return logits

    def forward(
        self,
        student_logits: torch.Tensor,
        labels: torch.Tensor,
        teacher_indices: Optional[torch.Tensor] = None,
        teacher_values: Optional[torch.Tensor] = None,
        mtp_logits: Optional[torch.Tensor] = None,
        router_z_loss: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            student_logits: (bsz, seqlen, vocab_size) - raw logits from student model
            labels: (bsz, seqlen) - ground truth target token ids (-100 for ignore)
            teacher_indices: Optional (bsz, seqlen, top_k) - top-K token IDs from teacher
            teacher_values: Optional (bsz, seqlen, top_k) - unnormalized top-K teacher logits
            mtp_logits: Optional (bsz, seqlen - 1, vocab_size) - prediction for t+2
            router_z_loss: Optional scalar tensor - Router Z-loss aggregated across MoE layers

        Returns:
            total_loss: scalar tensor with gradient graph attached
            metrics: dictionary of detached float metrics for logging
        """
        bsz, seqlen, vocab_size = student_logits.shape
        metrics: Dict[str, float] = {}

        # 1. Apply Logit Soft-Capping (Anti-Saturation)
        capped_student_logits = self._soft_cap_logits(student_logits)

        # 2. Shift logits and labels for next-token prediction
        # student_logits[:, :-1, :] predicts labels[:, 1:]
        shift_student = capped_student_logits[:, :-1, :].contiguous().view(-1, vocab_size)
        shift_labels = labels[:, 1:].contiguous().view(-1)

        valid_mask = (shift_labels != self.ignore_index)
        n_valid_tokens = valid_mask.sum().clamp(min=1.0)

        # 3. Label-Smoothed Cross-Entropy Loss
        ce_loss = F.cross_entropy(
            shift_student,
            shift_labels,
            label_smoothing=self.label_smoothing,
            ignore_index=self.ignore_index,
            reduction="mean",
        )
        metrics["loss_ce"] = float(ce_loss.detach().item())

        total_loss = self.ce_weight * ce_loss

        # 4. Top-K Teacher Distillation (Forward KL Divergence)
        if teacher_indices is not None and teacher_values is not None:
            # Shift teacher tensors: teacher on token t guides student predicting token t+1
            shift_t_idx = teacher_indices[:, 1:, :].contiguous().view(-1, teacher_indices.size(-1))
            shift_t_val = teacher_values[:, 1:, :].contiguous().view(-1, teacher_values.size(-1))

            # Filter valid tokens (non-masked)
            valid_t_idx = shift_t_idx[valid_mask]      # (N_valid, top_k)
            valid_t_val = shift_t_val[valid_mask].to(torch.float32)  # (N_valid, top_k)

            # Clamp valid indices to prevent out-of-bounds indexing in student vocab
            valid_t_idx = valid_t_idx.clamp(min=0, max=vocab_size - 1)

            # Gather student logits corresponding to the teacher's top-K candidate tokens
            valid_student_subset = shift_student[valid_mask]  # (N_valid, vocab_size)
            gathered_s_logits = torch.gather(valid_student_subset, dim=-1, index=valid_t_idx).to(torch.float32)

            # Compute distributions with Temperature T over the top-K subset
            T = self.temperature
            p_teacher = F.softmax(valid_t_val / T, dim=-1)
            log_q_student = F.log_softmax(gathered_s_logits / T, dim=-1)

            # Forward KL Divergence: D_KL(P_teacher || Q_student) = sum P * (log P - log Q)
            # P_teacher * log(P_teacher) is constant w.r.t student weights, so minimizing
            # cross-entropy -sum(P_teacher * log_Q_student) is mathematically equivalent.
            # Using full KL preserves exact metric interpretability.
            log_p_teacher = F.log_softmax(valid_t_val / T, dim=-1)
            kl_div = (p_teacher * (log_p_teacher - log_q_student)).sum(dim=-1).mean()

            # Scale by T^2 to preserve gradient magnitude balance
            kd_loss = (T ** 2) * kl_div
            total_loss = total_loss + (self.kd_weight * kd_loss)
            metrics["loss_kd"] = float(kd_loss.detach().item())
            metrics["kl_div_raw"] = float(kl_div.detach().item())
        else:
            # Fallback to full CE weight when teacher data is not in current batch
            total_loss = ce_loss

        # 5. Multi-Token Prediction (MTP Depth 1) Loss
        if mtp_logits is not None and seqlen > 2:
            capped_mtp = self._soft_cap_logits(mtp_logits)
            # mtp predicts token t+2 given context up to t
            shift_mtp = capped_mtp[:, :-1, :].contiguous().view(-1, vocab_size)
            shift_labels_t2 = labels[:, 2:].contiguous().view(-1)

            mtp_loss = F.cross_entropy(
                shift_mtp,
                shift_labels_t2,
                label_smoothing=self.label_smoothing,
                ignore_index=self.ignore_index,
                reduction="mean",
            )
            total_loss = total_loss + (self.mtp_weight * mtp_loss)
            metrics["loss_mtp"] = float(mtp_loss.detach().item())

        # 6. MoE Router Z-Loss
        if router_z_loss is not None:
            z_term = self.router_z_weight * router_z_loss
            total_loss = total_loss + z_term
            metrics["loss_router_z"] = float(z_term.detach().item())

        metrics["loss_total"] = float(total_loss.detach().item())
        return total_loss, metrics
