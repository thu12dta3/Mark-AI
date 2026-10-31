"""
Hugging Face Transformers Compatible Modeling Implementation for Mark-Engineer-V2-8B
Self-contained causal language model architecture with Multi-Head Latent Attention (MLA),
DeepSeekMoE (1 Shared + 24 Routed Experts, Top-2 active), and Multi-Token Prediction (MTP).
"""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

try:
    from .configuration_mark_engineer_v2_8b import MarkEngineerV2_8BConfig
except (ImportError, ValueError):
    from configuration_mark_engineer_v2_8b import MarkEngineerV2_8BConfig


class MarkRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * self.weight


def precompute_rope_freqs(head_dim: int, max_seq_len: int, theta: float = 500000.0) -> Tuple[torch.Tensor, torch.Tensor]:
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float)
    freqs = torch.outer(t, freqs)
    cos = freqs.cos()
    sin = freqs.sin()
    return cos, sin


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    cos = torch.cat([cos, cos], dim=-1)
    sin = torch.cat([sin, sin], dim=-1)
    return (x * cos) + (rotated * sin)


class MarkSwiGLUFFN(nn.Module):
    def __init__(self, dim: int, intermediate_size: int):
        super().__init__()
        self.w_gate = nn.Linear(dim, intermediate_size, bias=False)
        self.w_up = nn.Linear(dim, intermediate_size, bias=False)
        self.w_down = nn.Linear(intermediate_size, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class MarkMLAAttention(nn.Module):
    def __init__(self, config: MarkEngineerV2_8BConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.dim = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

        self.wq_a = nn.Linear(self.dim, self.q_lora_rank, bias=False)
        self.q_norm = MarkRMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.qk_head_dim, bias=False)

        self.wkv_a = nn.Linear(self.dim, self.kv_lora_rank + self.qk_rope_head_dim, bias=False)
        self.kv_norm = MarkRMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.wkv_b = nn.Linear(
            self.kv_lora_rank,
            self.n_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )

        self.wo = nn.Linear(self.n_heads * self.v_head_dim, self.dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape

        q_latent = self.q_norm(self.wq_a(x))
        q = self.wq_b(q_latent).view(bsz, seqlen, self.n_heads, self.qk_head_dim)
        q_nope = q[..., : self.qk_nope_head_dim]
        q_pe = q[..., self.qk_nope_head_dim :]

        kv_combined = self.wkv_a(x)
        kv_latent = self.kv_norm(kv_combined[..., : self.kv_lora_rank])
        k_pe = kv_combined[..., self.kv_lora_rank :].view(bsz, seqlen, 1, self.qk_rope_head_dim)

        kv = self.wkv_b(kv_latent).view(bsz, seqlen, self.n_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope = kv[..., : self.qk_nope_head_dim]
        v = kv[..., self.qk_nope_head_dim :]

        q_pe = apply_rotary_emb(q_pe, cos[:seqlen], sin[:seqlen])
        k_pe = apply_rotary_emb(k_pe.expand(-1, -1, self.n_heads, -1), cos[:seqlen], sin[:seqlen])

        k = torch.cat([k_nope, k_pe], dim=-1)
        q = torch.cat([q_nope, q_pe], dim=-1)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.qk_head_dim)
        if attention_mask is not None:
            scores = scores + attention_mask

        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class MarkMoELayer(nn.Module):
    def __init__(self, config: MarkEngineerV2_8BConfig):
        super().__init__()
        self.n_routed = config.n_routed_experts
        self.n_shared = config.n_shared_experts
        self.top_k = config.num_experts_per_tok
        self.dim = config.hidden_size
        self.moe_intermediate = config.moe_intermediate_size

        self.gate = nn.Linear(self.dim, self.n_routed, bias=False)
        self.register_buffer("router_bias", torch.zeros(self.n_routed))
        self.router_bias_rate = config.router_bias_update_rate

        self.routed_experts = nn.ModuleList([
            MarkSwiGLUFFN(self.dim, self.moe_intermediate) for _ in range(self.n_routed)
        ])
        if self.n_shared > 0:
            self.shared_experts = MarkSwiGLUFFN(self.dim, self.moe_intermediate * self.n_shared)
        else:
            self.shared_experts = None

        self.last_router_z_loss = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, dim = x.shape
        flat_x = x.view(-1, dim)

        logits = self.gate(flat_x)
        if self.training:
            self.last_router_z_loss = torch.logsumexp(logits, dim=-1).pow(2).mean()

        biased_logits = logits + self.router_bias
        topk_weights, topk_indices = torch.topk(F.softmax(biased_logits, dim=-1), self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        out = torch.zeros_like(flat_x)
        for i, expert in enumerate(self.routed_experts):
            mask = (topk_indices == i).any(dim=-1)
            if mask.any():
                idx_in_topk = (topk_indices[mask] == i).nonzero(as_tuple=True)[1]
                w = topk_weights[mask, idx_in_topk].unsqueeze(-1)
                out[mask] += w * expert(flat_x[mask])

        if self.shared_experts is not None:
            out += self.shared_experts(flat_x)

        return out.view(bsz, seqlen, dim)


class MarkEngineerV2Block(nn.Module):
    def __init__(self, config: MarkEngineerV2_8BConfig, layer_idx: int):
        super().__init__()
        self.attn_norm = MarkRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = MarkMLAAttention(config, layer_idx)
        self.ffn_norm = MarkRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if layer_idx < config.n_dense_layers:
            self.ffn = MarkSwiGLUFFN(config.hidden_size, config.intermediate_size)
            self.is_moe = False
        else:
            self.ffn = MarkMoELayer(config)
            self.is_moe = True

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = x + self.attn(self.attn_norm(x), cos, sin, attention_mask)
        out = h + self.ffn(self.ffn_norm(h))
        return out


class MarkEngineerV2ForCausalLM(PreTrainedModel):
    config_class = MarkEngineerV2_8BConfig
    base_model_prefix = "model"
    _supports_flash_attn_2 = False

    def __init__(self, config: MarkEngineerV2_8BConfig):
        super().__init__(config)
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            MarkEngineerV2Block(config, i) for i in range(config.num_hidden_layers)
        ])
        self.norm = MarkRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.mtp_depth > 0:
            self.mtp_proj = nn.Linear(config.hidden_size * 2, config.hidden_size, bias=False)
            self.mtp_norm = MarkRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.mtp_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        else:
            self.mtp_proj = None

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

        cos, sin = precompute_rope_freqs(
            config.qk_rope_head_dim,
            config.max_position_embeddings,
            theta=config.rope_theta,
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.post_init()

    def load_mark_state_dict(self, state_dict: dict, strict: bool = False):
        mapped_sd = {}
        for k, v in state_dict.items():
            new_k = k
            if new_k.startswith('tok_embeddings.'):
                new_k = new_k.replace('tok_embeddings.', 'embed_tokens.')
            new_k = new_k.replace('.ffn.router.', '.ffn.gate.')
            new_k = new_k.replace('.ffn.shared_experts.w_', '.ffn.shared_experts.0.w_')
            mapped_sd[new_k] = v
        return self.load_state_dict(mapped_sd, strict=strict)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        return_mtp: bool = False,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        bsz, seqlen = input_ids.shape
        x = self.embed_tokens(input_ids)

        if attention_mask is None:
            causal_mask = torch.triu(
                torch.full((seqlen, seqlen), float("-inf"), device=input_ids.device),
                diagonal=1,
            ).unsqueeze(0).unsqueeze(0)
        else:
            causal_mask = attention_mask

        for layer in self.layers:
            x = layer(x, self.rope_cos, self.rope_sin, causal_mask)

        h = self.norm(x)
        logits = self.lm_head(h)

        mtp_logits = None
        if return_mtp and self.mtp_proj is not None and seqlen > 1:
            next_emb = self.embed_tokens(input_ids[:, 1:])
            h_prev = h[:, :-1]
            cat_h = torch.cat([h_prev, next_emb], dim=-1)
            h_mtp = self.mtp_norm(self.mtp_proj(cat_h))
            mtp_logits = self.mtp_head(h_mtp)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

            if mtp_logits is not None and seqlen > 2:
                shift_mtp_logits = mtp_logits[..., :-1, :].contiguous()
                shift_mtp_labels = labels[..., 2:].contiguous()
                loss_mtp = loss_fct(shift_mtp_logits.view(-1, self.config.vocab_size), shift_mtp_labels.view(-1))
                loss = loss + (self.config.mtp_loss_weight * loss_mtp)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
        )
