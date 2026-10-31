"""
Mark-v3-Core-1.1B Neural Architecture (deepseek2 Graph llama.cpp / Ollama Native)
================================================================================
"""
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mark_arch import RMSNorm


@dataclass
class MarkV3Config:
    model_type: str = "deepseek2"
    model_name: str = "Mark-v3-Core-1.1B"
    vocab_size: int = 32032
    hidden_size: int = 1152
    num_hidden_layers: int = 18
    num_attention_heads: int = 18
    num_key_value_heads: int = 18
    intermediate_size: int = 3072

    # Multi-Head Latent Attention (MLA)
    q_lora_rank: int = 384
    kv_lora_rank: int = 192
    qk_nope_head_dim: int = 64
    qk_rope_head_dim: int = 64
    v_head_dim: int = 64

    # DeepSeekMoE (16 Routed + 2 Shared)
    n_routed_experts: int = 16
    num_experts_per_tok: int = 2
    n_shared_experts: int = 2
    moe_intermediate_size: int = 512
    aux_loss_alpha: float = 0.001

    # Geometry & YaRN RoPE
    max_position_embeddings: int = 32768
    original_max_position_embeddings: int = 4096
    rope_theta: float = 10000.0
    rope_scaling_factor: float = 8.0
    rms_norm_eps: float = 1e-6
    tie_word_embeddings: bool = True

    # Tokens
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0

    @property
    def dim(self) -> int:
        return self.hidden_size

    @property
    def n_layers(self) -> int:
        return self.num_hidden_layers

    @property
    def n_heads(self) -> int:
        return self.num_attention_heads

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim


class MarkV3YaRNRotaryEmbedding(nn.Module):
    def __init__(self, config: MarkV3Config):
        super().__init__()
        self.dim = config.qk_rope_head_dim
        self.max_position_embeddings = config.max_position_embeddings
        self.original_max_position_embeddings = config.original_max_position_embeddings
        self.base = config.rope_theta
        self.scale = config.rope_scaling_factor

        pos_freqs = self.base ** (torch.arange(0, self.dim, 2).float() / self.dim)
        inv_freq = 1.0 / pos_freqs
        low_rot = 1.0
        high_rot = 32.0
        ramp = torch.clamp((self.dim / (2 * math.pi * pos_freqs) - low_rot) / (high_rot - low_rot), 0.0, 1.0)
        yarn_inv_freq = inv_freq * ((1.0 - ramp) + ramp / self.scale)
        self.register_buffer("inv_freq", yarn_inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        return cos, sin


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    x_f = x.float().reshape(*x.shape[:-1], -1, 2)
    cos = cos.unsqueeze(0).unsqueeze(2).to(x.device)
    sin = sin.unsqueeze(0).unsqueeze(2).to(x.device)

    x_r, x_i = x_f[..., 0], x_f[..., 1]
    x_out = torch.stack([x_r * cos - x_i * sin, x_r * sin + x_i * cos], dim=-1).flatten(3)
    return x_out.to(dtype)


class MarkV3MLA(nn.Module):
    def __init__(self, config: MarkV3Config, layer_idx: int = 0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.qk_head_dim = config.qk_head_dim

        self.wq_a = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
        self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.wq_b = nn.Linear(self.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)

        self.q_b_layernorm = RMSNorm(self.qk_nope_head_dim, eps=config.rms_norm_eps)
        self.k_b_layernorm = RMSNorm(self.qk_nope_head_dim, eps=config.rms_norm_eps)

        self.wkv_a = nn.Linear(self.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias=False)
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.wkv_b = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )

        self.wo = nn.Linear(self.num_heads * self.v_head_dim, self.hidden_size, bias=False)
        self.softmax_scale = self.qk_head_dim ** (-0.5)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape

        q = self.wq_b(self.q_a_layernorm(self.wq_a(x)))
        q = q.view(bsz, seqlen, self.num_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        q_nope = self.q_b_layernorm(q_nope)

        kv_latent = self.wkv_a(x)
        c_kv, k_pe = torch.split(kv_latent, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        c_kv = self.kv_a_layernorm(c_kv)
        k_pe = k_pe.unsqueeze(2)

        q_pe = apply_rotary_pos_emb(q_pe, cos, sin)
        k_pe = apply_rotary_pos_emb(k_pe, cos, sin)

        kv_decomp = self.wkv_b(c_kv).view(
            bsz, seqlen, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope, v = torch.split(kv_decomp, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        k_nope = self.k_b_layernorm(k_nope)

        q_full = torch.cat([q_nope, q_pe], dim=-1).transpose(1, 2)
        k_pe_exp = k_pe.expand(bsz, seqlen, self.num_heads, self.qk_rope_head_dim)
        k_full = torch.cat([k_nope, k_pe_exp], dim=-1).transpose(1, 2)

        v_trans = v.transpose(1, 2)
        v_pad = F.pad(v_trans, (0, self.qk_head_dim - self.v_head_dim))

        is_causal = (mask is None) and (seqlen > 1)
        attn_out = F.scaled_dot_product_attention(
            q_full,
            k_full,
            v_pad,
            attn_mask=mask,
            dropout_p=0.0,
            is_causal=is_causal,
            scale=self.softmax_scale,
        )

        attn_out = attn_out[..., :self.v_head_dim].transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(attn_out)


class MarkV3SwiGLU(nn.Module):
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.w_gate = nn.Linear(in_features, hidden_features, bias=False)
        self.w_up = nn.Linear(in_features, hidden_features, bias=False)
        self.w_down = nn.Linear(hidden_features, in_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class MarkV3MoE(nn.Module):
    def __init__(self, config: MarkV3Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.n_routed_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok
        self.n_shared_experts = config.n_shared_experts
        self.moe_intermediate_size = config.moe_intermediate_size

        self.shared_experts = MarkV3SwiGLU(
            self.hidden_size, self.moe_intermediate_size * self.n_shared_experts
        )

        self.routed_experts = nn.ModuleList([
            MarkV3SwiGLU(self.hidden_size, self.moe_intermediate_size)
            for _ in range(self.n_routed_experts)
        ])

        self.router = nn.Linear(self.hidden_size, self.n_routed_experts, bias=False)
        self.register_buffer("exp_probs_b", torch.zeros(self.n_routed_experts))

    @torch.no_grad()
    def update_aux_loss_free_bias(self, topk_idx: torch.Tensor, gamma: float = 1e-3):
        counts = torch.bincount(topk_idx.flatten(), minlength=self.n_routed_experts).float()
        avg_count = counts.mean()
        violation = avg_count - counts
        self.exp_probs_b.add_(gamma * torch.sign(violation))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, dim = x.shape
        x_flat = x.view(-1, dim)

        shared_out = self.shared_experts(x_flat)

        router_logits = self.router(x_flat)
        scores = torch.sigmoid(router_logits.float())
        scores_for_choice = scores + self.exp_probs_b.unsqueeze(0)

        _, topk_idx = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)
        topk_weights = torch.gather(scores, dim=-1, index=topk_idx)
        topk_weights = (topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)).to(x.dtype)

        if self.training:
            self.update_aux_loss_free_bias(topk_idx)

        routed_out = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            expert_indices = topk_idx[:, k]
            weights = topk_weights[:, k].unsqueeze(-1)
            for e_idx in range(self.n_routed_experts):
                mask = expert_indices == e_idx
                if mask.any():
                    selected = x_flat[mask]
                    routed_out[mask] += weights[mask] * self.routed_experts[e_idx](selected)

        return (shared_out + routed_out).view(bsz, seqlen, dim)


class MarkV3Block(nn.Module):
    def __init__(self, config: MarkV3Config, layer_idx: int):
        super().__init__()
        self.attn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = MarkV3MLA(config, layer_idx=layer_idx)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.ffn = MarkV3MoE(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = x + self.attn(self.attn_norm(x), cos, sin, mask=mask)
        out = h + self.ffn(self.ffn_norm(h))
        return out


class MarkV3Model(nn.Module):
    def __init__(self, config: MarkV3Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList([
            MarkV3Block(config, layer_idx=i) for i in range(config.num_hidden_layers)
        ])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = MarkV3YaRNRotaryEmbedding(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seqlen = input_ids.shape
        h = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(seqlen, device=input_ids.device)

        causal_mask = None
        if attention_mask is not None:
            causal_mask = torch.ones((bsz, 1, seqlen, seqlen), dtype=torch.bool, device=input_ids.device).tril(0)
            causal_mask = causal_mask & attention_mask[:, None, None, :].bool()

        for layer in self.layers:
            h = layer(h, cos, sin, mask=causal_mask)

        return self.norm(h)


class MarkV3ForCausalLM(nn.Module):
    def __init__(self, config: Optional[MarkV3Config] = None):
        super().__init__()
        if config is None:
            config = MarkV3Config()
        self.config = config
        self.model = MarkV3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        hidden_states = self.model(input_ids, attention_mask=attention_mask)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            return {"loss": loss, "logits": logits}

        return logits
