"""
Hugging Face Transformers Compatible Modeling Implementation for Mark-SLM-V2-302M
Self-contained causal language model architecture with MLA, DeepSeekMoE, and MTP.
"""

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

try:
    from .configuration_mark_slm_v2 import MarkSLMV2Config
except (ImportError, ValueError):
    from configuration_mark_slm_v2 import MarkSLMV2Config



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
    # x: (bsz, seqlen, n_heads, head_dim)
    d = x.shape[-1]
    x1 = x[..., : d // 2]
    x2 = x[..., d // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)
    cos = cos.unsqueeze(0).unsqueeze(2)  # (1, seqlen, 1, d//2)
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
    def __init__(self, config: MarkSLMV2Config, layer_idx: int):
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

        # Query projection with low-rank compression
        self.wq_a = nn.Linear(self.dim, self.q_lora_rank, bias=False)
        self.q_norm = MarkRMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.qk_head_dim, bias=False)

        # Key-Value projection with joint latent compression
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

        # Query processing
        q = self.wq_b(self.q_norm(self.wq_a(x)))
        q = q.view(bsz, seqlen, self.n_heads, self.qk_head_dim)
        q_nope = q[..., : self.qk_nope_head_dim]
        q_pe = q[..., self.qk_nope_head_dim :]
        q_pe = apply_rotary_emb(q_pe, cos[:seqlen], sin[:seqlen])

        # Key-Value processing
        kv_compressed = self.wkv_a(x)
        c_kv = kv_compressed[..., : self.kv_lora_rank]
        k_pe = kv_compressed[..., self.kv_lora_rank :]
        k_pe = k_pe.view(bsz, seqlen, 1, self.qk_rope_head_dim)
        k_pe = apply_rotary_emb(k_pe, cos[:seqlen], sin[:seqlen])

        c_kv_normed = self.kv_norm(c_kv)
        kv = self.wkv_b(c_kv_normed)
        kv = kv.view(bsz, seqlen, self.n_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope = kv[..., : self.qk_nope_head_dim]
        v = kv[..., self.qk_nope_head_dim :]

        k_pe_broadcast = k_pe.expand(-1, -1, self.n_heads, -1)
        q_full = torch.cat([q_nope, q_pe], dim=-1)
        k_full = torch.cat([k_nope, k_pe_broadcast], dim=-1)

        # Transpose for multi-head attention: (bsz, n_heads, seqlen, head_dim)
        q_t = q_full.transpose(1, 2)
        k_t = k_full.transpose(1, 2)
        v_t = v.transpose(1, 2)

        # Causal attention via SDPA
        is_causal = attention_mask is None and seqlen > 1
        output = F.scaled_dot_product_attention(
            q_t, k_t, v_t,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=is_causal,
        )

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, self.n_heads * self.v_head_dim)
        return self.wo(output)


class MarkDeepSeekMoE(nn.Module):
    def __init__(self, config: MarkSLMV2Config):
        super().__init__()
        self.dim = config.hidden_size
        self.n_shared_experts = config.n_shared_experts
        self.n_routed_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok
        self.bias_update_rate = config.router_bias_update_rate

        # 1. Universal Shared Expert
        self.shared_experts = nn.ModuleList([
            MarkSwiGLUFFN(self.dim, config.moe_intermediate_size)
            for _ in range(self.n_shared_experts)
        ])

        # 2. Routed Experts
        self.routed_experts = nn.ModuleList([
            MarkSwiGLUFFN(self.dim, config.moe_intermediate_size)
            for _ in range(self.n_routed_experts)
        ])

        # Router Gate & Dynamic Bias
        self.router_gate = nn.Linear(self.dim, self.n_routed_experts, bias=False)
        self.register_buffer("expert_bias", torch.zeros(self.n_routed_experts))
        self.register_buffer("window_expert_counts", torch.zeros(self.n_routed_experts))
        self.register_buffer("router_step_counter", torch.zeros(1, dtype=torch.long))
        self.rebalance_interval = 100
        self.last_router_z_loss: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, dim = x.shape
        x_flat = x.view(-1, dim)

        # Shared expert execution
        shared_out = sum(expert(x_flat) for expert in self.shared_experts)

        # Router logits
        router_logits = self.router_gate(x_flat)
        self.last_router_z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean()

        biased_logits = router_logits + self.expert_bias.unsqueeze(0)
        topk_scores, topk_idx = torch.topk(biased_logits, self.top_k, dim=-1)
        topk_weights = F.softmax(topk_scores, dim=-1)

        # Sparse dispatch
        routed_out = torch.zeros_like(x_flat)
        for k in range(self.top_k):
            expert_indices = topk_idx[:, k]
            weights = topk_weights[:, k].unsqueeze(-1)
            for e_idx in range(self.n_routed_experts):
                token_mask = expert_indices == e_idx
                if token_mask.any():
                    selected_tokens = x_flat[token_mask]
                    expert_output = self.routed_experts[e_idx](selected_tokens)
                    routed_out[token_mask] += weights[token_mask] * expert_output

        return (shared_out + routed_out).view(bsz, seqlen, dim)


class MarkTransformerBlock(nn.Module):
    def __init__(self, config: MarkSLMV2Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.attn_norm = MarkRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attn = MarkMLAAttention(config, layer_idx)
        self.ffn_norm = MarkRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        if layer_idx < config.n_dense_layers:
            self.ffn = MarkSwiGLUFFN(config.hidden_size, config.intermediate_size)
        else:
            self.ffn = MarkDeepSeekMoE(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = x + self.attn(self.attn_norm(x), cos, sin, attention_mask=attention_mask)
        out = h + self.ffn(self.ffn_norm(h))
        return out


class MarkMTPModule(nn.Module):
    def __init__(self, config: MarkSLMV2Config):
        super().__init__()
        self.h_norm = MarkRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.e_norm = MarkRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.proj = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=False)
        self.ffn = MarkSwiGLUFFN(config.hidden_size, config.moe_intermediate_size)
        self.out_norm = MarkRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, h_prev: torch.Tensor, next_tok_embed: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([self.h_norm(h_prev), self.e_norm(next_tok_embed)], dim=-1)
        h_mtp = self.proj(combined)
        h_mtp = h_mtp + self.ffn(h_mtp)
        return self.out_norm(h_mtp)


class MarkSLMV2PreTrainedModel(PreTrainedModel):
    config_class = MarkSLMV2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class MarkSLMV2Model(MarkSLMV2PreTrainedModel):
    def __init__(self, config: MarkSLMV2Config):
        super().__init__(config)
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList([
            MarkTransformerBlock(config, i) for i in range(config.num_hidden_layers)
        ])
        self.norm = MarkRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        cos, sin = precompute_rope_freqs(
            config.qk_rope_head_dim,
            config.max_position_embeddings,
            theta=config.rope_theta,
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.embed_tokens(input_ids)
        for layer in self.layers:
            h = layer(h, self.rope_cos, self.rope_sin, attention_mask=attention_mask)
        return self.norm(h)


class MarkSLMV2ForCausalLM(MarkSLMV2PreTrainedModel):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: MarkSLMV2Config):
        super().__init__(config)
        self.model = MarkSLMV2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        self.mtp_module = MarkMTPModule(config) if config.mtp_depth > 0 else None
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def load_mark_state_dict(self, state_dict: dict, strict: bool = False):
        mapped_sd = {}
        for k, v in state_dict.items():
            new_k = k
            if new_k.startswith('tok_embeddings.'):
                new_k = new_k.replace('tok_embeddings.', 'model.embed_tokens.')
            elif not new_k.startswith('model.') and not new_k.startswith('lm_head.') and not new_k.startswith('mtp_'):
                new_k = 'model.' + new_k
            new_k = new_k.replace('.ffn.router.', '.ffn.router_gate.')
            new_k = new_k.replace('.ffn.shared_experts.w_', '.ffn.shared_experts.0.w_')
            mapped_sd[new_k] = v
        return self.load_state_dict(mapped_sd, strict=strict)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = True,
        **kwargs,
    ) -> Union[CausalLMOutputWithPast, Tuple[torch.Tensor, ...]]:
        h = self.model(input_ids, attention_mask=attention_mask)
        logits = self.lm_head(h)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

            # MTP loss if present
            if self.mtp_module is not None and input_ids.size(1) > 2:
                next_tok_embeds = self.model.embed_tokens(input_ids[:, 1:])
                h_mtp = self.mtp_module(h[:, :-1], next_tok_embeds)
                mtp_logits = self.lm_head(h_mtp)
                shift_mtp_logits = mtp_logits[:, :-1, :].contiguous()
                shift_mtp_labels = labels[:, 2:].contiguous()
                mtp_loss = loss_fct(shift_mtp_logits.view(-1, self.config.vocab_size), shift_mtp_labels.view(-1))
                loss = loss + self.config.mtp_loss_weight * mtp_loss

        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=None,
            attentions=None,
        )
