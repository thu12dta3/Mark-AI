"""
Hugging Face Transformers Compatible Modeling Implementation for Mark-SLM-V3-150M
Sparse Mixture-of-Experts (MoE) Architecture with Multi-Head Latent Attention (MLA).
Total parameters: 146,899,248 (< 150M).
"""

import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel

try:
    from .configuration_mark_v3 import MarkV3Config
except (ImportError, ValueError):
    from configuration_mark_v3 import MarkV3Config


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = torch.mean(x.pow(2), dim=-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight


class MarkV3YaRNRotaryEmbedding(nn.Module):
    def __init__(self, config: MarkV3Config):
        super().__init__()
        self.dim = config.qk_rope_head_dim
        self.max_seq_len = config.max_position_embeddings
        self.base = config.rope_theta
        self.scale = config.rope_scaling_factor

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        inv_freq = inv_freq / self.scale
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        inv_freq = self.inv_freq.to(device)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (bsz, n_heads, seqlen, head_dim)
    seqlen = x.shape[2]
    cos = cos[:seqlen].unsqueeze(0).unsqueeze(1)  # (1, 1, seqlen, dim)
    sin = sin[:seqlen].unsqueeze(0).unsqueeze(1)
    return (x * cos) + (rotate_half(x) * sin)


class MarkV3MLA(nn.Module):
    def __init__(self, config: MarkV3Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.n_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

        # Q LoRA projection
        self.wq_a = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.qk_head_dim, bias=False)

        # KV LoRA projection
        self.wkv_a = nn.Linear(self.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias=False)
        self.kv_norm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.wk_b = nn.Linear(self.kv_lora_rank, self.n_heads * self.qk_nope_head_dim, bias=False)
        self.wv_b = nn.Linear(self.kv_lora_rank, self.n_heads * self.v_head_dim, bias=False)

        self.wo = nn.Linear(self.n_heads * self.v_head_dim, self.hidden_size, bias=False)
        self.softmax_scale = 1.0 / math.sqrt(self.qk_head_dim)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape

        # Query computation
        q_comp = self.wq_a(x)
        q_normed = self.q_norm(q_comp)
        q = self.wq_b(q_normed).view(bsz, seqlen, self.n_heads, self.qk_head_dim).transpose(1, 2)
        q_nope = q[..., : self.qk_nope_head_dim]
        q_rope = q[..., self.qk_nope_head_dim :]
        q_rope = apply_rotary_pos_emb(q_rope, cos, sin)
        q = torch.cat([q_nope, q_rope], dim=-1)

        # Key-Value computation
        kv_comp = self.wkv_a(x)
        c_kv = kv_comp[..., : self.kv_lora_rank]
        k_rope = kv_comp[..., self.kv_lora_rank :]

        c_kv_normed = self.kv_norm(c_kv)
        k_nope = self.wk_b(c_kv_normed).view(bsz, seqlen, self.n_heads, self.qk_nope_head_dim).transpose(1, 2)
        v = self.wv_b(c_kv_normed).view(bsz, seqlen, self.n_heads, self.v_head_dim).transpose(1, 2)

        k_rope = k_rope.view(bsz, seqlen, 1, self.qk_rope_head_dim).transpose(1, 2)
        k_rope = apply_rotary_pos_emb(k_rope, cos, sin)
        k_rope = k_rope.expand(-1, self.n_heads, -1, -1)
        k = torch.cat([k_nope, k_rope], dim=-1)

        # Attention masking
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                causal = torch.triu(
                    torch.full((seqlen, seqlen), float("-inf"), device=x.device, dtype=q.dtype),
                    diagonal=1,
                )
                pad_4d = (1.0 - attention_mask[:, None, None, :].to(dtype=q.dtype, device=x.device)) * -1e9
                attn_mask = causal + pad_4d
                is_causal = False
            elif attention_mask.dim() == 4:
                attn_mask = attention_mask.to(dtype=q.dtype, device=x.device)
                is_causal = False
            else:
                attn_mask = attention_mask.to(device=x.device)
                is_causal = False
        else:
            attn_mask = None
            is_causal = seqlen > 1

        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=is_causal,
            scale=self.softmax_scale,
        )

        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, self.n_heads * self.v_head_dim)
        return self.wo(out)


class MarkV3SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.w_gate = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w_up = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.w_down = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class MarkV3MoE(nn.Module):
    def __init__(self, config: MarkV3Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.n_routed_experts = config.n_routed_experts
        self.n_shared_experts = config.n_shared_experts
        self.top_k = config.num_experts_per_tok
        self.moe_intermediate_size = config.moe_intermediate_size

        self.shared_experts = MarkV3SwiGLU(self.hidden_size, self.moe_intermediate_size * self.n_shared_experts)
        self.routed_experts = nn.ModuleList([
            MarkV3SwiGLU(self.hidden_size, self.moe_intermediate_size)
            for _ in range(self.n_routed_experts)
        ])
        self.router = nn.Linear(self.hidden_size, self.n_routed_experts, bias=False)
        self.exp_probs_b = nn.Parameter(torch.zeros(self.n_routed_experts))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, dim = x.shape
        x_flat = x.view(-1, dim)

        shared_out = self.shared_experts(x_flat)

        router_logits = self.router(x_flat)
        p = torch.sigmoid(router_logits.float())
        s = p + self.exp_probs_b.unsqueeze(0)

        _, topk_idx = torch.topk(s, k=self.top_k, dim=-1, sorted=False)
        topk_weights = torch.gather(p, dim=-1, index=topk_idx)
        topk_weights = (topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)).to(x.dtype)

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
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = x + self.attn(self.attn_norm(x), cos, sin, attention_mask=attention_mask)
        out = h + self.ffn(self.ffn_norm(h))
        return out


class MarkV3PreTrainedModel(PreTrainedModel):
    config_class = MarkV3Config
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


class MarkV3Model(MarkV3PreTrainedModel):
    def __init__(self, config: MarkV3Config):
        super().__init__(config)
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList([MarkV3Block(config, i) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = MarkV3YaRNRotaryEmbedding(config)
        self.post_init()

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seqlen = input_ids.shape
        x = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(seqlen, device=input_ids.device)
        for layer in self.layers:
            x = layer(x, cos, sin, attention_mask=attention_mask)
        return self.norm(x)


class MarkV3ForCausalLM(MarkV3PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: MarkV3Config):
        super().__init__(config)
        self.model = MarkV3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

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
            # Shift internally so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

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
