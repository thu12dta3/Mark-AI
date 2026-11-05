"""
Hugging Face Transformers Compatible Modeling Implementation for Mark-V3-32B
Sparse Mixture-of-Experts (MoE) Architecture with Multi-Head Latent Attention (MLA).
Total parameters: 31,989,009,408 (~3.68B active per token).
Context window: 1,048,576 tokens (1M) via Hybrid 3:1 H-MLA + Log-Scale Attention.
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


def get_walsh_hadamard_matrix(n: int) -> torch.Tensor:
    h = torch.tensor([[1.0]], dtype=torch.float32)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    return h / math.sqrt(n)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_f = x.float()
        var = torch.mean(x_f.pow(2), dim=-1, keepdim=True)
        return (x_f * torch.rsqrt(var + self.eps) * self.weight.float()).to(orig_dtype)


class MarkV3YaRNRotaryEmbedding(nn.Module):
    def __init__(self, config: MarkV3Config):
        super().__init__()
        self.dim = config.qk_rope_head_dim
        self.max_position_embeddings = config.max_position_embeddings
        self.original_max_position_embeddings = config.original_max_position_embeddings
        self.base = config.rope_theta
        self.scale = config.rope_scaling_factor
        self.mscale = float(0.1 * math.log(self.scale) + 1.0)

        pos_freqs = self.base ** (torch.arange(0, self.dim, 2).float() / self.dim)
        inv_freq = 1.0 / pos_freqs
        low_rot = 1.0
        high_rot = 32.0
        ramp = torch.clamp(
            (self.dim / (2 * math.pi * pos_freqs) - low_rot) / (high_rot - low_rot), 0.0, 1.0
        )
        yarn_inv_freq = inv_freq * (1.0 - ramp + ramp / self.scale)
        self.register_buffer("inv_freq", yarn_inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))
        cos = torch.cos(freqs) * self.mscale
        sin = torch.sin(freqs) * self.mscale
        return cos, sin


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    x_f = x.float().reshape(*x.shape[:-1], -1, 2)
    cos = cos.unsqueeze(0).unsqueeze(2).to(x.device)
    sin = sin.unsqueeze(0).unsqueeze(2).to(x.device)
    x_r, x_i = (x_f[..., 0], x_f[..., 1])
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
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim

        # Q LoRA projection
        self.w_qa = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.w_qb = nn.Linear(self.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)

        # KV LoRA projection
        self.w_kv_a = nn.Linear(self.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias=False)
        self.kv_norm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.w_kv_b = nn.Linear(
            self.kv_lora_rank, self.num_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False
        )
        self.w_o = nn.Linear(self.num_heads * self.v_head_dim, self.hidden_size, bias=False)
        self.softmax_scale = self.qk_head_dim ** (-0.5)

    @torch.no_grad()
    def fuse_hadamard_weights_(self):
        h_dim = self.kv_lora_rank
        h_mat = get_walsh_hadamard_matrix(h_dim).to(
            device=self.w_kv_b.weight.device, dtype=self.w_kv_b.weight.dtype
        )
        self.w_kv_b.weight.copy_(self.w_kv_b.weight * self.kv_norm.weight.unsqueeze(0))
        self.kv_norm.weight.fill_(1.0)
        self.w_kv_a.weight[:h_dim, :].copy_(h_mat @ self.w_kv_a.weight[:h_dim, :])
        self.w_kv_b.weight.copy_(self.w_kv_b.weight @ h_mat.T)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        q_latent = self.q_norm(self.w_qa(x))
        q = self.w_qb(q_latent).view(bsz, seqlen, self.num_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        kv_comp = self.w_kv_a(x)
        c_kv, k_pe = torch.split(kv_comp, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        c_kv = self.kv_norm(c_kv)
        k_pe = k_pe.unsqueeze(2)

        q_pe = apply_rotary_pos_emb(q_pe, cos, sin)
        k_pe = apply_rotary_pos_emb(k_pe, cos, sin)

        kv = self.w_kv_b(c_kv).view(
            bsz, seqlen, self.num_heads, self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        q_full = torch.cat([q_nope, q_pe], dim=-1).transpose(1, 2)
        k_pe_expanded = k_pe.expand(bsz, seqlen, self.num_heads, self.qk_rope_head_dim)
        k_full = torch.cat([k_nope, k_pe_expanded], dim=-1).transpose(1, 2)
        v_trans = v.transpose(1, 2)

        pad_len = max(0, self.qk_head_dim - self.v_head_dim)
        v_pad = F.pad(v_trans, (0, pad_len)) if pad_len > 0 else v_trans

        # Long-Context Log-Scale Temperature Scaling on Query post-QKNorm
        if seqlen > 4096:
            log_scale = 1.0 + 0.12 * math.log(max(1.0, seqlen / 4096.0))
            q_full = q_full * log_scale

        # Interleaved 3:1 Local Sliding Window (8192) vs Global Attention
        sliding_window = getattr(self.config, "sliding_window", 8192)
        is_global_layer = (self.layer_idx % 4 == 3)
        attn_mask = mask
        is_causal = (attn_mask is None and seqlen > 1)

        if not is_global_layer and seqlen > sliding_window and attn_mask is None:
            row_idx = torch.arange(seqlen, device=x.device).unsqueeze(1)
            col_idx = torch.arange(seqlen, device=x.device).unsqueeze(0)
            diff = row_idx - col_idx
            valid = (diff >= 0) & (diff <= sliding_window)
            attn_mask = torch.where(valid, 0.0, float("-inf")).to(dtype=x.dtype)
            is_causal = False

        attn_out = F.scaled_dot_product_attention(
            q_full,
            k_full,
            v_pad,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=is_causal,
            scale=self.softmax_scale,
        )
        attn_out = (
            attn_out[..., : self.v_head_dim].transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        )
        return self.w_o(attn_out)


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

        clan_size = max(1, self.n_routed_experts // 4)
        for clan_id in range(1, 5):
            idx_start = (clan_id - 1) * clan_size
            idx_end = clan_size * clan_id if clan_id < 4 else self.n_routed_experts
            clan_indices = tuple(range(idx_start, idx_end))
            other_indices = tuple(i for i in range(self.n_routed_experts) if i not in clan_indices)
            self.register_buffer(
                f"_clan_{clan_id}_indices",
                torch.tensor(clan_indices, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                f"_clan_{clan_id}_other_indices",
                torch.tensor(other_indices, dtype=torch.long),
                persistent=False,
            )

    def forward(self, x: torch.Tensor, clan_id: Optional[int] = None) -> torch.Tensor:
        bsz, seqlen, dim = x.shape
        x_flat = x.view(-1, dim)
        shared_out = self.shared_experts(x_flat)
        router_logits = self.router(x_flat)
        p = torch.sigmoid(router_logits.float())
        s = p + self.exp_probs_b.unsqueeze(0)

        if clan_id in (0, 1, 2, 3, 4):
            effective_clan = (
                (clan_id + 1)
                if clan_id in (0, 1, 2, 3) and not hasattr(self, "_clan_0_indices")
                else clan_id
            )
            clan_idx_buf = getattr(self, f"_clan_{effective_clan}_indices")
            other_idx_buf = getattr(self, f"_clan_{effective_clan}_other_indices")
            s_clan = s[:, clan_idx_buf]
            k_clan = max(1, self.top_k - 1)
            k_other = max(1, self.top_k - k_clan)
            _, top_clan_rel = torch.topk(
                s_clan, k=min(k_clan, s_clan.shape[-1]), dim=-1, sorted=False
            )
            top_clan = clan_idx_buf[top_clan_rel]
            s_other = s[:, other_idx_buf]
            _, top_other_rel = torch.topk(
                s_other, k=min(k_other, s_other.shape[-1]), dim=-1, sorted=False
            )
            top_other = other_idx_buf[top_other_rel]
            topk_idx = torch.cat([top_clan, top_other], dim=-1)
        else:
            _, topk_idx = torch.topk(s, k=self.top_k, dim=-1, sorted=False)

        topk_weights = torch.gather(p, dim=-1, index=topk_idx)
        topk_weights = (topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)).to(
            x.dtype
        )

        if self.training:
            with torch.no_grad():
                counts = torch.bincount(
                    topk_idx.flatten(), minlength=self.n_routed_experts
                ).float()
                avg_count = counts.mean()
                violation = avg_count - counts
                self.exp_probs_b.add_(0.001 * torch.sign(violation))

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
        clan_id: Optional[int] = None,
    ) -> torch.Tensor:
        h = x + self.attn(self.attn_norm(x), cos, sin, mask=mask)
        out = h + self.ffn(self.ffn_norm(h), clan_id=clan_id)
        return out


class MarkV3Model(nn.Module):
    def __init__(self, config: MarkV3Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id
        )
        self.layers = nn.ModuleList([
            MarkV3Block(config, layer_idx=i) for i in range(config.num_hidden_layers)
        ])
        self.core_loop_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = MarkV3YaRNRotaryEmbedding(config)

    @torch.no_grad()
    def fuse_hadamard_weights_(self):
        for layer in self.layers:
            layer.attn.fuse_hadamard_weights_()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        clan_id: Optional[int] = None,
        core_loops: int = 1,
        inputs_embeds: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Must provide either input_ids or inputs_embeds")
            bsz, seqlen = input_ids.shape
            x = self.embed_tokens(input_ids)
            device = input_ids.device
        else:
            bsz, seqlen = inputs_embeds.shape[:2]
            x = inputs_embeds
            device = inputs_embeds.device

        cos, sin = self.rotary_emb(seqlen, device=device)
        causal_mask = None
        if attention_mask is not None:
            causal_mask = torch.ones((bsz, 1, seqlen, seqlen), dtype=torch.bool, device=device).tril(0)
            causal_mask = causal_mask & attention_mask[:, None, None, :].bool()

        for i in range(min(4, len(self.layers))):
            x = self.layers[i](x, cos, sin, mask=causal_mask, clan_id=clan_id)

        z0 = x
        core_end = min(12, len(self.layers))
        if len(self.layers) > 4:
            loops = max(1, core_loops)
            for loop_idx in range(loops):
                if loop_idx > 0:
                    x = self.core_loop_norm(x + 0.5 * z0)
                for i in range(4, core_end):
                    x = self.layers[i](x, cos, sin, mask=causal_mask, clan_id=clan_id)

        for i in range(core_end, len(self.layers)):
            x = self.layers[i](x, cos, sin, mask=causal_mask, clan_id=clan_id)

        return self.norm(x)


class MarkV3ForCausalLM(PreTrainedModel):
    config_class = MarkV3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def __init__(self, config: Optional[MarkV3Config] = None):
        if config is None:
            config = MarkV3Config()
        super().__init__(config)
        self.model = MarkV3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.post_init()

    def _init_weights(self, module):
        std = 0.02
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding):
        self.model.embed_tokens = value
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear):
        self.lm_head = new_embeddings

    @torch.no_grad()
    def fuse_hadamard_weights_(self):
        self.model.fuse_hadamard_weights_()

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        clan_id: Optional[int] = None,
        core_loops: int = 1,
        inputs_embeds: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast, Dict[str, torch.Tensor]]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        hidden_states = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            clan_id=clan_id,
            core_loops=core_loops,
            inputs_embeds=inputs_embeds,
        )
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1)
            )

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
