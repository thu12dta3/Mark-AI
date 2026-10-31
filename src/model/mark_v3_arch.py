import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from .mark_arch import RMSNorm

def get_walsh_hadamard_matrix(n: int) -> torch.Tensor:
    h = torch.tensor([[1.0]], dtype=torch.float32)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    return h / math.sqrt(n)

@dataclass
class MarkV3Config:
    model_type: str = 'deepseek2'
    model_name: str = 'Mark-v3-Core-1.1B'
    vocab_size: int = 32032
    hidden_size: int = 1152
    num_hidden_layers: int = 18
    num_attention_heads: int = 18
    num_key_value_heads: int = 18
    intermediate_size: int = 3072
    q_lora_rank: int = 384
    kv_lora_rank: int = 256
    qk_nope_head_dim: int = 64
    qk_rope_head_dim: int = 64
    v_head_dim: int = 64
    n_routed_experts: int = 32
    num_experts_per_tok: int = 4
    n_shared_experts: int = 2
    moe_intermediate_size: int = 512
    aux_loss_alpha: float = 0.001
    max_position_embeddings: int = 32768
    original_max_position_embeddings: int = 4096
    rope_theta: float = 10000.0
    rope_scaling_factor: float = 8.0
    rms_norm_eps: float = 1e-06
    tie_word_embeddings: bool = True
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
        self.mscale = float(0.1 * math.log(self.scale) + 1.0)
        pos_freqs = self.base ** (torch.arange(0, self.dim, 2).float() / self.dim)
        inv_freq = 1.0 / pos_freqs
        low_rot = 1.0
        high_rot = 32.0
        ramp = torch.clamp((self.dim / (2 * math.pi * pos_freqs) - low_rot) / (high_rot - low_rot), 0.0, 1.0)
        yarn_inv_freq = inv_freq * (1.0 - ramp + ramp / self.scale)
        self.register_buffer('inv_freq', yarn_inv_freq, persistent=False)

    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))
        cos = torch.cos(freqs) * self.mscale
        sin = torch.sin(freqs) * self.mscale
        return (cos, sin)

def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    x_f = x.float().reshape(*x.shape[:-1], -1, 2)
    cos = cos.unsqueeze(0).unsqueeze(2).to(x.device)
    sin = sin.unsqueeze(0).unsqueeze(2).to(x.device)
    x_r, x_i = (x_f[..., 0], x_f[..., 1])
    x_out = torch.stack([x_r * cos - x_i * sin, x_r * sin + x_i * cos], dim=-1).flatten(3)
    return x_out.to(dtype)

class MarkV3MLA(nn.Module):

    def __init__(self, config: MarkV3Config, layer_idx: int=0):
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
        self.w_qa = nn.Linear(self.hidden_size, self.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.w_qb = nn.Linear(self.q_lora_rank, self.num_heads * self.qk_head_dim, bias=False)
        self.w_kv_a = nn.Linear(self.hidden_size, self.kv_lora_rank + self.qk_rope_head_dim, bias=False)
        self.kv_norm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.w_kv_b = nn.Linear(self.kv_lora_rank, self.num_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False)
        self.w_o = nn.Linear(self.num_heads * self.v_head_dim, self.hidden_size, bias=False)
        self.softmax_scale = self.qk_head_dim ** (-0.5)

    @torch.no_grad()
    def fuse_hadamard_weights_(self):
        h256 = get_walsh_hadamard_matrix(self.kv_lora_rank).to(device=self.w_kv_b.weight.device, dtype=self.w_kv_b.weight.dtype)
        self.w_kv_b.weight.copy_(self.w_kv_b.weight * self.kv_norm.weight.unsqueeze(0))
        self.kv_norm.weight.fill_(1.0)
        self.w_kv_a.weight[:self.kv_lora_rank, :].copy_(h256 @ self.w_kv_a.weight[:self.kv_lora_rank, :])
        self.w_kv_b.weight.copy_(self.w_kv_b.weight @ h256.T)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mask: Optional[torch.Tensor]=None) -> torch.Tensor:
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
        kv = self.w_kv_b(c_kv).view(bsz, seqlen, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        q_full = torch.cat([q_nope, q_pe], dim=-1).transpose(1, 2)
        k_pe_expanded = k_pe.expand(bsz, seqlen, self.num_heads, self.qk_rope_head_dim)
        k_full = torch.cat([k_nope, k_pe_expanded], dim=-1).transpose(1, 2)
        v_trans = v.transpose(1, 2)
        v_pad = F.pad(v_trans, (0, 64))
        is_causal = mask is None and seqlen > 1
        attn_out = F.scaled_dot_product_attention(q_full, k_full, v_pad, attn_mask=mask, dropout_p=0.0, is_causal=is_causal, scale=self.softmax_scale)
        attn_out = attn_out[..., :64].transpose(1, 2).contiguous().view(bsz, seqlen, -1)
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
    DOMAIN_CLANS: Dict[int, Tuple[int, ...]] = {1: tuple(range(0, 10)), 2: tuple(range(10, 20)), 3: tuple(range(20, 28)), 4: tuple(range(28, 32))}

    def __init__(self, config: MarkV3Config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.n_routed_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok
        self.n_shared_experts = config.n_shared_experts
        self.moe_intermediate_size = config.moe_intermediate_size
        self.shared_experts = MarkV3SwiGLU(self.hidden_size, self.moe_intermediate_size * self.n_shared_experts)
        self.routed_experts = nn.ModuleList([MarkV3SwiGLU(self.hidden_size, self.moe_intermediate_size) for _ in range(self.n_routed_experts)])
        self.router = nn.Linear(self.hidden_size, self.n_routed_experts, bias=False)
        self.register_buffer('exp_probs_b', torch.zeros(self.n_routed_experts))
        for clan_id, indices in self.DOMAIN_CLANS.items():
            other_indices = tuple((i for i in range(self.n_routed_experts) if i not in indices))
            self.register_buffer(f'_clan_{clan_id}_indices', torch.tensor(indices, dtype=torch.long), persistent=False)
            self.register_buffer(f'_clan_{clan_id}_other_indices', torch.tensor(other_indices, dtype=torch.long), persistent=False)

    def forward(self, x: torch.Tensor, clan_id: Optional[int]=None) -> torch.Tensor:
        bsz, seqlen, dim = x.shape
        x_flat = x.view(-1, dim)
        shared_out = self.shared_experts(x_flat)
        router_logits = self.router(x_flat)
        p = torch.sigmoid(router_logits.float())
        s = p + self.exp_probs_b.unsqueeze(0)
        if clan_id in (1, 2, 3, 4):
            clan_idx_buf = getattr(self, f'_clan_{clan_id}_indices')
            other_idx_buf = getattr(self, f'_clan_{clan_id}_other_indices')
            s_clan = s[:, clan_idx_buf]
            _, top3_clan_rel = torch.topk(s_clan, k=3, dim=-1, sorted=False)
            top3_clan = clan_idx_buf[top3_clan_rel]
            s_other = s[:, other_idx_buf]
            _, top1_other_rel = torch.topk(s_other, k=1, dim=-1, sorted=False)
            top1_other = other_idx_buf[top1_other_rel]
            topk_idx = torch.cat([top3_clan, top1_other], dim=-1)
        else:
            _, topk_idx = torch.topk(s, k=self.top_k, dim=-1, sorted=False)
        topk_weights = torch.gather(p, dim=-1, index=topk_idx)
        topk_weights = (topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp(min=1e-06)).to(x.dtype)
        if self.training:
            with torch.no_grad():
                counts = torch.bincount(topk_idx.flatten(), minlength=self.n_routed_experts).float()
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

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mask: Optional[torch.Tensor]=None, clan_id: Optional[int]=None) -> torch.Tensor:
        h = x + self.attn(self.attn_norm(x), cos, sin, mask=mask)
        out = h + self.ffn(self.ffn_norm(h), clan_id=clan_id)
        return out

class MarkV3Model(nn.Module):

    def __init__(self, config: MarkV3Config):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList([MarkV3Block(config, layer_idx=i) for i in range(config.num_hidden_layers)])
        self.core_loop_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = MarkV3YaRNRotaryEmbedding(config)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, clan_id: Optional[int]=None, core_loops: int=1) -> torch.Tensor:
        bsz, seqlen = input_ids.shape
        x = self.embed_tokens(input_ids)
        cos, sin = self.rotary_emb(seqlen, device=input_ids.device)
        causal_mask = None
        if attention_mask is not None:
            causal_mask = torch.ones((bsz, 1, seqlen, seqlen), dtype=torch.bool, device=input_ids.device).tril(0)
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

class MarkV3ForCausalLM(nn.Module):

    def __init__(self, config: Optional[MarkV3Config]=None):
        super().__init__()
        if config is None:
            config = MarkV3Config()
        self.config = config
        self.model = MarkV3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, labels: Optional[torch.Tensor]=None, clan_id: Optional[int]=None, core_loops: int=1) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        hidden_states = self.model(input_ids, attention_mask=attention_mask, clan_id=clan_id, core_loops=core_loops)
        logits = self.lm_head(hidden_states)
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            return {'loss': loss, 'logits': logits}
        return logits
