import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class MarkConfig:
    vocab_size: int = 49152
    dim: int = 2304
    n_layers: int = 28
    n_heads: int = 24
    use_mla: bool = True
    q_lora_rank: int = 768
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 64
    qk_rope_head_dim: int = 32
    v_head_dim: int = 96
    n_kv_heads: int = 8
    kv_cache_quant: str = 'IQ4_NL'
    n_dense_layers: int = 4
    moe_layer_freq: int = 2
    recursive_depth_loops: int = 2
    intermediate_size: int = 6144
    n_shared_experts: int = 1
    n_routed_experts: int = 16
    num_experts_per_tok: int = 2
    moe_intermediate_size: int = 1536
    router_bias_update_rate: float = 0.001
    mtp_depth: int = 1
    mtp_loss_weight: float = 0.3
    vision_enabled: bool = True
    vision_in_channels: int = 3
    vision_patch_size: int = 14
    vision_dim: int = 768
    vision_layers: int = 6
    vision_heads: int = 12
    vision_pixel_shuffle_scale: int = 2
    voice_prosody_enabled: bool = True
    n_prosody_tags: int = 8
    max_seq_len: int = 8192
    rope_theta: float = 500000.0
    rope_scaling_factor: float = 1.0
    rms_norm_eps: float = 1e-05
    dropout: float = 0.0
    tie_word_embeddings: bool = True
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    think_start_id: int = 3
    think_end_id: int = 4
    retrieve_id: int = 5
    retrieve_end_id: int = 6
    evaluate_id: int = 7
    accept_id: int = 8
    reject_id: int = 9
    vision_start_id: int = 10
    vision_patch_id: int = 11
    vision_end_id: int = 12
    voice_stream_id: int = 13
    unknown_id: int = 14
    web_search_id: int = 15
    web_search_end_id: int = 16

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim

    @property
    def mla_cache_dim(self) -> int:
        return self.kv_lora_rank + self.qk_rope_head_dim

    def is_moe_layer(self, layer_idx: int) -> bool:
        if layer_idx < self.n_dense_layers:
            return False
        return (layer_idx - self.n_dense_layers) % self.moe_layer_freq == 0

    def count_parameters(self) -> int:
        total = self.vocab_size * self.dim
        if not self.tie_word_embeddings:
            total += self.vocab_size * self.dim
        mla_per_layer = self.dim * self.q_lora_rank + self.q_lora_rank * (self.n_heads * self.qk_head_dim) + self.dim * (self.kv_lora_rank + self.qk_rope_head_dim) + self.kv_lora_rank * (self.n_heads * (self.qk_nope_head_dim + self.v_head_dim)) + self.n_heads * self.v_head_dim * self.dim
        for i in range(self.n_layers):
            total += mla_per_layer + 2 * self.dim
            if self.is_moe_layer(i):
                shared = self.n_shared_experts * (3 * self.dim * self.moe_intermediate_size)
                routed = self.n_routed_experts * (3 * self.dim * self.moe_intermediate_size)
                gate = self.dim * self.n_routed_experts
                total += shared + routed + gate
            else:
                total += 3 * self.dim * self.intermediate_size
        if self.mtp_depth > 0:
            total += self.mtp_depth * (2 * self.dim * self.dim + 3 * self.dim * self.moe_intermediate_size)
        if self.vision_enabled:
            v_patch = self.vision_in_channels * self.vision_patch_size ** 2 * self.vision_dim
            v_layers = self.vision_layers * (4 * self.vision_dim ** 2 + 3 * self.vision_dim * (self.vision_dim * 4))
            v_proj = self.vision_dim * self.vision_pixel_shuffle_scale ** 2 * self.dim * 2
            total += v_patch + v_layers + v_proj
        return total

    @classmethod
    def create_engineer_1_5b(cls, vocab_size: int=4311) -> 'MarkConfig':
        return cls(vocab_size=vocab_size, dim=1280, n_layers=16, n_heads=16, n_kv_heads=4, use_mla=True, q_lora_rank=640, kv_lora_rank=320, qk_nope_head_dim=64, qk_rope_head_dim=32, v_head_dim=96, n_dense_layers=2, moe_layer_freq=1, recursive_depth_loops=2, intermediate_size=3840, n_shared_experts=1, n_routed_experts=16, num_experts_per_tok=2, moe_intermediate_size=1504, mtp_depth=1, mtp_loss_weight=0.3, vision_enabled=False, voice_prosody_enabled=False, max_seq_len=4096)

    @classmethod
    def create_general_1_5b(cls, vocab_size: int=4377) -> 'MarkConfig':
        return cls(vocab_size=vocab_size, dim=1280, n_layers=16, n_heads=16, n_kv_heads=4, use_mla=True, q_lora_rank=640, kv_lora_rank=320, qk_nope_head_dim=64, qk_rope_head_dim=32, v_head_dim=96, n_dense_layers=2, moe_layer_freq=1, recursive_depth_loops=2, intermediate_size=3840, n_shared_experts=1, n_routed_experts=16, num_experts_per_tok=2, moe_intermediate_size=1504, mtp_depth=1, mtp_loss_weight=0.3, vision_enabled=False, voice_prosody_enabled=False, max_seq_len=4096)

    @classmethod
    def create_engineer_300m(cls, vocab_size: int=4311) -> 'MarkConfig':
        return cls(vocab_size=vocab_size, dim=1024, n_layers=8, n_heads=16, n_kv_heads=4, use_mla=True, kv_lora_rank=256, q_lora_rank=512, qk_rope_head_dim=32, v_head_dim=64, intermediate_size=3072, n_dense_layers=2, moe_layer_freq=1, n_routed_experts=16, n_shared_experts=1, num_experts_per_tok=2, moe_intermediate_size=1536, recursive_depth_loops=2, mtp_depth=1, mtp_loss_weight=0.3, vision_enabled=False, voice_prosody_enabled=False, max_seq_len=4096)

    @classmethod
    def create_primal(cls, vocab_size: int=49152) -> 'MarkConfig':
        return cls(vocab_size=vocab_size, dim=2304, n_layers=28, n_heads=24, use_mla=True, q_lora_rank=768, kv_lora_rank=512, qk_nope_head_dim=64, qk_rope_head_dim=32, v_head_dim=96, n_dense_layers=4, moe_layer_freq=2, recursive_depth_loops=2, intermediate_size=6144, n_shared_experts=1, n_routed_experts=16, num_experts_per_tok=2, moe_intermediate_size=1536, mtp_depth=1, mtp_loss_weight=0.3, vision_enabled=True, voice_prosody_enabled=True, max_seq_len=8192)

    @classmethod
    def create_engineer_v2_8b(cls, vocab_size: int=49152) -> 'MarkConfig':
        return cls(vocab_size=vocab_size, dim=2560, n_layers=32, n_heads=32, n_kv_heads=8, use_mla=True, q_lora_rank=1024, kv_lora_rank=512, qk_nope_head_dim=64, qk_rope_head_dim=32, v_head_dim=80, n_dense_layers=2, moe_layer_freq=1, recursive_depth_loops=2, intermediate_size=7680, n_shared_experts=1, n_routed_experts=24, num_experts_per_tok=2, moe_intermediate_size=1254, router_bias_update_rate=0.001, mtp_depth=1, mtp_loss_weight=0.3, vision_enabled=False, voice_prosody_enabled=False, max_seq_len=8192)
IQ4_NL_LOOKUP_VALUES = [-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224, 0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0]

class RMSNorm(nn.Module):

    def __init__(self, dim: int, eps: float=1e-05):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._norm(x.float()).type_as(x) * self.weight

def precompute_freqs_cis(head_dim: int, max_seq_len: int, theta: float=500000.0, scaling_factor: float=1.0, device: Optional[torch.device]=None) -> Tuple[torch.Tensor, torch.Tensor]:
    if scaling_factor > 1.0:
        theta = theta * (scaling_factor * max_seq_len / max_seq_len - (scaling_factor - 1.0)) ** (head_dim / max(2, head_dim - 2))
    freqs = 1.0 / theta ** (torch.arange(0, head_dim, 2, device=device)[:head_dim // 2].float() / head_dim)
    t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return (torch.cos(freqs), torch.sin(freqs))

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    dtype = xq.dtype
    xq_f = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_f = xk.float().reshape(*xk.shape[:-1], -1, 2)
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    xq_r, xq_i = (xq_f[..., 0], xq_f[..., 1])
    xk_r, xk_i = (xk_f[..., 0], xk_f[..., 1])
    xq_out = torch.stack([xq_r * cos - xq_i * sin, xq_r * sin + xq_i * cos], dim=-1).flatten(3)
    xk_out = torch.stack([xk_r * cos - xk_i * sin, xk_r * sin + xk_i * cos], dim=-1).flatten(3)
    return (xq_out.to(dtype), xk_out.to(dtype))

class PagedKVCache:

    def __init__(self, n_layers: int, max_batch_size: int, max_seq_len: int, n_kv_heads: int, head_dim: int, dtype: torch.dtype=torch.float16, device: torch.device=torch.device('cuda'), use_mla: bool=True, kv_lora_rank: int=512, qk_rope_head_dim: int=32, quant_mode: str='F16'):
        self.max_seq_len = max_seq_len
        self.use_mla = use_mla
        self.kv_lora_rank = kv_lora_rank
        self.qk_rope_head_dim = qk_rope_head_dim
        self.total_latent_dim = kv_lora_rank + qk_rope_head_dim
        self.dtype = dtype
        self.device = device
        self.quant_mode = quant_mode.upper()
        self.iq4_lut = torch.tensor(IQ4_NL_LOOKUP_VALUES, dtype=dtype, device=device)
        if use_mla:
            if self.quant_mode in ('IQ4_NL', 'Q4_0'):
                packed_dim = self.total_latent_dim // 2
                self.latent_cache = torch.zeros((n_layers, max_batch_size, max_seq_len, packed_dim), dtype=torch.uint8, device=device)
                self.latent_scales = torch.ones((n_layers, max_batch_size, max_seq_len, 1), dtype=torch.float16, device=device)
            elif self.quant_mode == 'Q8_0':
                self.latent_cache = torch.zeros((n_layers, max_batch_size, max_seq_len, self.total_latent_dim), dtype=torch.int8, device=device)
                self.latent_scales = torch.ones((n_layers, max_batch_size, max_seq_len, 1), dtype=torch.float16, device=device)
            else:
                self.latent_cache = torch.zeros((n_layers, max_batch_size, max_seq_len, self.total_latent_dim), dtype=dtype, device=device)
                self.latent_scales = None
            self.k_cache = None
            self.v_cache = None
        else:
            self.latent_cache = None
            self.latent_scales = None
            self.k_cache = torch.zeros((n_layers, max_batch_size, n_kv_heads, max_seq_len, head_dim), dtype=dtype, device=device)
            self.v_cache = torch.zeros((n_layers, max_batch_size, n_kv_heads, max_seq_len, head_dim), dtype=dtype, device=device)

    def _quantize_latent(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-06)
        norm_x = (x / scale).clamp(-1.0, 1.0)
        if self.quant_mode == 'Q8_0':
            q = torch.round(norm_x * 127.0).to(torch.int8)
            return (q, scale.to(torch.float16))
        if self.quant_mode == 'Q4_0':
            idx = torch.round((norm_x + 1.0) * 7.5).clamp(0, 15).to(torch.uint8)
        else:
            diffs = (norm_x.unsqueeze(-1) - self.iq4_lut).abs()
            idx = torch.argmin(diffs, dim=-1).to(torch.uint8)
        low = idx[..., 0::2]
        high = idx[..., 1::2]
        packed = high << 4 | low & 15
        return (packed, scale.to(torch.float16))

    def _dequantize_latent(self, q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        scale_d = scale.to(self.dtype)
        if self.quant_mode == 'Q8_0':
            return q.to(self.dtype) / 127.0 * scale_d
        low = (q & 15).long()
        high = (q >> 4 & 15).long()
        idx = torch.empty((*q.shape[:-1], q.shape[-1] * 2), dtype=torch.long, device=q.device)
        idx[..., 0::2] = low
        idx[..., 1::2] = high
        if self.quant_mode == 'Q4_0':
            norm_x = idx.to(self.dtype) / 7.5 - 1.0
        else:
            norm_x = self.iq4_lut[idx]
        return norm_x * scale_d

    def update_mla(self, layer_idx: int, c_kv: torch.Tensor, k_pe: torch.Tensor, start_pos: int) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, seqlen, _ = c_kv.shape
        end_pos = start_pos + seqlen
        combined = torch.cat([c_kv, k_pe.squeeze(2)], dim=-1)
        if end_pos > self.max_seq_len:
            sink = 128
            shift = end_pos - self.max_seq_len
            keep = self.max_seq_len - sink - seqlen
            self.latent_cache[layer_idx, :bsz, sink:sink + keep, :] = self.latent_cache[layer_idx, :bsz, sink + shift:sink + shift + keep, :].clone()
            if self.latent_scales is not None:
                self.latent_scales[layer_idx, :bsz, sink:sink + keep, :] = self.latent_scales[layer_idx, :bsz, sink + shift:sink + shift + keep, :].clone()
            start_pos = self.max_seq_len - seqlen
            end_pos = self.max_seq_len
        if self.latent_scales is not None:
            q_new, s_new = self._quantize_latent(combined)
            self.latent_cache[layer_idx, :bsz, start_pos:end_pos, :] = q_new
            self.latent_scales[layer_idx, :bsz, start_pos:end_pos, :] = s_new
            cached = self._dequantize_latent(self.latent_cache[layer_idx, :bsz, :end_pos, :], self.latent_scales[layer_idx, :bsz, :end_pos, :])
        else:
            self.latent_cache[layer_idx, :bsz, start_pos:end_pos, :] = combined
            cached = self.latent_cache[layer_idx, :bsz, :end_pos, :]
        c_kv_full = cached[..., :self.kv_lora_rank]
        k_pe_full = cached[..., self.kv_lora_rank:].unsqueeze(2)
        return (c_kv_full, k_pe_full)

    def update(self, layer_idx: int, xk: torch.Tensor, xv: torch.Tensor, start_pos: int) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, _, seqlen, _ = xk.shape
        end_pos = start_pos + seqlen
        if end_pos > self.max_seq_len:
            sink = 128
            shift = end_pos - self.max_seq_len
            keep = self.max_seq_len - sink - seqlen
            self.k_cache[layer_idx, :bsz, :, sink:sink + keep, :] = self.k_cache[layer_idx, :bsz, :, sink + shift:sink + shift + keep, :].clone()
            self.v_cache[layer_idx, :bsz, :, sink:sink + keep, :] = self.v_cache[layer_idx, :bsz, :, sink + shift:sink + shift + keep, :].clone()
            start_pos = self.max_seq_len - seqlen
            end_pos = self.max_seq_len
        self.k_cache[layer_idx, :bsz, :, start_pos:end_pos, :] = xk
        self.v_cache[layer_idx, :bsz, :, start_pos:end_pos, :] = xv
        return (self.k_cache[layer_idx, :bsz, :, :end_pos, :], self.v_cache[layer_idx, :bsz, :, :end_pos, :])

    def reset(self):
        if self.latent_cache is not None:
            self.latent_cache.zero_()
        if self.latent_scales is not None:
            self.latent_scales.fill_(1.0)
        if self.k_cache is not None:
            self.k_cache.zero_()
            self.v_cache.zero_()

    def memory_allocated_mb(self) -> float:
        if self.latent_cache is not None:
            cache_bytes = self.latent_cache.numel() * self.latent_cache.element_size()
            scale_bytes = self.latent_scales.numel() * self.latent_scales.element_size() if self.latent_scales is not None else 0
            return (cache_bytes + scale_bytes) / 1024 ** 2
        total_bytes = self.k_cache.numel() * self.k_cache.element_size() * 2
        return total_bytes / 1024 ** 2

class DeepSeekMLAAttention(nn.Module):

    def __init__(self, config: MarkConfig, layer_idx: int=0):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_heads = config.n_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.qk_head_dim = config.qk_head_dim
        self.v_head_dim = config.v_head_dim
        self.dropout = config.dropout
        self.wq_a = nn.Linear(config.dim, config.q_lora_rank, bias=False)
        self.q_norm = RMSNorm(config.q_lora_rank, eps=config.rms_norm_eps)
        self.wq_b = nn.Linear(config.q_lora_rank, config.n_heads * self.qk_head_dim, bias=False)
        self.wkv_a = nn.Linear(config.dim, config.kv_lora_rank + config.qk_rope_head_dim, bias=False)
        self.kv_norm = RMSNorm(config.kv_lora_rank, eps=config.rms_norm_eps)
        self.wkv_b = nn.Linear(config.kv_lora_rank, config.n_heads * (config.qk_nope_head_dim + config.v_head_dim), bias=False)
        self.wo = nn.Linear(config.n_heads * config.v_head_dim, config.dim, bias=False)
        self.softmax_scale = self.qk_head_dim ** (-0.5)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mask: Optional[torch.Tensor]=None, kv_cache: Optional[PagedKVCache]=None, start_pos: int=0) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        q = self.wq_b(self.q_norm(self.wq_a(x)))
        q = q.view(bsz, seqlen, self.n_heads, self.qk_head_dim)
        q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        kv_latent = self.wkv_a(x)
        c_kv, k_pe = torch.split(kv_latent, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        c_kv = self.kv_norm(c_kv)
        k_pe = k_pe.unsqueeze(2)
        q_pe, k_pe = apply_rotary_emb(q_pe, k_pe, cos, sin)
        if kv_cache is not None and kv_cache.use_mla:
            c_kv, k_pe = kv_cache.update_mla(self.layer_idx, c_kv, k_pe, start_pos)
        total_seqlen = c_kv.shape[1]
        kv = self.wkv_b(c_kv).view(bsz, total_seqlen, self.n_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        q_full = torch.cat([q_nope, q_pe], dim=-1).transpose(1, 2)
        k_pe_expanded = k_pe.expand(bsz, total_seqlen, self.n_heads, self.qk_rope_head_dim)
        k_full = torch.cat([k_nope, k_pe_expanded], dim=-1).transpose(1, 2)
        v_full = v.transpose(1, 2)
        is_causal = mask is None and (kv_cache is None or seqlen > 1)
        if q_full.is_cuda and q_full.dtype in (torch.float16, torch.bfloat16):
            try:
                from torch.nn.attention import SDPBackend, sdpa_kernel
                with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                    output = F.scaled_dot_product_attention(q_full, k_full, v_full, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0, is_causal=is_causal, scale=self.softmax_scale)
            except Exception:
                output = F.scaled_dot_product_attention(q_full, k_full, v_full, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0, is_causal=is_causal, scale=self.softmax_scale)
        else:
            output = F.scaled_dot_product_attention(q_full, k_full, v_full, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0, is_causal=is_causal, scale=self.softmax_scale)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)
MarkSelfAttention = DeepSeekMLAAttention

class MarkSwiGLUFFN(nn.Module):

    def __init__(self, dim: int, intermediate_size: int):
        super().__init__()
        self.w_gate = nn.Linear(dim, intermediate_size, bias=False)
        self.w_up = nn.Linear(dim, intermediate_size, bias=False)
        self.w_down = nn.Linear(intermediate_size, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))

class DeepSeekMoELayer(nn.Module):

    def __init__(self, config: MarkConfig):
        super().__init__()
        self.dim = config.dim
        self.n_routed_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok
        self.bias_update_rate = config.router_bias_update_rate
        self.rebalance_interval = 100
        self.shared_experts = MarkSwiGLUFFN(config.dim, config.moe_intermediate_size * config.n_shared_experts)
        self.routed_experts = nn.ModuleList([MarkSwiGLUFFN(config.dim, config.moe_intermediate_size) for _ in range(config.n_routed_experts)])
        self.router = nn.Linear(config.dim, config.n_routed_experts, bias=False)
        self.register_buffer('expert_bias', torch.zeros(config.n_routed_experts))
        self.register_buffer('window_expert_counts', torch.zeros(config.n_routed_experts))
        self.register_buffer('router_step_counter', torch.zeros(1, dtype=torch.long))
        self.last_router_z_loss: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, dim = x.shape
        x_flat = x.view(-1, dim)
        shared_out = self.shared_experts(x_flat)
        router_logits = self.router(x_flat)
        if self.training:
            z_loss = torch.logsumexp(router_logits.float(), dim=-1).pow(2).mean()
            self.last_router_z_loss = 0.0001 * z_loss
        else:
            self.last_router_z_loss = None
        scores = torch.sigmoid(router_logits.float())
        scores_for_choice = scores + self.expert_bias.unsqueeze(0)
        _, topk_idx = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)
        topk_weights = torch.gather(scores, dim=-1, index=topk_idx)
        topk_weights = (topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp(min=1e-06)).to(x.dtype)
        if self.training:
            with torch.no_grad():
                counts = torch.bincount(topk_idx.flatten(), minlength=self.n_routed_experts).float()
                avg_count = counts.mean()
                load_violation = avg_count - counts
                self.expert_bias.add_(self.bias_update_rate * torch.sign(load_violation))
                self.window_expert_counts.add_(counts)
                self.router_step_counter.add_(1)
                if int(self.router_step_counter.item()) % self.rebalance_interval == 0:
                    total_calls = self.window_expert_counts.sum().clamp(min=1.0)
                    freq = self.window_expert_counts / total_calls
                    target_freq = 1.0 / float(self.n_routed_experts)
                    imbalance = target_freq - freq
                    self.expert_bias.add_(5.0 * self.bias_update_rate * (imbalance / target_freq).clamp(-3.0, 3.0))
                    self.expert_bias.sub_(self.expert_bias.mean())
                    self.expert_bias.clamp_(-0.35, 0.35)
                    self.window_expert_counts.zero_()
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

class MarkVisionEncoder(nn.Module):

    def __init__(self, config: MarkConfig):
        super().__init__()
        self.patch_size = config.vision_patch_size
        self.vision_dim = config.vision_dim
        self.scale = config.vision_pixel_shuffle_scale
        self.patch_embed = nn.Conv2d(config.vision_in_channels, config.vision_dim, kernel_size=config.vision_patch_size, stride=config.vision_patch_size, bias=False)
        self.blocks = nn.ModuleList([nn.TransformerEncoderLayer(d_model=config.vision_dim, nhead=config.vision_heads, dim_feedforward=config.vision_dim * 4, dropout=0.0, activation='gelu', batch_first=True, norm_first=True) for _ in range(config.vision_layers)])
        self.norm = RMSNorm(config.vision_dim)
        merged_dim = config.vision_dim * self.scale ** 2
        self.projector = nn.Sequential(nn.Linear(merged_dim, config.dim, bias=False), nn.SiLU(), nn.Linear(config.dim, config.dim, bias=False))

    def pixel_shuffle_merge(self, x: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
        bsz, _, d = x.shape
        x = x.view(bsz, grid_h, grid_w, d)
        pad_h = (self.scale - grid_h % self.scale) % self.scale
        pad_w = (self.scale - grid_w % self.scale) % self.scale
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
            grid_h += pad_h
            grid_w += pad_w
        x = x.view(bsz, grid_h // self.scale, self.scale, grid_w // self.scale, self.scale, d)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        return x.view(bsz, grid_h // self.scale * (grid_w // self.scale), d * self.scale ** 2)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        patches = self.patch_embed(pixel_values)
        grid_h, grid_w = (patches.shape[2], patches.shape[3])
        x = patches.flatten(2).transpose(1, 2)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        x_merged = self.pixel_shuffle_merge(x, grid_h, grid_w)
        return self.projector(x_merged)

class DeepSeekMTPModule(nn.Module):

    def __init__(self, config: MarkConfig):
        super().__init__()
        self.h_norm = RMSNorm(config.dim, eps=config.rms_norm_eps)
        self.e_norm = RMSNorm(config.dim, eps=config.rms_norm_eps)
        self.proj = nn.Linear(2 * config.dim, config.dim, bias=False)
        self.ffn = MarkSwiGLUFFN(config.dim, config.moe_intermediate_size)
        self.out_norm = RMSNorm(config.dim, eps=config.rms_norm_eps)

    def forward(self, h_prev: torch.Tensor, next_tok_embed: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([self.h_norm(h_prev), self.e_norm(next_tok_embed)], dim=-1)
        h_mtp = self.proj(combined)
        h_mtp = h_mtp + self.ffn(h_mtp)
        return self.out_norm(h_mtp)

class MarkTransformerBlock(nn.Module):

    def __init__(self, config: MarkConfig, layer_idx: int):
        super().__init__()
        self.attn_norm = RMSNorm(config.dim, eps=config.rms_norm_eps)
        self.attn = DeepSeekMLAAttention(config, layer_idx=layer_idx)
        self.ffn_norm = RMSNorm(config.dim, eps=config.rms_norm_eps)
        if config.is_moe_layer(layer_idx):
            self.ffn = DeepSeekMoELayer(config)
        else:
            self.ffn = MarkSwiGLUFFN(config.dim, config.intermediate_size)
        self.recursive_loops = max(1, getattr(config, 'recursive_depth_loops', 1))
        if self.recursive_loops > 1:
            rank = 64
            self.refine_norm = RMSNorm(config.dim, eps=config.rms_norm_eps)
            self.refine_down = nn.Linear(config.dim, rank, bias=False)
            self.refine_up = nn.Linear(rank, config.dim, bias=False)
            self.refine_post_norm = RMSNorm(config.dim, eps=config.rms_norm_eps)
            self.refine_scale = nn.Parameter(torch.zeros(1))
        else:
            self.refine_norm = None

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, mask: Optional[torch.Tensor]=None, kv_cache: Optional[PagedKVCache]=None, start_pos: int=0) -> torch.Tensor:
        h = x + self.attn(self.attn_norm(x), cos, sin, mask=mask, kv_cache=kv_cache, start_pos=start_pos)
        out = h + self.ffn(self.ffn_norm(h))
        if self.refine_norm is not None:
            r_in = self.refine_norm(out)
            delta = self.refine_up(F.silu(self.refine_down(r_in)))
            refined = self.refine_post_norm(self.ffn(r_in + delta))
            out = out + torch.tanh(self.refine_scale) * refined
        return out

class MarkModel(nn.Module):

    def __init__(self, config: MarkConfig):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList([MarkTransformerBlock(config, layer_idx=i) for i in range(config.n_layers)])
        self.norm = RMSNorm(config.dim, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.tok_embeddings.weight
        self.mtp_module = DeepSeekMTPModule(config) if config.mtp_depth > 0 else None
        self.vision_encoder = MarkVisionEncoder(config) if config.vision_enabled else None
        self.prosody_head = nn.Linear(config.dim, config.n_prosody_tags, bias=False) if config.voice_prosody_enabled else None
        rope_dim = config.qk_rope_head_dim if config.use_mla else config.head_dim
        cos, sin = precompute_freqs_cis(rope_dim, config.max_seq_len * int(max(1.0, config.rope_scaling_factor)), theta=config.rope_theta, scaling_factor=config.rope_scaling_factor)
        self.register_buffer('rope_cos', cos, persistent=False)
        self.register_buffer('rope_sin', sin, persistent=False)
        self.gradient_checkpointing = False
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        std = 0.02
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        for name, p in module.named_parameters():
            if name.endswith('wo.weight') or name.endswith('w_down.weight'):
                nn.init.normal_(p, mean=0.0, std=std / math.sqrt(2 * self.config.n_layers))
            elif name.endswith('refine_up.weight') or name.endswith('refine_scale'):
                nn.init.zeros_(p)

    def get_router_z_loss(self) -> torch.Tensor:
        z_losses = [layer.ffn.last_router_z_loss for layer in self.layers if isinstance(layer.ffn, DeepSeekMoELayer) and layer.ffn.last_router_z_loss is not None]
        if not z_losses:
            return torch.tensor(0.0, device=self.tok_embeddings.weight.device)
        return torch.stack(z_losses).sum()

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, kv_cache: Optional[PagedKVCache]=None, start_pos: int=0, pixel_values: Optional[torch.Tensor]=None, return_mtp: bool=False):
        bsz, seqlen = input_ids.shape
        h = self.tok_embeddings(input_ids)
        if pixel_values is not None and self.vision_encoder is not None:
            vis_embeds = self.vision_encoder(pixel_values).to(h.dtype)
            vis_mask = input_ids == self.config.vision_patch_id
            if vis_mask.any():
                n_replace = min(vis_mask.sum().item(), vis_embeds.reshape(-1, vis_embeds.size(-1)).size(0))
                flat_vis = vis_embeds.reshape(-1, vis_embeds.size(-1))[:n_replace]
                h_masked_indices = vis_mask.nonzero(as_tuple=False)[:n_replace]
                for idx_i, (b_idx, s_idx) in enumerate(h_masked_indices):
                    h[b_idx, s_idx] = flat_vis[idx_i]
        rope_dim = self.config.qk_rope_head_dim if self.config.use_mla else self.config.head_dim
        if start_pos + seqlen > self.rope_cos.shape[0]:
            new_max = start_pos + seqlen + 1024
            cos, sin = precompute_freqs_cis(rope_dim, new_max, theta=self.config.rope_theta, scaling_factor=max(1.0, new_max / self.config.max_seq_len), device=input_ids.device)
            self.rope_cos = cos
            self.rope_sin = sin
        cos = self.rope_cos[start_pos:start_pos + seqlen].to(h.device)
        sin = self.rope_sin[start_pos:start_pos + seqlen].to(h.device)
        causal_mask = None
        if attention_mask is not None and kv_cache is None:
            causal_mask = torch.ones((bsz, 1, seqlen, seqlen), dtype=torch.bool, device=input_ids.device).tril(diagonal=0)
            padding_mask = attention_mask[:, None, None, :].bool()
            causal_mask = causal_mask & padding_mask
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                h = torch.utils.checkpoint.checkpoint(layer, h, cos, sin, causal_mask, None, 0, use_reentrant=False)
            else:
                h = layer(h, cos, sin, mask=causal_mask, kv_cache=kv_cache, start_pos=start_pos)
        h_final = self.norm(h)
        logits = self.lm_head(h_final)
        if return_mtp and self.mtp_module is not None and (seqlen > 1):
            next_embeds = self.tok_embeddings(input_ids[:, 1:])
            h_mtp = self.mtp_module(h_final[:, :-1, :], next_embeds)
            mtp_logits = self.lm_head(h_mtp)
            return (logits, mtp_logits)
        return logits

    @torch.inference_mode()
    def predict_next_two_tokens_speculative(self, cur_ids: torch.Tensor, kv_cache: PagedKVCache, start_pos: int) -> Tuple[int, Optional[int]]:
        bsz, seqlen = cur_ids.shape
        h = self.tok_embeddings(cur_ids)
        cos = self.rope_cos[start_pos:start_pos + seqlen].to(h.device)
        sin = self.rope_sin[start_pos:start_pos + seqlen].to(h.device)
        for layer in self.layers:
            h = layer(h, cos, sin, mask=None, kv_cache=kv_cache, start_pos=start_pos)
        h_last = self.norm(h[:, -1:, :])
        logits_1 = self.lm_head(h_last[:, -1, :])
        tok_1 = int(torch.argmax(logits_1, dim=-1).item())
        tok_2 = None
        if self.mtp_module is not None:
            e_next = self.tok_embeddings(torch.tensor([[tok_1]], device=h.device))
            h_mtp = self.mtp_module(h_last, e_next)
            logits_2 = self.lm_head(h_mtp[:, -1, :])
            tok_2 = int(torch.argmax(logits_2, dim=-1).item())
        return (tok_1, tok_2)
MarkForCausalLM = MarkModel
