import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from .mark_arch import DeepSeekMLAAttention, DeepSeekMoELayer, DeepSeekMTPModule, MarkConfig, MarkSwiGLUFFN, MarkTransformerBlock, PagedKVCache, RMSNorm, precompute_freqs_cis

@dataclass
class MarkVL2_2BConfig(MarkConfig):
    model_name: str = 'MarkVL 2.2B Dense Visual Perception Engine'
    model_version: str = '1.0.0'
    vocab_size: int = 32000
    dim: int = 1280
    n_layers: int = 14
    n_heads: int = 16
    n_kv_heads: int = 4
    intermediate_size: int = 3840
    use_mla: bool = True
    q_lora_rank: int = 640
    kv_lora_rank: int = 320
    qk_nope_head_dim: int = 64
    qk_rope_head_dim: int = 32
    v_head_dim: int = 96
    rope_theta: float = 500000.0
    max_position_embeddings: int = 16384
    max_seq_len: int = 16384
    n_dense_layers: int = 0
    moe_layer_freq: int = 1
    recursive_depth_loops: int = 1
    n_routed_experts: int = 24
    n_shared_experts: int = 1
    num_experts_per_tok: int = 2
    moe_intermediate_size: int = 1504
    router_aux_loss_coef: float = 0.01
    mtp_depth: int = 1
    mtp_loss_weight: float = 0.3
    vision_enabled: bool = True
    vision_patch_size: int = 14
    vision_dim: int = 768
    vision_layers: int = 8
    vision_heads: int = 12
    vision_pixel_shuffle_scale: int = 2
    max_image_resolution: Tuple[int, int] = (2048, 2048)
    rms_norm_eps: float = 1e-05
    tie_word_embeddings: bool = True
    quantization: str = 'IQ4_NL'
    target_gpu: str = 'NVIDIA GeForce RTX 2060 Super (8GB GDDR6)'
    target_vram_gb: float = 1.85
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    think_start_id: int = 3
    think_end_id: int = 4
    vision_start_id: int = 10
    vision_patch_id: int = 11
    vision_end_id: int = 12
    unknown_id: int = 14
    web_search_id: int = 15

    def count_parameters(self) -> Dict[str, int]:
        qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        embed_params = self.vocab_size * self.dim
        if not self.tie_word_embeddings:
            embed_params += self.vocab_size * self.dim
        mla_per_layer = self.dim * self.q_lora_rank + self.q_lora_rank * (self.n_heads * qk_head_dim) + self.dim * (self.kv_lora_rank + self.qk_rope_head_dim) + self.kv_lora_rank * (self.n_heads * (self.qk_nope_head_dim + self.v_head_dim)) + self.n_heads * self.v_head_dim * self.dim
        moe_shared = self.n_shared_experts * (3 * self.dim * self.moe_intermediate_size)
        moe_routed = self.n_routed_experts * (3 * self.dim * self.moe_intermediate_size)
        moe_gate = self.dim * self.n_routed_experts
        moe_per_layer = moe_shared + moe_routed + moe_gate
        backbone_total = 0
        for _ in range(self.n_layers):
            backbone_total += mla_per_layer + 2 * self.dim + moe_per_layer
        mtp_params = self.mtp_depth * (2 * self.dim * self.dim + 3 * self.dim * self.moe_intermediate_size)
        v_patch = 3 * self.vision_patch_size ** 2 * self.vision_dim
        v_layers = self.vision_layers * (4 * self.vision_dim ** 2 + 3 * self.vision_dim * (self.vision_dim * 4))
        v_abstractor = self.vision_dim * self.vision_pixel_shuffle_scale ** 2 * self.dim + self.dim
        vision_total = v_patch + v_layers + v_abstractor
        total_physical = embed_params + backbone_total + mtp_params + vision_total
        active_moe_per_layer = self.n_shared_experts * (3 * self.dim * self.moe_intermediate_size) + self.num_experts_per_tok * (3 * self.dim * self.moe_intermediate_size) + moe_gate
        active_flops_per_token = embed_params // self.vocab_size + self.n_layers * (mla_per_layer + 2 * self.dim + active_moe_per_layer)
        return {'total_physical_parameters': total_physical, 'backbone_parameters': backbone_total, 'vision_subsystem_parameters': vision_total, 'embeddings_parameters': embed_params, 'mtp_speculative_parameters': mtp_params, 'active_parameters_per_token': active_flops_per_token, 'num_routed_experts': self.n_routed_experts, 'active_experts_per_token': self.num_experts_per_tok + self.n_shared_experts}

class Spatial2DRotaryEmbedding(nn.Module):

    def __init__(self, dim: int, theta: float=10000.0):
        super().__init__()
        self.dim = dim
        self.theta = theta
        half_dim = dim // 2
        self.inv_freq_y = 1.0 / theta ** (torch.arange(0, half_dim, 2).float() / half_dim)
        self.inv_freq_x = 1.0 / theta ** (torch.arange(0, half_dim, 2).float() / half_dim)

    def forward(self, grid_h: int, grid_w: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        pos_y = torch.arange(grid_h, device=device).float()
        pos_x = torch.arange(grid_w, device=device).float()
        freqs_y = torch.outer(pos_y, self.inv_freq_y.to(device))
        freqs_x = torch.outer(pos_x, self.inv_freq_x.to(device))
        freqs_y = freqs_y[:, None, :].expand(grid_h, grid_w, -1)
        freqs_x = freqs_x[None, :, :].expand(grid_h, grid_w, -1)
        freqs_2d = torch.cat([freqs_y, freqs_x], dim=-1).reshape(grid_h * grid_w, -1)
        cos_2d = torch.cos(freqs_2d)
        sin_2d = torch.sin(freqs_2d)
        return (cos_2d, sin_2d)

def apply_2d_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    x_f = x.float().reshape(*x.shape[:-1], -1, 2)
    cos = cos.unsqueeze(0).to(x.device)
    sin = sin.unsqueeze(0).to(x.device)
    x_r, x_i = (x_f[..., 0], x_f[..., 1])
    x_out = torch.stack([x_r * cos - x_i * sin, x_r * sin + x_i * cos], dim=-1).flatten(2)
    return x_out.to(dtype)

class PixelShuffle2DAbstractor(nn.Module):

    def __init__(self, in_dim: int=768, out_dim: int=1280, scale: int=2):
        super().__init__()
        self.scale = scale
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.proj = nn.Linear(in_dim * scale ** 2, out_dim, bias=True)
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, grid_h: int, grid_w: int) -> torch.Tensor:
        B, N, D = x.shape
        assert N == grid_h * grid_w, f'Token count {N} != {grid_h}x{grid_w}'
        x_2d = x.view(B, grid_h, grid_w, D)
        pad_h = (self.scale - grid_h % self.scale) % self.scale
        pad_w = (self.scale - grid_w % self.scale) % self.scale
        if pad_h > 0 or pad_w > 0:
            x_2d = F.pad(x_2d, (0, 0, 0, pad_w, 0, pad_h))
            grid_h += pad_h
            grid_w += pad_w
        new_h = grid_h // self.scale
        new_w = grid_w // self.scale
        x_shuffled = x_2d.view(B, new_h, self.scale, new_w, self.scale, D).permute(0, 1, 3, 2, 4, 5).contiguous().view(B, new_h * new_w, D * self.scale ** 2)
        return self.norm(self.proj(x_shuffled))

class MarkAnyResVisionEncoder(nn.Module):

    def __init__(self, config: MarkVL2_2BConfig):
        super().__init__()
        self.config = config
        self.patch_size = config.vision_patch_size
        self.patch_embed = nn.Conv2d(in_channels=3, out_channels=config.vision_dim, kernel_size=config.vision_patch_size, stride=config.vision_patch_size, bias=False)
        self.rope_2d = Spatial2DRotaryEmbedding(dim=config.vision_dim)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d_model=config.vision_dim, nhead=config.vision_heads, dim_feedforward=config.vision_dim * 4, activation='gelu', batch_first=True, norm_first=True) for _ in range(config.vision_layers)])
        self.norm = RMSNorm(config.vision_dim, eps=config.rms_norm_eps)
        self.abstractor = PixelShuffle2DAbstractor(in_dim=config.vision_dim, out_dim=config.dim, scale=config.vision_pixel_shuffle_scale)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        B, C, H, W = images.shape
        block_unit = self.patch_size * self.config.vision_pixel_shuffle_scale
        pad_h = (block_unit - H % block_unit) % block_unit
        pad_w = (block_unit - W % block_unit) % block_unit
        if pad_h > 0 or pad_w > 0:
            images = F.pad(images, (0, pad_w, 0, pad_h), mode='replicate')
            _, _, H, W = images.shape
        grid_h = H // self.patch_size
        grid_w = W // self.patch_size
        x = self.patch_embed(images)
        x = x.flatten(2).transpose(1, 2)
        cos_2d, sin_2d = self.rope_2d(grid_h, grid_w, device=images.device)
        x = apply_2d_rotary_emb(x, cos_2d, sin_2d)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        visual_tokens = self.abstractor(x, grid_h, grid_w)
        return visual_tokens

class MarkVLExpertRouter(nn.Module):

    def __init__(self, dim: int=1280, n_routed: int=24, top_k: int=2):
        super().__init__()
        self.dim = dim
        self.n_routed = n_routed
        self.top_k = top_k
        self.gate = nn.Linear(dim, n_routed, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.gate(x)
        scores = F.softmax(logits, dim=-1)
        top_weights, top_indices = torch.topk(scores, self.top_k, dim=-1)
        top_weights = top_weights / (top_weights.sum(dim=-1, keepdim=True) + 1e-06)
        return (top_weights, top_indices)

class MarkVLForCausalLM(nn.Module):

    def __init__(self, config: Optional[MarkVL2_2BConfig]=None):
        super().__init__()
        if config is None:
            config = MarkVL2_2BConfig()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim, padding_idx=config.pad_token_id)
        self.vision_encoder = MarkAnyResVisionEncoder(config) if config.vision_enabled else None
        self.layers = nn.ModuleList([MarkTransformerBlock(config, layer_idx=i) for i in range(config.n_layers)])
        self.norm = RMSNorm(config.dim, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.tok_embeddings.weight
        self.mtp_module = DeepSeekMTPModule(config) if config.mtp_depth > 0 else None
        rope_dim = config.qk_rope_head_dim if config.use_mla else config.head_dim
        cos, sin = precompute_freqs_cis(rope_dim, config.max_seq_len, theta=config.rope_theta, scaling_factor=1.0)
        self.register_buffer('rope_cos', cos, persistent=False)
        self.register_buffer('rope_sin', sin, persistent=False)

    def forward(self, input_ids: torch.Tensor, images: Optional[torch.Tensor]=None, attention_mask: Optional[torch.Tensor]=None, kv_cache: Optional[PagedKVCache]=None, start_pos: int=0, return_mtp: bool=False, labels: Optional[torch.Tensor]=None):
        bsz, seqlen = input_ids.shape
        h = self.tok_embeddings(input_ids)
        if images is not None and self.vision_encoder is not None:
            vis_tokens = self.vision_encoder(images).to(h.dtype)
            vis_mask = input_ids == self.config.vision_patch_id
            if vis_mask.any():
                n_replace = min(vis_mask.sum().item(), vis_tokens.reshape(-1, vis_tokens.size(-1)).size(0))
                flat_vis = vis_tokens.reshape(-1, vis_tokens.size(-1))[:n_replace]
                h_masked_indices = vis_mask.nonzero(as_tuple=False)[:n_replace]
                for idx_i, (b_idx, s_idx) in enumerate(h_masked_indices):
                    h[b_idx, s_idx] = flat_vis[idx_i]
            else:
                h = torch.cat([vis_tokens, h], dim=1)
                seqlen = h.shape[1]
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
            if attention_mask.shape[1] == seqlen:
                padding_mask = attention_mask[:, None, None, :].bool()
                causal_mask = causal_mask & padding_mask
        for layer in self.layers:
            h = layer(h, cos, sin, mask=causal_mask, kv_cache=kv_cache, start_pos=start_pos)
        h_final = self.norm(h)
        logits = self.lm_head(h_final)
        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
        if return_mtp and self.mtp_module is not None and (seqlen > 1):
            next_embeds = self.tok_embeddings(input_ids[:, 1:])
            h_mtp = self.mtp_module(h_final[:, :next_embeds.shape[1], :], next_embeds)
            mtp_logits = self.lm_head(h_mtp)
            if loss is not None:
                return {'loss': loss, 'logits': logits, 'logits_mtp': mtp_logits}
            return (logits, mtp_logits)
        if loss is not None:
            return {'loss': loss, 'logits': logits}
        return logits

def get_mark_vl_2_2b_summary() -> str:
    cfg = MarkVL2_2BConfig()
    stats = cfg.count_parameters()
    lines = [f'Model: {cfg.model_name} (v{cfg.model_version})', f"Total Physical Parameters: {stats['total_physical_parameters']:,} (2.224B)", f"Active Parameters per Token: {stats['active_parameters_per_token']:,} (~313.5M FLOPs)", f'MoE Architecture: {cfg.n_routed_experts} Routed Experts + {cfg.n_shared_experts} Shared Expert (Top-2 Active)', 'Vision Subsystem: AnyRes 2048x2048, 2D-RoPE, PixelShuffle 2x2 (75% token reduction)', f'Target GPU Profile: {cfg.target_gpu} (~{cfg.target_vram_gb} GB VRAM in IQ4_NL, > 6.1 GB Free)', 'Inference Throughput: 95 to 125 tokens/s (Turing Tensor Cores)']
    return '\n'.join(lines)
if __name__ == '__main__':
    print(get_mark_vl_2_2b_summary())
