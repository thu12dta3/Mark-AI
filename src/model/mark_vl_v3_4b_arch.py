"""
Mark-VL-V3-4B Multimodal Architecture
High-efficiency 4B vision-language model with Pixel-Shuffle 2D compression,
16-layer ViT visual encoder with 2D-RoPE, and 24-layer H-MLA + PB-MoE language backbone.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mark_v3_arch import (
    RMSNorm,
    MarkV3MLA,
    MarkV3SwiGLU,
    MarkV3MoE,
    MarkV3Block,
    MarkV3Config,
    MarkV3Model,
    MarkV3ForCausalLM,
    get_walsh_hadamard_matrix,
)


@dataclass
class MarkVLV3Config(MarkV3Config):
    model_type: str = "deepseek2"
    model_name: str = "Mark-VL-V3-4B"
    vocab_size: int = 32032
    hidden_size: int = 1792
    num_hidden_layers: int = 24
    num_attention_heads: int = 28
    num_key_value_heads: int = 28
    intermediate_size: int = 4096
    q_lora_rank: int = 512
    kv_lora_rank: int = 256
    qk_nope_head_dim: int = 64
    qk_rope_head_dim: int = 64
    v_head_dim: int = 64
    n_routed_experts: int = 32
    num_experts_per_tok: int = 4
    n_shared_experts: int = 2
    moe_intermediate_size: int = 832  # 26 * 32 = 832 (strictly aligned to IQ4_NL 32-element blocks)
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

    # Vision Subsystem Hyperparameters
    vision_hidden_size: int = 1024
    vision_layers: int = 16
    vision_heads: int = 16
    vision_intermediate_size: int = 2816
    patch_size: int = 14
    image_size: int = 448
    pixel_shuffle_scale: int = 2
    image_token_id: int = 32016
    image_newline_token_id: int = 32017
    alpha_vis_init: float = 1.0

    @property
    def vision_dim(self) -> int:
        return self.vision_hidden_size

    @property
    def vision_patch_size(self) -> int:
        return self.patch_size

    @property
    def vision_pixel_shuffle_scale(self) -> int:
        return self.pixel_shuffle_scale

    def count_parameters(self) -> Dict[str, int]:
        """Calculates precise physical parameter count breakdown."""
        # Embeddings (tied with lm_head)
        embed_params = self.vocab_size * self.hidden_size
        if not self.tie_word_embeddings:
            embed_params += self.vocab_size * self.hidden_size

        # Backbone MLA per layer
        qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        mla_per_layer = (
            self.hidden_size * self.q_lora_rank
            + self.q_lora_rank * (self.num_attention_heads * qk_head_dim)
            + self.hidden_size * (self.kv_lora_rank + self.qk_rope_head_dim)
            + self.kv_lora_rank * (self.num_attention_heads * (self.qk_nope_head_dim + self.v_head_dim))
            + self.num_attention_heads * self.v_head_dim * self.hidden_size
            + self.q_lora_rank  # q_norm
            + self.kv_lora_rank  # kv_norm
        )

        # Backbone MoE per layer
        moe_shared = self.n_shared_experts * (3 * self.hidden_size * self.moe_intermediate_size)
        moe_routed = self.n_routed_experts * (3 * self.hidden_size * self.moe_intermediate_size)
        moe_router = self.hidden_size * self.n_routed_experts
        moe_per_layer = moe_shared + moe_routed + moe_router

        # Layer norms + core loop norm + final norm
        layer_norms = 2 * self.hidden_size
        backbone_total = (
            self.num_hidden_layers * (mla_per_layer + moe_per_layer + layer_norms)
            + self.hidden_size  # final norm
            + self.hidden_size  # core_loop_norm
        )

        # Vision Tower
        v_patch = 3 * (self.patch_size ** 2) * self.vision_hidden_size
        v_attn = 4 * (self.vision_hidden_size ** 2)
        v_mlp = 3 * self.vision_hidden_size * self.vision_intermediate_size
        v_norms = 2 * self.vision_hidden_size
        v_layer = v_attn + v_mlp + v_norms
        v_tower_total = v_patch + self.vision_layers * v_layer + self.vision_hidden_size

        # PixelShuffle Projector (4096 -> 4096 -> 1792 SwiGLU)
        in_feat = self.vision_hidden_size * (self.pixel_shuffle_scale ** 2)
        p_norm_in = in_feat
        p_swiglu = in_feat * in_feat * 2 + in_feat * self.hidden_size
        p_norm_out = self.hidden_size
        p_alpha = 1
        p_newline = self.hidden_size
        projector_total = p_norm_in + p_swiglu + p_norm_out + p_alpha + p_newline

        vision_subsystem_total = v_tower_total + projector_total
        total_physical = embed_params + backbone_total + vision_subsystem_total

        # Active parameters per token
        active_moe_per_layer = (
            self.n_shared_experts * (3 * self.hidden_size * self.moe_intermediate_size)
            + self.num_experts_per_tok * (3 * self.hidden_size * self.moe_intermediate_size)
            + moe_router
        )
        active_backbone = (
            self.hidden_size
            + self.num_hidden_layers * (mla_per_layer + layer_norms + active_moe_per_layer)
        )
        active_with_vision = active_backbone + vision_subsystem_total

        return {
            "total_physical_parameters": total_physical,
            "backbone_parameters": backbone_total + embed_params,
            "vision_subsystem_parameters": vision_subsystem_total,
            "vision_tower_parameters": v_tower_total,
            "projector_parameters": projector_total,
            "embeddings_parameters": embed_params,
            "active_parameters_per_token_backbone": active_backbone,
            "active_parameters_per_token_multimodal": active_with_vision,
            "num_routed_experts": self.n_routed_experts,
            "active_experts_per_token": self.num_experts_per_tok + self.n_shared_experts,
        }


class Spatial2DRoPE(nn.Module):
    """
    2D Spatial Rotary Position Embedding for Vision Transformer.
    Separates attention head dimension into vertical Y (0..31) and horizontal X (32..63).
    """

    def __init__(self, head_dim: int = 64, theta: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta
        dim_per_axis = head_dim // 2  # 32 dims per axis
        freq_dim = dim_per_axis // 2   # 16 frequency pairs
        inv_freq = 1.0 / (theta ** (torch.arange(0, freq_dim, dtype=torch.float32) / float(freq_dim)))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, grid_h: int, grid_w: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        pos_y = torch.arange(grid_h, device=device, dtype=torch.float32)
        pos_x = torch.arange(grid_w, device=device, dtype=torch.float32)

        freqs_y = torch.outer(pos_y, self.inv_freq.to(device))  # [H, 16]
        freqs_x = torch.outer(pos_x, self.inv_freq.to(device))  # [W, 16]

        freqs_y = freqs_y[:, None, :].expand(grid_h, grid_w, -1)  # [H, W, 16]
        freqs_x = freqs_x[None, :, :].expand(grid_h, grid_w, -1)  # [H, W, 16]

        freqs_2d = torch.cat([freqs_y, freqs_x], dim=-1).reshape(grid_h * grid_w, -1)  # [H*W, 32]
        cos_2d = torch.cos(freqs_2d)
        sin_2d = torch.sin(freqs_2d)
        return cos_2d, sin_2d


def apply_2d_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Applies 2D trigonometric rotary rotation on tensor x of shape [B, num_heads, N, head_dim].
    cos, sin: [N, head_dim // 2]
    """
    orig_dtype = x.dtype
    x_f = x.float().reshape(*x.shape[:-1], -1, 2)
    cos = cos.unsqueeze(0).unsqueeze(1).to(x.device)  # [1, 1, N, 32]
    sin = sin.unsqueeze(0).unsqueeze(1).to(x.device)  # [1, 1, N, 32]
    x_r, x_i = x_f[..., 0], x_f[..., 1]
    out = torch.stack([x_r * cos - x_i * sin, x_r * sin + x_i * cos], dim=-1).flatten(-2)
    return out.to(orig_dtype)


class MarkV3VisionAttention(nn.Module):
    """Multi-Head Self-Attention for ViT with Spatial2DRoPE."""

    def __init__(self, hidden_size: int = 1024, num_heads: int = 16):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size, "hidden_size must be divisible by num_heads"

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.scale = self.head_dim ** -0.5

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        q = self.q_proj(x).view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seqlen, self.num_heads, self.head_dim).transpose(1, 2)

        q = apply_2d_rotary(q, cos, sin)
        k = apply_2d_rotary(k, cos, sin)

        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=self.scale
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seqlen, self.hidden_size)
        return self.out_proj(attn_out)


class MarkV3VisionBlock(nn.Module):
    """Transformer block for Vision Tower (LayerNorm/RMSNorm + MHA with 2D-RoPE + MLP SwiGLU)."""

    def __init__(self, config: MarkVLV3Config):
        super().__init__()
        self.norm1 = RMSNorm(config.vision_hidden_size, eps=config.rms_norm_eps)
        self.attn = MarkV3VisionAttention(
            hidden_size=config.vision_hidden_size,
            num_heads=config.vision_heads,
        )
        self.norm2 = RMSNorm(config.vision_hidden_size, eps=config.rms_norm_eps)
        self.mlp = MarkV3SwiGLU(
            in_features=config.vision_hidden_size,
            hidden_features=config.vision_intermediate_size,
        )

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


class MarkV3PixelShuffleProjector(nn.Module):
    """
    PixelShuffle 2x2 Spatial Abstractor with SwiGLU projection and Anti-Outlier Limiter.
    Compresses 1024 patches (32x32) into 256 super-tokens (16x16) and projects from 4096 to 1792.
    """

    def __init__(self, config: MarkVLV3Config):
        super().__init__()
        self.scale = config.pixel_shuffle_scale
        self.in_dim = config.vision_hidden_size
        self.in_features = self.in_dim * (self.scale ** 2)  # 1024 * 4 = 4096
        self.out_dim = config.hidden_size                   # 1792

        self.norm_in = RMSNorm(self.in_features, eps=config.rms_norm_eps)
        self.w_gate = nn.Linear(self.in_features, self.in_features, bias=False)
        self.w_up = nn.Linear(self.in_features, self.in_features, bias=False)
        self.w_down = nn.Linear(self.in_features, self.out_dim, bias=False)
        self.norm_out = RMSNorm(self.out_dim, eps=config.rms_norm_eps)

        # Anti-outlier limiter: learnable scalar initialized to 1.0 (eliminates activation spikes for IQ4_NL)
        self.alpha_vis = nn.Parameter(torch.tensor([config.alpha_vis_init], dtype=torch.float32))

        # 2D line-break anchoring token
        self.image_newline = nn.Parameter(torch.zeros(self.out_dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor, grid_h: int, grid_w: int, insert_newline: bool = False) -> torch.Tensor:
        B, N, D = x.shape
        assert N == grid_h * grid_w, f"Token count {N} != {grid_h}x{grid_w}"

        # Pad spatial dimensions if not divisible by scale
        pad_h = (self.scale - grid_h % self.scale) % self.scale
        pad_w = (self.scale - grid_w % self.scale) % self.scale
        if pad_h > 0 or pad_w > 0:
            x_2d = x.view(B, grid_h, grid_w, D)
            x_2d = F.pad(x_2d, (0, 0, 0, pad_w, 0, pad_h))
            grid_h += pad_h
            grid_w += pad_w
            x = x_2d.view(B, grid_h * grid_w, D)

        new_h = grid_h // self.scale
        new_w = grid_w // self.scale

        # 2x2 spatial neighbor regrouping
        x_2d = x.view(B, grid_h, grid_w, D)
        x_shuffled = (
            x_2d.view(B, new_h, self.scale, new_w, self.scale, D)
            .permute(0, 1, 3, 2, 4, 5)
            .contiguous()
            .view(B, new_h * new_w, D * (self.scale ** 2))
        )

        # SwiGLU projection with norm_in and anti-outlier limiter
        h = self.norm_in(x_shuffled)
        act = F.silu(self.w_gate(h)) * self.w_up(h)
        proj = self.w_down(act)
        out = self.norm_out(proj) * torch.tanh(self.alpha_vis)

        if insert_newline:
            out_2d = out.view(B, new_h, new_w, self.out_dim)
            nl = self.image_newline.view(1, 1, 1, self.out_dim).expand(B, new_h, 1, self.out_dim)
            out = torch.cat([out_2d, nl], dim=2).view(B, new_h * (new_w + 1), self.out_dim)

        return out


class MarkV3VisionTower(nn.Module):
    """
    Vision Tower (Órgão 1):
    16 ViT layers, hidden size 1024, 16 heads, intermediate 2816, input [B, 3, 448, 448].
    """

    def __init__(self, config: MarkVLV3Config):
        super().__init__()
        self.config = config
        self.patch_size = config.patch_size
        self.patch_embed = nn.Conv2d(
            in_channels=3,
            out_channels=config.vision_hidden_size,
            kernel_size=config.patch_size,
            stride=config.patch_size,
            bias=False,
        )
        self.rope_2d = Spatial2DRoPE(
            head_dim=config.vision_hidden_size // config.vision_heads,
            theta=config.rope_theta,
        )
        self.layers = nn.ModuleList([
            MarkV3VisionBlock(config) for _ in range(config.vision_layers)
        ])
        self.norm = RMSNorm(config.vision_hidden_size, eps=config.rms_norm_eps)

    def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, int, int]:
        B, C, H, W = pixel_values.shape
        block_unit = self.patch_size * self.config.pixel_shuffle_scale
        pad_h = (block_unit - H % block_unit) % block_unit
        pad_w = (block_unit - W % block_unit) % block_unit
        if pad_h > 0 or pad_w > 0:
            pixel_values = F.pad(pixel_values, (0, pad_w, 0, pad_h), mode="replicate")
            _, _, H, W = pixel_values.shape

        grid_h = H // self.patch_size
        grid_w = W // self.patch_size

        x = self.patch_embed(pixel_values)  # [B, vision_hidden_size, grid_h, grid_w]
        x = x.flatten(2).transpose(1, 2)    # [B, grid_h * grid_w, vision_hidden_size]

        cos_2d, sin_2d = self.rope_2d(grid_h, grid_w, device=pixel_values.device)

        for layer in self.layers:
            x = layer(x, cos_2d, sin_2d)

        x = self.norm(x)
        return x, grid_h, grid_w


class MarkVLV3ForConditionalGeneration(nn.Module):
    """
    Mark-VL-V3-4B Complete Vision-Language Perceptual Engine:
    - Órgão 1: Vision Tower (16 ViT layers, dim 1024, 2D-RoPE)
    - Órgão 2: Pixel-Shuffle Projector (4096 -> 1792 SwiGLU with tanh limiter)
    - Órgão 3: V3 Backbone (24 layers, dim 1792, 28 heads, H-MLA + 32R/2S PB-MoE, tied vocab 32032)
    """

    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: Optional[MarkVLV3Config] = None):
        super().__init__()
        if config is None:
            config = MarkVLV3Config()
        self.config = config

        self.vision_tower = MarkV3VisionTower(config)
        self.projector = MarkV3PixelShuffleProjector(config)
        self.model = MarkV3Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    def tie_weights(self, *args, **kwargs):
        if getattr(self.config, "tie_word_embeddings", True):
            self.lm_head.weight = self.model.embed_tokens.weight

    @torch.no_grad()
    def fuse_hadamard_weights_(self):
        """Offline Walsh-Hadamard H_256 weight fusion across all 24 backbone layers."""
        self.model.fuse_hadamard_weights_()

    def encode_images(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        insert_newline: bool = False,
        images: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encodes RGB images through the Vision Tower and Pixel-Shuffle Projector."""
        if pixel_values is None and images is not None:
            pixel_values = images
        if pixel_values is None:
            raise ValueError("Must provide either pixel_values or images")
        x, grid_h, grid_w = self.vision_tower(pixel_values)
        return self.projector(x, grid_h, grid_w, insert_newline=insert_newline)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        clan_id: Optional[int] = None,
        core_loops: int = 1,
        inputs_embeds: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        if pixel_values is None and images is not None:
            pixel_values = images

        if inputs_embeds is None:
            if input_ids is None:
                raise ValueError("Must provide either input_ids or inputs_embeds")
            inputs_embeds = self.model.embed_tokens(input_ids)

        if pixel_values is not None:
            visual_tokens = self.encode_images(pixel_values)
            B, T, D = inputs_embeds.shape
            B_v, N_v, _ = visual_tokens.shape
            assert B == B_v, f"Batch size mismatch: text {B} != image {B_v}"

            if input_ids is not None:
                merged_list = []
                for b in range(B):
                    row_ids = input_ids[b]
                    img_idx = (row_ids == self.config.image_token_id).nonzero(as_tuple=True)[0]
                    if len(img_idx) == 0:
                        # Prepend visual tokens if image token not found in prompt
                        merged = torch.cat([visual_tokens[b], inputs_embeds[b]], dim=0)
                    elif len(img_idx) == N_v:
                        # In-place substitution
                        merged = inputs_embeds[b].clone()
                        merged[img_idx] = visual_tokens[b]
                    elif len(img_idx) == 1:
                        # Expand single placeholder token into full visual sequence
                        idx = img_idx[0].item()
                        merged = torch.cat([
                            inputs_embeds[b, :idx],
                            visual_tokens[b],
                            inputs_embeds[b, idx + 1:],
                        ], dim=0)
                    else:
                        idx = img_idx[0].item()
                        merged = torch.cat([
                            inputs_embeds[b, :idx],
                            visual_tokens[b],
                            inputs_embeds[b, idx + len(img_idx):],
                        ], dim=0)
                    merged_list.append(merged)
                inputs_embeds = torch.stack(merged_list, dim=0)
            else:
                inputs_embeds = torch.cat([visual_tokens, inputs_embeds], dim=1)

        hidden_states = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            clan_id=clan_id,
            core_loops=core_loops,
            inputs_embeds=inputs_embeds,
        )
        logits = self.lm_head(hidden_states)

        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
            return {"loss": loss, "logits": logits}

        return logits
