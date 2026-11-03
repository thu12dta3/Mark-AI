---
license: agpl-3.0
library_name: gguf
pipeline_tag: image-text-to-text
language:
  - en
  - pt
tags:
  - gguf
  - deepseek2
  - multimodal
  - vision-language
  - pixel-shuffle
  - mixture-of-experts
  - multi-head-latent-attention
  - iq4_nl
  - mark-ai
  - local-ai
  - systems-engineering
---

# Mark-VL-V3-4B

## Overview

Mark-VL-V3-4B is a third-generation multimodal vision-language model integrating a 16-layer Vision Transformer (ViT) with Spatial 2D-RoPE, a 2x2 PixelShuffle projection adapter with non-linear outlier limiting, and a 24-layer DeepSeek2 H-MLA and PB-MoE language backbone.

The model comprises 4,134,688,769 total physical parameters and 824,483,584 active parameters per token across 24 language layers and 16 vision layers. Quantized in native IQ4_NL, the language backbone and vision projector are partitioned for streaming execution on 8 GB GDDR6 consumer GPUs.

## Specifications

| Field | Value |
| :--- | :--- |
| Architecture | `deepseek2` (Language Backbone) + `clip/mmproj` (Vision Subsystem) |
| Weight State | `UN-TRAINED_ARCHITECTURAL_BASE` |
| Total Parameters | `4,134,688,769` physical (4134688769) |
| Active Parameters | `824,483,584` active / token (backbone) |
| Language Backbone Layers | `24` |
| Language Hidden Dimension | `1792` |
| Language Attention Heads | `28` |
| Multi-Head Latent Attention (MLA) | `q_lora_rank=512`, `kv_lora_rank=256`, `qk_nope_head_dim=64`, `qk_rope_head_dim=64`, `v_head_dim=64` |
| PB-MoE Configuration | `32 routed experts, 2 shared experts (Top-4 active)` |
| MoE Intermediate Size | `832` (strictly aligned to 32-element quantization blocks) |
| Vision Tower Layers | `16` ViT layers |
| Vision Hidden Dimension | `1024` |
| Vision Intermediate Size | `2816` |
| Vision Attention Heads | `16` |
| Vision Patch Size | `14x14` |
| Positional Encoding | `Spatial 2D-RoPE` (32 dim Y, 32 dim X) |
| Projector Subsystem | `2x2 PixelShuffle + SwiGLU (4096 to 4096 to 1792) + RMSNorm + tanh(alpha_vis)` |
| Context Length | `32768` tokens |
| Vocabulary Size | `32032` |
| Quantization Type | `IQ4_NL` (Language Backbone) / `Q8_0` or `FP16` (Vision Adapter) |

## Architecture

Mark-VL-V3-4B operates through three synchronized modules:

### 1. Vision Tower

The visual perception module processes high-resolution images ($448 \times 448 \times 3$) into spatial patch embeddings via a convolution projection:

```text
Patches = Conv2D(in_channels=3, out_channels=1024, kernel_size=14, stride=14, bias=False)
```

This yields a $32 \times 32 = 1024$ patch grid. Each patch is encoded across 16 Vision Transformer layers equipped with RMSNorm, 2D rotary position embeddings (Spatial 2D-RoPE), and SwiGLU feed-forward networks:

```text
Spatial 2D-RoPE:
Y-coords: rotary frequencies along dim 0..31
X-coords: rotary frequencies along dim 32..63
```

### 2. Pixel-Shuffle 2x2 Projector

To compress visual token length while preserving high spatial density, 4 adjacent spatial patches in each $2 \times 2$ neighborhood are folded along the channel axis:

```text
[B, 32, 32, 1024] -> [B, 16, 16, 4096] -> Flatten [B, 256, 4096]
```

The resulting 256 visual tokens are normalized and mapped into the 1792-dimensional language manifold:

```text
h_proj = RMSNorm_out(SwiGLU(RMSNorm_in(x_folded))) * tanh(alpha_vis)
```

The learnable scalar `alpha_vis` (initialized to 1.0) constrains multimodal token magnitudes, suppressing activation outliers prior to non-linear 4-bit quantization. Row-boundary `<image_newline>` tokens are inserted every 16 tokens to preserve spatial 2D grid structure.

### 3. Language Backbone (H-MLA + PB-MoE)

The language backbone features 24 Transformer layers combining:
- **Hadamard-Transformed Multi-Head Latent Attention (H-MLA)**: Projects queries, keys, and values into compressed latent subspaces (`q_lora_rank=512`, `kv_lora_rank=256`), absorbing the normalized Walsh-Hadamard matrix $H_{256}$ into weights during compilation.
- **Partitioned-Balancing Mixture-of-Experts (PB-MoE)**: 32 routed experts partitioned across 4 functional clans (8 experts per clan) alongside 2 isolated shared experts. Dynamic bias updating (`exp_probs_b.bias`) maintains load balance without requiring auxiliary loss gradients.

## Quantization

The model is packaged in native non-linear 4-bit (`IQ4_NL`) GGUF format:
- MoE feed-forward intermediate size is 832, which is an integer multiple of 32 ($832 / 32 = 26$ blocks), ensuring clean alignment without padding or memory bloat.
- All 1D RMSNorm weights and MoE router bias vectors remain in full precision `FP32`.
- MoE expert weights are stored as contiguous 3D tensors (`blk.{i}.ffn_gate_exps.weight`, `blk.{i}.ffn_up_exps.weight`, `blk.{i}.ffn_down_exps.weight`).

## Training Status

Untrained architectural base (`UN-TRAINED_ARCHITECTURAL_BASE`). The architecture is verified for numerical stability, exact tensor shapes, and offline Walsh-Hadamard weight absorption.

## Usage

### Ollama

Create a `Modelfile` containing:

```dockerfile
FROM ./mark_vl_v3_4b_iq4_nl.gguf
ADAPTER ./mmproj-mark_vl_v3_4b.gguf

PARAMETER temperature 0.2
PARAMETER top_p 0.9
PARAMETER stop "<|eos|>"
PARAMETER num_ctx 32768

SYSTEM "You are Mark-VL-V3-4B, a high-efficiency multimodal vision-language system engineered by Miceworks. Analyze images, diagrams, circuits, code screenshots, and scientific documents with exact spatial precision and epistemic honesty."
```

Build and run:

```bash
ollama create mark-vl-v3-4b -f Modelfile
ollama run mark-vl-v3-4b
```

### llama.cpp

```bash
# Vision-language CLI execution
llama-mtmd-cli -m mark_vl_v3_4b_iq4_nl.gguf \
  --mmproj mmproj-mark_vl_v3_4b.gguf \
  --image diagram.png \
  -p "<|bos|>System: Analyze the circuit schematic.<|eos|>\n<|bos|>User: <|vision_start|><|vision_patch|><|vision_end|> Identify component R1.<|eos|>\n<|bos|>Assistant:\n<think>\n"
```
