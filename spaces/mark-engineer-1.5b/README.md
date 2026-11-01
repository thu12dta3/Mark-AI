---
title: Mark Engineer 1.5B
emoji: 💻
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: false
---

# Mark Engineer 1.5B - Interactive Inference Space

Interactive CPU inference testbed for Mark Engineer 1.5B running on native DeepSeek2 architecture (MLA + 3D MoE) in non-linear 4-bit quantization (IQ4_NL).

## Technical Specifications

- **Model Repository**: [miceworks/Mark-Engineer-1.5B](https://huggingface.co/miceworks/Mark-Engineer-1.5B)
- **Architecture**: DeepSeek2 MLA + 3D MoE
- **Quantization**: IQ4_NL (4-bit non-linear)
- **Total Physical Parameters**: 1,496,121,824
- **Active Parameters per Token**: 364,151,264 (24.34% active)
- **Attention**: Multi-Head Latent Attention (q_lora_rank=640, kv_lora_rank=320, v_head_dim=96)
- **MoE Topology**: 16 Layers (4 Dense + 12 MoE), 16 Routed Experts + 1 Shared Expert (Top-2 active per token)
- **Hardware Tier**: cpu-basic (2 vCPU, 16 GB RAM)
