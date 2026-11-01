---
license: agpl-3.0
library_name: gguf
pipeline_tag: image-text-to-text
language:
  - en
  - pt
datasets:
  - miceworks/Mark-Ultra-Curriculum
tags:
  - gguf
  - deepseek2
  - mixture-of-experts
  - multi-head-latent-attention
  - iq4_nl
  - mark-ai
  - local-ai
  - systems-engineering
---

# Mark-VL-2.2B

## Overview

Mark-VL-2.2B is a vision-language model pairing an AnyRes SigLIP vision encoder with a 14-layer DeepSeek2 MoE language model backbone. It uses 2D Rotary Position Embeddings (2D-RoPE) and a 2x2 PixelShuffle projection adapter to process images alongside text.

The model contains 2,174,310,096 total parameters and 395.50M active parameters per token. The language model binary occupies 1166.72 MiB (1223.40 MB / 1,223,397,152 bytes) in IQ4_NL, operating within 8 GB VRAM limits alongside the mmproj visual adapter.

## Specifications

| Field | Value |
| :--- | :--- |
| Architecture | `deepseek2` |
| Weight State | `TRAINED_CHECKPOINT_VALIDATED` |
| Binary Size | `1166.72 MiB (1223.40 MB / 1,223,397,152 bytes)` |
| Total Parameters | `2,174,310,096` (2174310096 physical) |
| Active Parameters | `395,499,216` (395499216 active) |
| Vocabulary Size | `32000` |
| Layers | `14` |
| Hidden Dimension | `1280` |
| Attention Heads | `16` |
| MLA Ranks | `q_lora_rank=640`, `kv_lora_rank=320`, `qk_rope_head_dim=32` |
| MoE Experts | `24 routed, 1 shared (2 active/token)` |
| Expert FFN Dim | `1504` |
| Context Length | `16384` tokens |
| Tensor Count | `255` |
| Quantization Types | `{'IQ4_NL': 184, 'F32': 71}` |
| SHA-256 | `073d3c00eaadfb98ef7c8b5276e355db134b4b1282ea74d8ce3d6116535cf354` |

## Architecture

### Multi-Head Latent Attention (MLA)

The model projects key and value states into a shared low-rank latent representation to reduce KV cache memory:

```text
c_KV = h_t * W_DKV          (dim: 320)
c_Q  = h_t * W_DQ           (dim: 640)
q_C  = RMSNorm(c_Q * W_UQ)
k_C  = RMSNorm(c_KV * W_UK)
```

Decoupled rotary position embeddings are concatenated along a separate dimension to track token positions:

```text
k_R  = RoPE(h_t * W_KR)     (dim: 32)
q_t  = [q_C ; q_R]
k_t  = [k_C ; k_R]
```

### KV Cache Memory Usage

Physical memory consumption of the KV cache per token:

`KV_bytes_per_token = 14 * (320 + 32) * 2 = 9,856 bytes/token`

| Context Window | MLA KV Cache | Standard MHA KV Cache | Memory Savings |
| :--- | :--- | :--- | :--- |
| 4,096 tokens | 38.50 MiB | 280.00 MiB | 7.3x |
| 8,192 tokens | 77.00 MiB | 560.00 MiB | 7.3x |
| 32,768 tokens | 308.00 MiB | 2240.00 MiB | 7.3x |

### Mixture-of-Experts (MoE)

- **Shared Experts (1S)**: Constantly active path for common features:
  `F_shared(x) = (SiLU(x * W_gate) * (x * W_up)) * W_down`
- **Routed Experts (24R, Top-2)**: Routed using learned bias offsets without auxiliary load balancing loss:
  `s_{t,e} = Softmax(x_t * W_router + b_e)`
  `y = x + F_shared(x) + sum_{i in TopK} g_i * F_routed_i(x)`

### Vision-Language Projector (SigLIP & PixelShuffle)

Mark-VL-2.2B provides a vision-language setup across three files:
1. `mark_vl_2_2b_iq4_nl.gguf`: Language backbone (DeepSeek2 MoE + MLA, 14 layers, 24 routed experts, 1 shared expert).
2. `mmproj_mark_vl_2_2b.gguf`: Vision projector module embedding the AnyRes SigLIP visual feature encoder with 2D-RoPE and 2x2 PixelShuffle projection.
3. `mark_vl_2_2b_q4_k_m.gguf`: Legacy quantization file retained for compatibility.

## Quantization

The model is quantized to GGML_TYPE_IQ4_NL (4-bit non-linear) within the native `deepseek2` GGUF container:
- Linear weights are structured in 18-byte blocks for each group of 32 weights:
  - 2 bytes: Half-precision (`FP16`) scale factor `d`.
  - 16 bytes: 32 four-bit nibbles referencing the non-linear lookup table `kvalues_iq4nl`:
    `[-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]`
- 1D normalization weights (`*_norm.weight`) and router bias vectors (`exp_probs_b.bias`) remain in full precision `FP32` to preserve numerical accuracy at layer boundaries.
- Routed MoE weights are serialized as contiguous 3D tensors (`blk.{i}.ffn_gate_exps.weight`, `blk.{i}.ffn_up_exps.weight`, `blk.{i}.ffn_down_exps.weight`), avoiding tensor splitting and kernel overhead.

## Training Status

Trained on multimodal diagrammatic pairs, technical schematics, and structured documents from miceworks/Mark-Ultra-Curriculum. The repository provides the primary IQ4_NL language container (mark_vl_2_2b_iq4_nl.gguf), the vision adapter (mmproj_mark_vl_2_2b.gguf), and the legacy Q4_K_M artifact.

All reported metrics represent physical parameter counts and deterministic SHA-256 checksums from the GGUF binary.

## Usage

### Download

```bash
# Using huggingface-cli
huggingface-cli download miceworks/Mark-VL-2.2B mark_vl_2_2b_iq4_nl.gguf --local-dir .

# Using Python
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="miceworks/Mark-VL-2.2B", filename="mark_vl_2_2b_iq4_nl.gguf")
```

### Cryptographic Verification

```bash
echo "073d3c00eaadfb98ef7c8b5276e355db134b4b1282ea74d8ce3d6116535cf354  mark_vl_2_2b_iq4_nl.gguf" | sha256sum -c -
```

### Ollama

```dockerfile
# Modelfile
FROM ./mark_vl_2_2b_iq4_nl.gguf
ADAPTER ./mmproj_mark_vl_2_2b.gguf
TEMPLATE """{{ if .System }}<|bos|>System: {{ .System }}<|eos|>
{{ end }}{{ if .Prompt }}<|bos|>User: {{ .Prompt }}<|eos|>
{{ end }}<|bos|>Assistant:
<think>
"""

PARAMETER stop "<|eos|>"
PARAMETER temperature 0.3
PARAMETER top_p 0.95
PARAMETER repeat_penalty 1.1

SYSTEM """You are Mark-VL-2.2B, a systems-focused neural model."""
```

```bash
ollama create mark-vl-2.2b -f Modelfile
ollama run mark-vl-2.2b
```

### llama.cpp

```bash
# CLI generation
llama-cli -m mark_vl_2_2b_iq4_nl.gguf \
  -p "<|bos|>System: Systems mode active.<|eos|>\n<|bos|>User: Explain Linux memory barriers.<|eos|>\n<|bos|>Assistant:\n<think>\n" \
  -n 512 --temp 0.2

# HTTP server
llama-server -m mark_vl_2_2b_iq4_nl.gguf --port 8080 -c 16384
```

### Python Inspection

```python
import gguf

reader = gguf.GGUFReader("mark_vl_2_2b_iq4_nl.gguf")
print("Architecture:", reader.get_field("general.architecture"))
print("Total Tensors:", len(reader.tensors))
total_params = sum(int(t.shape.prod()) for t in reader.tensors)
print(f"Total Parameters: {total_params:,}")
```
