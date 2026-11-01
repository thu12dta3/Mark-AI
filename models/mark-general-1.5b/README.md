---
license: agpl-3.0
library_name: gguf
pipeline_tag: text-generation
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

# Mark-General-1.5B

## Overview

Mark-General-1.5B is a general reasoning and STEM-oriented sparse Transformer with 1,496,185,824 total parameters and 364.22M active parameters per token across 16 layers. It features an extended 4,377 token vocabulary and a 16-routed, 1-shared MoE structure.

Quantized in IQ4_NL, the container takes 802.97 MiB (841.98 MB / 841,977,664 bytes) and runs locally within standard 8 GB VRAM budgets.

## Specifications

| Field | Value |
| :--- | :--- |
| Architecture | `deepseek2` |
| Weight State | `TRAINED_CHECKPOINT_VALIDATED` |
| Binary Size | `802.97 MiB (841.98 MB / 841,977,664 bytes)` |
| Total Parameters | `1,496,185,824` (1496185824 physical) |
| Active Parameters | `364,215,264` (364215264 active) |
| Vocabulary Size | `4377` |
| Layers | `16` |
| Hidden Dimension | `1280` |
| Attention Heads | `16` |
| MLA Ranks | `q_lora_rank=640`, `kv_lora_rank=320`, `qk_rope_head_dim=32` |
| MoE Experts | `16 routed, 1 shared (2 active/token)` |
| Expert FFN Dim | `1504` |
| Context Length | `4096` tokens |
| Tensor Count | `281` |
| Quantization Types | `{'IQ4_NL': 202, 'F32': 79}` |
| SHA-256 | `4f2be0d7c3445333c3b35f4ff1ec5d4303de3cfbe6d1b6a01a2dd8efa09fab4b` |

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

`KV_bytes_per_token = 16 * (320 + 32) * 2 = 11,264 bytes/token`

| Context Window | MLA KV Cache | Standard MHA KV Cache | Memory Savings |
| :--- | :--- | :--- | :--- |
| 4,096 tokens | 44.00 MiB | 320.00 MiB | 7.3x |
| 8,192 tokens | 88.00 MiB | 640.00 MiB | 7.3x |
| 32,768 tokens | 352.00 MiB | 2560.00 MiB | 7.3x |

### Mixture-of-Experts (MoE)

- **Shared Experts (1S)**: Constantly active path for common features:
  `F_shared(x) = (SiLU(x * W_gate) * (x * W_up)) * W_down`
- **Routed Experts (16R, Top-2)**: Routed using learned bias offsets without auxiliary load balancing loss:
  `s_{t,e} = Softmax(x_t * W_router + b_e)`
  `y = x + F_shared(x) + sum_{i in TopK} g_i * F_routed_i(x)`

## Quantization

The model is quantized to GGML_TYPE_IQ4_NL (4-bit non-linear) within the native `deepseek2` GGUF container:
- Linear weights are structured in 18-byte blocks for each group of 32 weights:
  - 2 bytes: Half-precision (`FP16`) scale factor `d`.
  - 16 bytes: 32 four-bit nibbles referencing the non-linear lookup table `kvalues_iq4nl`:
    `[-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]`
- 1D normalization weights (`*_norm.weight`) and router bias vectors (`exp_probs_b.bias`) remain in full precision `FP32` to preserve numerical accuracy at layer boundaries.
- Routed MoE weights are serialized as contiguous 3D tensors (`blk.{i}.ffn_gate_exps.weight`, `blk.{i}.ffn_up_exps.weight`, `blk.{i}.ffn_down_exps.weight`), avoiding tensor splitting and kernel overhead.

## Training Status

Trained on miceworks/Mark-Ultra-Curriculum with curated STEM data, formal proofs, algorithmic logic, and technical Portuguese and English texts.

All reported metrics represent physical parameter counts and deterministic SHA-256 checksums from the GGUF binary.

## Usage

### Download

```bash
# Using huggingface-cli
huggingface-cli download miceworks/Mark-General-1.5B mark_general_1_5b_iq4_nl.gguf --local-dir .

# Using Python
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="miceworks/Mark-General-1.5B", filename="mark_general_1_5b_iq4_nl.gguf")
```

### Cryptographic Verification

```bash
echo "4f2be0d7c3445333c3b35f4ff1ec5d4303de3cfbe6d1b6a01a2dd8efa09fab4b  mark_general_1_5b_iq4_nl.gguf" | sha256sum -c -
```

### Ollama

```dockerfile
# Modelfile
FROM ./mark_general_1_5b_iq4_nl.gguf
TEMPLATE """{{ if .System }}<|bos|>System: {{ .System }}<|eos|>
{{ end }}{{ if .Prompt }}<|bos|>User: {{ .Prompt }}<|eos|>
{{ end }}<|bos|>Assistant:
<think>
"""

PARAMETER stop "<|eos|>"
PARAMETER temperature 0.3
PARAMETER top_p 0.95
PARAMETER repeat_penalty 1.1

SYSTEM """You are Mark-General-1.5B, a systems-focused neural model."""
```

```bash
ollama create mark-general-1.5b -f Modelfile
ollama run mark-general-1.5b
```

### llama.cpp

```bash
# CLI generation
llama-cli -m mark_general_1_5b_iq4_nl.gguf \
  -p "<|bos|>System: Systems mode active.<|eos|>\n<|bos|>User: Explain Linux memory barriers.<|eos|>\n<|bos|>Assistant:\n<think>\n" \
  -n 512 --temp 0.2

# HTTP server
llama-server -m mark_general_1_5b_iq4_nl.gguf --port 8080 -c 4096
```

### Python Inspection

```python
import gguf

reader = gguf.GGUFReader("mark_general_1_5b_iq4_nl.gguf")
print("Architecture:", reader.get_field("general.architecture"))
print("Total Tensors:", len(reader.tensors))
total_params = sum(int(t.shape.prod()) for t in reader.tensors)
print(f"Total Parameters: {total_params:,}")
```
