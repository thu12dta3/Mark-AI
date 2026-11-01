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

# Mark-SLM-V2-302M

## Overview

Mark-SLM-V2-302M is a compact language model with 329.13M total parameters and 155.72M active parameters per token across 16 layers. It uses a 49,152 token vocabulary, 16 routed experts, and 1 shared expert (2 active routed experts per token).

Quantized in IQ4_NL, the binary occupies 177.76 MiB (186.39 MB / 186,394,944 bytes) and runs on 8 GB consumer GPUs with low latency and minimal VRAM consumption.

## Specifications

| Field | Value |
| :--- | :--- |
| Architecture | `deepseek2` |
| Weight State | `TRAINED_CHECKPOINT_VALIDATED` |
| Binary Size | `177.76 MiB (186.39 MB / 186,394,944 bytes)` |
| Total Parameters | `329,132,000` (329132000 physical) |
| Active Parameters | `155,723,744` (155723744 active) |
| Vocabulary Size | `49152` |
| Layers | `16` |
| Hidden Dimension | `768` |
| Attention Heads | `12` |
| MLA Ranks | `q_lora_rank=384`, `kv_lora_rank=192`, `qk_rope_head_dim=32` |
| MoE Experts | `16 routed, 1 shared (2 active/token)` |
| Expert FFN Dim | `384` |
| Context Length | `4096` tokens |
| Tensor Count | `281` |
| Quantization Types | `{'IQ4_NL': 202, 'F32': 79}` |
| SHA-256 | `34a3e0e39cd02cd6a12345b9558ea094877530ef3bde52e64552f20ca48ce38e` |

## Architecture

### Multi-Head Latent Attention (MLA)

The model projects key and value states into a shared low-rank latent representation to reduce KV cache memory:

```text
c_KV = h_t * W_DKV          (dim: 192)
c_Q  = h_t * W_DQ           (dim: 384)
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

`KV_bytes_per_token = 16 * (192 + 32) * 2 = 7,168 bytes/token`

| Context Window | MLA KV Cache | Standard MHA KV Cache | Memory Savings |
| :--- | :--- | :--- | :--- |
| 4,096 tokens | 28.00 MiB | 192.00 MiB | 6.9x |
| 8,192 tokens | 56.00 MiB | 384.00 MiB | 6.9x |
| 32,768 tokens | 224.00 MiB | 1536.00 MiB | 6.9x |

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

Trained for 10,000 steps on miceworks/Mark-Ultra-Curriculum, achieving a validation loss of 0.263975. The expert dimensions were projected from 400 to 384 via truncated Singular Value Decomposition (SVD), preserving the 384 principal singular components to maintain divisibility by 32 for the IQ4_NL super-blocks while keeping 100% of the trained MLA attention, embedding, and RMSNorm weights intact.

All reported metrics represent physical parameter counts and deterministic SHA-256 checksums from the GGUF binary.

## Usage

### Download

```bash
# Using huggingface-cli
huggingface-cli download miceworks/Mark-SLM-V2-302M mark_slm_v2_302m_iq4_nl.gguf --local-dir .

# Using Python
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="miceworks/Mark-SLM-V2-302M", filename="mark_slm_v2_302m_iq4_nl.gguf")
```

### Cryptographic Verification

```bash
echo "34a3e0e39cd02cd6a12345b9558ea094877530ef3bde52e64552f20ca48ce38e  mark_slm_v2_302m_iq4_nl.gguf" | sha256sum -c -
```

### Ollama

```dockerfile
# Modelfile
FROM ./mark_slm_v2_302m_iq4_nl.gguf
TEMPLATE """{{ if .System }}<|bos|>System: {{ .System }}<|eos|>
{{ end }}{{ if .Prompt }}<|bos|>User: {{ .Prompt }}<|eos|>
{{ end }}<|bos|>Assistant:
<think>
"""

PARAMETER stop "<|eos|>"
PARAMETER temperature 0.3
PARAMETER top_p 0.95
PARAMETER repeat_penalty 1.1

SYSTEM """You are Mark-SLM-V2-302M, a systems-focused neural model."""
```

```bash
ollama create mark-slm-v2-302m -f Modelfile
ollama run mark-slm-v2-302m
```

### llama.cpp

```bash
# CLI generation
llama-cli -m mark_slm_v2_302m_iq4_nl.gguf \
  -p "<|bos|>System: Systems mode active.<|eos|>\n<|bos|>User: Explain Linux memory barriers.<|eos|>\n<|bos|>Assistant:\n<think>\n" \
  -n 512 --temp 0.2

# HTTP server
llama-server -m mark_slm_v2_302m_iq4_nl.gguf --port 8080 -c 4096
```

### Python Inspection

```python
import gguf

reader = gguf.GGUFReader("mark_slm_v2_302m_iq4_nl.gguf")
print("Architecture:", reader.get_field("general.architecture"))
print("Total Tensors:", len(reader.tensors))
total_params = sum(int(t.shape.prod()) for t in reader.tensors)
print(f"Total Parameters: {total_params:,}")
```
