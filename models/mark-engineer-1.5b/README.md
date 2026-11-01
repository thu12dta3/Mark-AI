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

# Mark-Engineer-1.5B

## Overview

Mark-Engineer-1.5B is a 1.5B-parameter sparse Transformer exported to the native deepseek2 GGUF format. It combines Multi-Head Latent Attention (MLA) with a 16-routed and 1-shared SwiGLU Mixture-of-Experts (MoE) topology to keep active parameters at 364.15M per token (24.34% active) out of 1,496,121,824 total parameters.

Quantized in IQ4_NL, the binary occupies 802.94 MiB (841.94 MB / 841,941,664 bytes), leaving over 7 GB of VRAM free on an 8 GB GPU (tested on NVIDIA RTX 2060 Super 8 GB) for large context windows and fast local decoding.

## Specifications

| Field | Value |
| :--- | :--- |
| Architecture | `deepseek2` |
| Weight State | `TRAINED_CHECKPOINT_VALIDATED` |
| Binary Size | `802.94 MiB (841.94 MB / 841,941,664 bytes)` |
| Total Parameters | `1,496,121,824` (1496121824 physical) |
| Active Parameters | `364,151,264` (364151264 active) |
| Vocabulary Size | `4311` |
| Layers | `16` |
| Hidden Dimension | `1280` |
| Attention Heads | `16` |
| MLA Ranks | `q_lora_rank=640`, `kv_lora_rank=320`, `qk_rope_head_dim=32` |
| MoE Experts | `16 routed, 1 shared (2 active/token)` |
| Expert FFN Dim | `1504` |
| Context Length | `4096` tokens |
| Tensor Count | `281` |
| Quantization Types | `{'IQ4_NL': 202, 'F32': 79}` |
| SHA-256 | `fa7d3061f33a20beaae5c2b85aceaf720c75b811efd8bceae0c57c57037e838e` |

## Architecture

### Multi-Head Latent Attention (MLA)

The model projects key and value states into a shared low-rank latent representation to reduce KV cache memory:

$$\mathbf{c}_t^{KV} = W^{DKV} \mathbf{h}_t \quad (\mathbf{c}_t^{KV} \in \mathbb{R}^{320})$$

$$\mathbf{c}_t^Q = W^{DQ} \mathbf{h}_t \quad (\mathbf{c}_t^Q \in \mathbb{R}^{640})$$

$$\mathbf{q}_t^C = \text{RMSNorm}(W^{UQ} \mathbf{c}_t^Q), \quad \mathbf{k}_t^C = \text{RMSNorm}(W^{UK} \mathbf{c}_t^{KV})$$

Decoupled rotary position embeddings are concatenated along a separate dimension to track token positions:

$$\mathbf{k}_t^R = \text{RoPE}(W^{KR} \mathbf{h}_t) \quad (\mathbf{k}_t^R \in \mathbb{R}^{32})$$

$$\mathbf{q}_{t, i} = [\mathbf{q}_{t, i}^C; \mathbf{q}_{t, i}^R], \quad \mathbf{k}_{t, i} = [\mathbf{k}_{t, i}^C; \mathbf{k}_t^R]$$

### KV Cache Memory Usage

Physical memory consumption of the KV cache per token:

$$\text{KV}_{\text{bytes/token}} = 16 \times (320 + 32) \times 2 \text{ bytes} = 11,264 \text{ bytes/token}$$

| Context Window | MLA KV Cache | Standard MHA KV Cache | Memory Savings |
| :--- | :--- | :--- | :--- |
| 4,096 tokens | 44.00 MiB | 320.00 MiB | 7.3x |
| 8,192 tokens | 88.00 MiB | 640.00 MiB | 7.3x |
| 32,768 tokens | 352.00 MiB | 2560.00 MiB | 7.3x |

### Mixture-of-Experts (MoE)

- **Shared Experts (1S)**: Constantly active path for common features:
  $$F_{\text{shared}}(\mathbf{x}) = (\text{SiLU}(\mathbf{x} W_{\text{gate}}^S) \odot (\mathbf{x} W_{\text{up}}^S)) W_{\text{down}}^S$$
- **Routed Experts (16R, Top-2)**: Routed using learned bias offsets without auxiliary load balancing loss:
  $$s_{t, e} = \text{Softmax}(\mathbf{x}_t W_{\text{router}} + \mathbf{b}_e)$$

## Quantization

The model is quantized to GGML_TYPE_IQ4_NL (4-bit non-linear) within the native `deepseek2` GGUF container:
- Linear weights are structured in 18-byte blocks for each group of 32 weights:
  - 2 bytes: Half-precision (`FP16`) scale factor $d$.
  - 16 bytes: 32 four-bit nibbles referencing the non-linear lookup table `kvalues_iq4nl`:
    `[-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]`
- 1D normalization weights (`*_norm.weight`) and router bias vectors (`exp_probs_b.bias`) remain in full precision `FP32` to preserve numerical accuracy at layer boundaries.
- Routed MoE weights are serialized as contiguous 3D tensors (`blk.{i}.ffn_gate_exps.weight`, `blk.{i}.ffn_up_exps.weight`, `blk.{i}.ffn_down_exps.weight`), avoiding tensor splitting and kernel overhead.

## Training Status

Trained on miceworks/Mark-Ultra-Curriculum using low-level kernel sources, systems manuals, RFCs, and concurrency benchmarks. Verified checkpoint (TRAINED_CHECKPOINT_VALIDATED) for code synthesis and systems programming.

All reported metrics represent physical parameter counts and deterministic SHA-256 checksums from the GGUF binary.

## Usage

### Download

```bash
# Using huggingface-cli
huggingface-cli download miceworks/Mark-Engineer-1.5B mark_engineer_1_5b_iq4_nl.gguf --local-dir .

# Using Python
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="miceworks/Mark-Engineer-1.5B", filename="mark_engineer_1_5b_iq4_nl.gguf")
```

### Cryptographic Verification

```bash
echo "fa7d3061f33a20beaae5c2b85aceaf720c75b811efd8bceae0c57c57037e838e  mark_engineer_1_5b_iq4_nl.gguf" | sha256sum -c -
```

### Ollama

```dockerfile
# Modelfile
FROM ./mark_engineer_1_5b_iq4_nl.gguf
TEMPLATE """{{ if .System }}<|bos|>System: {{ .System }}<|eos|>
{{ end }}{{ if .Prompt }}<|bos|>User: {{ .Prompt }}<|eos|>
{{ end }}<|bos|>Assistant:
<think>
"""

PARAMETER stop "<|eos|>"
PARAMETER temperature 0.3
PARAMETER top_p 0.95
PARAMETER repeat_penalty 1.1

SYSTEM """You are Mark-Engineer-1.5B, a systems-focused neural model."""
```

```bash
ollama create mark-engineer-1.5b -f Modelfile
ollama run mark-engineer-1.5b
```

### llama.cpp

```bash
# CLI generation
llama-cli -m mark_engineer_1_5b_iq4_nl.gguf \
  -p "<|bos|>System: Systems mode active.<|eos|>\n<|bos|>User: Explain Linux memory barriers.<|eos|>\n<|bos|>Assistant:\n<think>\n" \
  -n 512 --temp 0.2

# HTTP server
llama-server -m mark_engineer_1_5b_iq4_nl.gguf --port 8080 -c 4096
```

### Python Inspection

```python
import gguf

reader = gguf.GGUFReader("mark_engineer_1_5b_iq4_nl.gguf")
print("Architecture:", reader.get_field("general.architecture"))
print("Total Tensors:", len(reader.tensors))
total_params = sum(int(t.shape.prod()) for t in reader.tensors)
print(f"Total Parameters: {total_params:,}")
```
