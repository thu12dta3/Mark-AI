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

# Mark-Engineer-302M

## Overview

Mark-Engineer-302M is a compact systems engineering model with 251.45M total parameters and 119.33M active parameters per token. It uses a native DeepSeek2 Mixture-of-Experts (MoE) topology with Multi-Head Latent Attention (MLA), quantized to 4-bit non-linear GGUF (IQ4_NL).

The container occupies 135.12 MiB (141.69 MB / 141,686,432 bytes) of storage and runs on low-power edge devices, Raspberry Pi 5 (8 GB), or entry-level consumer GPUs (such as NVIDIA RTX 2060/3060 8 GB, 256-bit GDDR6, 448 GB/s bandwidth), consuming less than 2% of an 8 GB VRAM allocation.

## Specifications

| Field | Value |
| :--- | :--- |
| Architecture | `deepseek2` |
| Weight State | `TRAINED_CHECKPOINT_VALIDATED` |
| Binary Size | `135.12 MiB (141.69 MB / 141,686,432 bytes)` |
| Total Parameters | `251,452,416` (251452416 physical) |
| Active Parameters | `119,331,840` (119331840 active) |
| Vocabulary Size | `4311` |
| Layers | `8` |
| Hidden Dimension | `1024` |
| Attention Heads | `16` |
| MLA Ranks | `q_lora_rank=512`, `kv_lora_rank=256`, `qk_rope_head_dim=32` |
| MoE Experts | `16 routed, 1 shared (2 active/token)` |
| Expert FFN Dim | `1536` |
| Context Length | `4096` tokens |
| Tensor Count | `115` |
| Quantization Types | `{'IQ4_NL': 82, 'F32': 33}` |
| SHA-256 | `5e9d7d6feb9bc0b6ad5da98e444e3e083a1936e58ad6091230ee674d1ef714b0` |

## Architecture

### Multi-Head Latent Attention (MLA)

The model projects key and value states into a shared low-rank latent representation to reduce KV cache memory:

$$\mathbf{c}_t^{KV} = W^{DKV} \mathbf{h}_t \quad (\mathbf{c}_t^{KV} \in \mathbb{R}^{256})$$

$$\mathbf{c}_t^Q = W^{DQ} \mathbf{h}_t \quad (\mathbf{c}_t^Q \in \mathbb{R}^{512})$$

$$\mathbf{q}_t^C = \text{RMSNorm}(W^{UQ} \mathbf{c}_t^Q), \quad \mathbf{k}_t^C = \text{RMSNorm}(W^{UK} \mathbf{c}_t^{KV})$$

Decoupled rotary position embeddings are concatenated along a separate dimension to track token positions:

$$\mathbf{k}_t^R = \text{RoPE}(W^{KR} \mathbf{h}_t) \quad (\mathbf{k}_t^R \in \mathbb{R}^{32})$$

$$\mathbf{q}_{t, i} = [\mathbf{q}_{t, i}^C; \mathbf{q}_{t, i}^R], \quad \mathbf{k}_{t, i} = [\mathbf{k}_{t, i}^C; \mathbf{k}_t^R]$$

### KV Cache Memory Usage

Physical memory consumption of the KV cache per token:

$$\text{KV}_{\text{bytes/token}} = 8 \times (256 + 32) \times 2 \text{ bytes} = 4,608 \text{ bytes/token}$$

| Context Window | MLA KV Cache | Standard MHA KV Cache | Memory Savings |
| :--- | :--- | :--- | :--- |
| 4,096 tokens | 18.00 MiB | 128.00 MiB | 7.1x |
| 8,192 tokens | 36.00 MiB | 256.00 MiB | 7.1x |
| 32,768 tokens | 144.00 MiB | 1024.00 MiB | 7.1x |

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

Trained on miceworks/Mark-Ultra-Curriculum across 4,821 deduplicated samples covering POSIX C interfaces, Linux system call stubs, memory barrier semantics, and bare-metal linker scripts.

All reported metrics represent physical parameter counts and deterministic SHA-256 checksums from the GGUF binary.

## Usage

### Download

```bash
# Using huggingface-cli
huggingface-cli download miceworks/Mark-Engineer-302M mark_engineer_302m_iq4_nl.gguf --local-dir .

# Using Python
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="miceworks/Mark-Engineer-302M", filename="mark_engineer_302m_iq4_nl.gguf")
```

### Cryptographic Verification

```bash
echo "5e9d7d6feb9bc0b6ad5da98e444e3e083a1936e58ad6091230ee674d1ef714b0  mark_engineer_302m_iq4_nl.gguf" | sha256sum -c -
```

### Ollama

```dockerfile
# Modelfile
FROM ./mark_engineer_302m_iq4_nl.gguf
TEMPLATE """{{ if .System }}<|bos|>System: {{ .System }}<|eos|>
{{ end }}{{ if .Prompt }}<|bos|>User: {{ .Prompt }}<|eos|>
{{ end }}<|bos|>Assistant:
<think>
"""

PARAMETER stop "<|eos|>"
PARAMETER temperature 0.3
PARAMETER top_p 0.95
PARAMETER repeat_penalty 1.1

SYSTEM """You are Mark-Engineer-302M, a systems-focused neural model."""
```

```bash
ollama create mark-engineer-302m -f Modelfile
ollama run mark-engineer-302m
```

### llama.cpp

```bash
# CLI generation
llama-cli -m mark_engineer_302m_iq4_nl.gguf \
  -p "<|bos|>System: Systems mode active.<|eos|>\n<|bos|>User: Explain Linux memory barriers.<|eos|>\n<|bos|>Assistant:\n<think>\n" \
  -n 512 --temp 0.2

# HTTP server
llama-server -m mark_engineer_302m_iq4_nl.gguf --port 8080 -c 4096
```

### Python Inspection

```python
import gguf

reader = gguf.GGUFReader("mark_engineer_302m_iq4_nl.gguf")
print("Architecture:", reader.get_field("general.architecture"))
print("Total Tensors:", len(reader.tensors))
total_params = sum(int(t.shape.prod()) for t in reader.tensors)
print(f"Total Parameters: {total_params:,}")
```
