---
license: agpl-3.0
library_name: gguf
pipeline_tag: text-generation
language:
  - en
  - pt
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

# Mark-Engineer-V2-8B

## Overview

Mark-Engineer-V2-8B is an 8B-class sparse Transformer featuring 32 layers, 24 routed experts, and 1 shared expert with a 49,152 token vocabulary. It has 8,258,809,040 total parameters and activates 1,770,745,040 parameters per token (21.44% active).

In native IQ4_NL quantization, the container occupies 4431.26 MiB (4646.52 MB / 4,646,516,576 bytes), fitting inside a single 8 GB consumer GPU with over 3.3 GB of VRAM remaining for KV cache and scratch space.

## Specifications

| Field | Value |
| :--- | :--- |
| Architecture | `deepseek2` |
| Weight State | `UN-TRAINED_ARCHITECTURAL_BASE` |
| Binary Size | `4431.26 MiB (4646.52 MB / 4,646,516,576 bytes)` |
| Total Parameters | `8,258,809,040` (8258809040 physical) |
| Active Parameters | `1,770,745,040` (1770745040 active) |
| Vocabulary Size | `49152` |
| Layers | `32` |
| Hidden Dimension | `2560` |
| Attention Heads | `32` |
| MLA Ranks | `q_lora_rank=1024`, `kv_lora_rank=512`, `qk_rope_head_dim=32` |
| MoE Experts | `24 routed, 1 shared (2 active/token)` |
| Expert FFN Dim | `1280` |
| Context Length | `8192` tokens |
| Tensor Count | `569` |
| Quantization Types | `{'IQ4_NL': 410, 'F32': 159}` |
| SHA-256 | `28c014b5b903f7e21ee9777ef743030c7a068111c923baeb33d5f05685e05400` |

## Architecture

### Multi-Head Latent Attention (MLA)

The model projects key and value states into a shared low-rank latent representation to reduce KV cache memory:

$$\mathbf{c}_t^{KV} = W^{DKV} \mathbf{h}_t \quad (\mathbf{c}_t^{KV} \in \mathbb{R}^{512})$$

$$\mathbf{c}_t^Q = W^{DQ} \mathbf{h}_t \quad (\mathbf{c}_t^Q \in \mathbb{R}^{1024})$$

$$\mathbf{q}_t^C = \text{RMSNorm}(W^{UQ} \mathbf{c}_t^Q), \quad \mathbf{k}_t^C = \text{RMSNorm}(W^{UK} \mathbf{c}_t^{KV})$$

Decoupled rotary position embeddings are concatenated along a separate dimension to track token positions:

$$\mathbf{k}_t^R = \text{RoPE}(W^{KR} \mathbf{h}_t) \quad (\mathbf{k}_t^R \in \mathbb{R}^{32})$$

$$\mathbf{q}_{t, i} = [\mathbf{q}_{t, i}^C; \mathbf{q}_{t, i}^R], \quad \mathbf{k}_{t, i} = [\mathbf{k}_{t, i}^C; \mathbf{k}_t^R]$$

### KV Cache Memory Usage

Physical memory consumption of the KV cache per token:

$$\text{KV}_{\text{bytes/token}} = 32 \times (512 + 32) \times 2 \text{ bytes} = 34,816 \text{ bytes/token}$$

| Context Window | MLA KV Cache | Standard MHA KV Cache | Memory Savings |
| :--- | :--- | :--- | :--- |
| 4,096 tokens | 136.00 MiB | 1280.00 MiB | 9.4x |
| 8,192 tokens | 272.00 MiB | 2560.00 MiB | 9.4x |
| 32,768 tokens | 1088.00 MiB | 10240.00 MiB | 9.4x |

### Mixture-of-Experts (MoE)

- **Shared Experts (1S)**: Constantly active path for common features:
  $$F_{\text{shared}}(\mathbf{x}) = (\text{SiLU}(\mathbf{x} W_{\text{gate}}^S) \odot (\mathbf{x} W_{\text{up}}^S)) W_{\text{down}}^S$$
- **Routed Experts (24R, Top-2)**: Routed using learned bias offsets without auxiliary load balancing loss:
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

Untrained architectural base (UN-TRAINED_ARCHITECTURAL_BASE). The GGUF file contains initialized weights for testing binary graph compatibility with llama.cpp, measuring memory usage on target hardware, and serving as a base for downstream pre-training and fine-tuning.

All reported metrics represent physical parameter counts and deterministic SHA-256 checksums from the GGUF binary.

## Usage

### Download

```bash
# Using huggingface-cli
huggingface-cli download miceworks/Mark-Engineer-V2-8B mark_engineer_v2_8b_iq4_nl.gguf --local-dir .

# Using Python
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="miceworks/Mark-Engineer-V2-8B", filename="mark_engineer_v2_8b_iq4_nl.gguf")
```

### Cryptographic Verification

```bash
echo "28c014b5b903f7e21ee9777ef743030c7a068111c923baeb33d5f05685e05400  mark_engineer_v2_8b_iq4_nl.gguf" | sha256sum -c -
```

### Ollama

```dockerfile
# Modelfile
FROM ./mark_engineer_v2_8b_iq4_nl.gguf
TEMPLATE """{{ if .System }}<|bos|>System: {{ .System }}<|eos|>
{{ end }}{{ if .Prompt }}<|bos|>User: {{ .Prompt }}<|eos|>
{{ end }}<|bos|>Assistant:
<think>
"""

PARAMETER stop "<|eos|>"
PARAMETER temperature 0.3
PARAMETER top_p 0.95
PARAMETER repeat_penalty 1.1

SYSTEM """You are Mark-Engineer-V2-8B, a systems-focused neural model."""
```

```bash
ollama create mark-engineer-v2-8b -f Modelfile
ollama run mark-engineer-v2-8b
```

### llama.cpp

```bash
# CLI generation
llama-cli -m mark_engineer_v2_8b_iq4_nl.gguf \
  -p "<|bos|>System: Systems mode active.<|eos|>\n<|bos|>User: Explain Linux memory barriers.<|eos|>\n<|bos|>Assistant:\n<think>\n" \
  -n 512 --temp 0.2

# HTTP server
llama-server -m mark_engineer_v2_8b_iq4_nl.gguf --port 8080 -c 8192
```

### Python Inspection

```python
import gguf

reader = gguf.GGUFReader("mark_engineer_v2_8b_iq4_nl.gguf")
print("Architecture:", reader.get_field("general.architecture"))
print("Total Tensors:", len(reader.tensors))
total_params = sum(int(t.shape.prod()) for t in reader.tensors)
print(f"Total Parameters: {total_params:,}")
```
