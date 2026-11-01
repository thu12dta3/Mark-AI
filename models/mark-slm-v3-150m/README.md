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

# Mark-SLM-V3-150M

## Overview

Mark-SLM-V3-150M is a compact edge model built to stay under the 150M parameter ceiling. By using tied word embeddings (reusing token_embd.weight as the language model output projection), the GGUF file contains 272 tensors and exactly 146,899,248 parameters (54.00M active per token).

Quantized in IQ4_NL, the container occupies 79.97 MiB (83.85 MB / 83,849,984 bytes), making it suitable for low-power edge microcontrollers and embedded devices.

## Specifications

| Field | Value |
| :--- | :--- |
| Architecture | `deepseek2` |
| Weight State | `UN-TRAINED_ARCHITECTURAL_BASE` |
| Binary Size | `79.97 MiB (83.85 MB / 83,849,984 bytes)` |
| Total Parameters | `146,899,248` (146899248 physical) |
| Active Parameters | `54,001,968` (54001968 active) |
| Vocabulary Size | `32032` |
| Layers | `15` |
| Hidden Dimension | `576` |
| Attention Heads | `9` |
| MLA Ranks | `q_lora_rank=256`, `kv_lora_rank=128`, `qk_rope_head_dim=64` |
| MoE Experts | `16 routed, 1 shared (2 active/token)` |
| Expert FFN Dim | `256` |
| Context Length | `8192` tokens |
| Tensor Count | `272` |
| Quantization Types | `{'IQ4_NL': 196, 'F32': 76}` |
| SHA-256 | `6308c8c1529b79e3965131693dbbf46937950b38f88d9791aa6d97a4b87b9a34` |

## Architecture

### Multi-Head Latent Attention (MLA)

The model projects key and value states into a shared low-rank latent representation to reduce KV cache memory:

```text
c_KV = h_t * W_DKV          (dim: 128)
c_Q  = h_t * W_DQ           (dim: 256)
q_C  = RMSNorm(c_Q * W_UQ)
k_C  = RMSNorm(c_KV * W_UK)
```

Decoupled rotary position embeddings are concatenated along a separate dimension to track token positions:

```text
k_R  = RoPE(h_t * W_KR)     (dim: 64)
q_t  = [q_C ; q_R]
k_t  = [k_C ; k_R]
```

### KV Cache Memory Usage

Physical memory consumption of the KV cache per token:

`KV_bytes_per_token = 15 * (128 + 64) * 2 = 5,760 bytes/token`

| Context Window | MLA KV Cache | Standard MHA KV Cache | Memory Savings |
| :--- | :--- | :--- | :--- |
| 4,096 tokens | 22.50 MiB | 135.00 MiB | 6.0x |
| 8,192 tokens | 45.00 MiB | 270.00 MiB | 6.0x |
| 32,768 tokens | 180.00 MiB | 1080.00 MiB | 6.0x |

### Mixture-of-Experts (MoE)

- **Shared Experts (1S)**: Constantly active path for common features:
  `F_shared(x) = (SiLU(x * W_gate) * (x * W_up)) * W_down`
- **Routed Experts (16R, Top-2)**: Routed using learned bias offsets without auxiliary load balancing loss:
  `s_{t,e} = Softmax(x_t * W_router + b_e)`
  `y = x + F_shared(x) + sum_{i in TopK} g_i * F_routed_i(x)`

### V3 Architecture Details

1. **Hadamard-Transformed MLA (H-MLA)**:
   Suppresses latent activation outliers by absorbing the normalized Walsh-Hadamard matrix:
   `H_256 = (1/16) * H_2^(x8)`, `W_fused = H_256 * W_proj`
   directly into key-value projection weights (`W_wkv_a` and `W_wkv_b`) during compilation. This absorption eliminates activation spikes without adding runtime execution overhead.

2. **Partitioned-Balancing Mixture-of-Experts (PB-MoE)**:
   The 16 routed experts are partitioned into 4 functional clans:
   - **Clan 0 (Experts 0-7)**: Systems code, kernel semantics, and concurrency.
   - **Clan 1 (Experts 8-15)**: Mathematics, formal logic, and STEM.
   - **Clan 2 (Experts 16-23)**: Security boundaries, protocols, and validation.
   - **Clan 3 (Experts 24-31)**: Multimodal and spatial reasoning.
   Routing enforces soft-pinning constraints ensuring balanced domain representation per token.

3. **Latent Recursive Depth (LRD) & Tied Embeddings**:
   Enables recurrent intermediate latent steps while enforcing weight-tying between input and output representations to minimize physical disk and VRAM footprints.

## Quantization

The model is quantized to GGML_TYPE_IQ4_NL (4-bit non-linear) within the native `deepseek2` GGUF container:
- Linear weights are structured in 18-byte blocks for each group of 32 weights:
  - 2 bytes: Half-precision (`FP16`) scale factor `d`.
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
huggingface-cli download miceworks/Mark-SLM-V3-150M mark_slm_v3_150m_iq4_nl.gguf --local-dir .

# Using Python
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="miceworks/Mark-SLM-V3-150M", filename="mark_slm_v3_150m_iq4_nl.gguf")
```

### Cryptographic Verification

```bash
echo "6308c8c1529b79e3965131693dbbf46937950b38f88d9791aa6d97a4b87b9a34  mark_slm_v3_150m_iq4_nl.gguf" | sha256sum -c -
```

### Ollama

```dockerfile
# Modelfile
FROM ./mark_slm_v3_150m_iq4_nl.gguf
TEMPLATE """{{ if .System }}<|bos|>System: {{ .System }}<|eos|>
{{ end }}{{ if .Prompt }}<|bos|>User: {{ .Prompt }}<|eos|>
{{ end }}<|bos|>Assistant:
<think>
"""

PARAMETER stop "<|eos|>"
PARAMETER temperature 0.3
PARAMETER top_p 0.95
PARAMETER repeat_penalty 1.1

SYSTEM """You are Mark-SLM-V3-150M, a systems-focused neural model."""
```

```bash
ollama create mark-slm-v3-150m -f Modelfile
ollama run mark-slm-v3-150m
```

### llama.cpp

```bash
# CLI generation
llama-cli -m mark_slm_v3_150m_iq4_nl.gguf \
  -p "<|bos|>System: Systems mode active.<|eos|>\n<|bos|>User: Explain Linux memory barriers.<|eos|>\n<|bos|>Assistant:\n<think>\n" \
  -n 512 --temp 0.2

# HTTP server
llama-server -m mark_slm_v3_150m_iq4_nl.gguf --port 8080 -c 8192
```

### Python Inspection

```python
import gguf

reader = gguf.GGUFReader("mark_slm_v3_150m_iq4_nl.gguf")
print("Architecture:", reader.get_field("general.architecture"))
print("Total Tensors:", len(reader.tensors))
total_params = sum(int(t.shape.prod()) for t in reader.tensors)
print(f"Total Parameters: {total_params:,}")
```
