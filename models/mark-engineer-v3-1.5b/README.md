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

# Mark-Engineer-V3-1.5B

## Overview

Mark-Engineer-V3-1.5B is a systems engineering variant with 1,520,496,960 total physical parameters and 405.73M active parameters per token. It uses 32 routed experts, 2 shared experts, and domain soft-pinning configured for code generation, compilers, and kernel internals.

Quantized in IQ4_NL, the container occupies 816.03 MiB (855.67 MB / 855,669,248 bytes), leaving ample VRAM on 8 GB GPUs for large source code files.

## Specifications

| Field | Value |
| :--- | :--- |
| Architecture | `deepseek2` |
| Weight State | `UN-TRAINED_ARCHITECTURAL_BASE` |
| Binary Size | `816.03 MiB (855.67 MB / 855,669,248 bytes)` |
| Total Parameters | `1,520,496,960` (1520496960 physical) |
| Active Parameters | `405,729,600` (405729600 active) |
| Vocabulary Size | `32032` |
| Layers | `18` |
| Hidden Dimension | `1280` |
| Attention Heads | `20` |
| MLA Ranks | `q_lora_rank=512`, `kv_lora_rank=256`, `qk_rope_head_dim=64` |
| MoE Experts | `32 routed, 2 shared (4 active/token)` |
| Expert FFN Dim | `576` |
| Context Length | `32768` tokens |
| Tensor Count | `327` |
| Quantization Types | `{'IQ4_NL': 236, 'F32': 91}` |
| SHA-256 | `62b8bd003adcd0780a967faca7163f98e79334809fae2effea5e7f55d4c9bfe4` |

## Architecture

### Multi-Head Latent Attention (MLA)

The model projects key and value states into a shared low-rank latent representation to reduce KV cache memory:

$$\mathbf{c}_t^{KV} = W^{DKV} \mathbf{h}_t \quad (\mathbf{c}_t^{KV} \in \mathbb{R}^{256})$$

$$\mathbf{c}_t^Q = W^{DQ} \mathbf{h}_t \quad (\mathbf{c}_t^Q \in \mathbb{R}^{512})$$

$$\mathbf{q}_t^C = \text{RMSNorm}(W^{UQ} \mathbf{c}_t^Q), \quad \mathbf{k}_t^C = \text{RMSNorm}(W^{UK} \mathbf{c}_t^{KV})$$

Decoupled rotary position embeddings are concatenated along a separate dimension to track token positions:

$$\mathbf{k}_t^R = \text{RoPE}(W^{KR} \mathbf{h}_t) \quad (\mathbf{k}_t^R \in \mathbb{R}^{64})$$

$$\mathbf{q}_{t, i} = [\mathbf{q}_{t, i}^C; \mathbf{q}_{t, i}^R], \quad \mathbf{k}_{t, i} = [\mathbf{k}_{t, i}^C; \mathbf{k}_t^R]$$

### KV Cache Memory Usage

Physical memory consumption of the KV cache per token:

$$\text{KV}_{\text{bytes/token}} = 18 \times (256 + 64) \times 2 \text{ bytes} = 11,520 \text{ bytes/token}$$

| Context Window | MLA KV Cache | Standard MHA KV Cache | Memory Savings |
| :--- | :--- | :--- | :--- |
| 4,096 tokens | 45.00 MiB | 360.00 MiB | 8.0x |
| 8,192 tokens | 90.00 MiB | 720.00 MiB | 8.0x |
| 32,768 tokens | 360.00 MiB | 2880.00 MiB | 8.0x |

### Mixture-of-Experts (MoE)

- **Shared Experts (2S)**: Constantly active path for common features:
  $$F_{\text{shared}}(\mathbf{x}) = (\text{SiLU}(\mathbf{x} W_{\text{gate}}^S) \odot (\mathbf{x} W_{\text{up}}^S)) W_{\text{down}}^S$$
- **Routed Experts (32R, Top-4)**: Routed using learned bias offsets without auxiliary load balancing loss:
  $$s_{t, e} = \text{Softmax}(\mathbf{x}_t W_{\text{router}} + \mathbf{b}_e)$$

### V3 Architecture Details

1. **Hadamard-Transformed MLA (H-MLA)**:
   Suppresses latent activation outliers by absorbing the normalized Walsh-Hadamard matrix:
   $$\mathbf{H}_{256} = \frac{1}{16} \mathbf{H}_2^{\otimes 8}$$
   directly into key-value projection weights ($W_{wkv_a}$ and $W_{wkv_b}$) during compilation. This absorption eliminates activation spikes without adding runtime execution overhead.

2. **Partitioned-Balancing Mixture-of-Experts (PB-MoE)**:
   The 32 routed experts are partitioned into 4 functional clans:
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
huggingface-cli download miceworks/Mark-Engineer-V3-1.5B mark_engineer_v3_1_5b_iq4_nl.gguf --local-dir .

# Using Python
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="miceworks/Mark-Engineer-V3-1.5B", filename="mark_engineer_v3_1_5b_iq4_nl.gguf")
```

### Cryptographic Verification

```bash
echo "62b8bd003adcd0780a967faca7163f98e79334809fae2effea5e7f55d4c9bfe4  mark_engineer_v3_1_5b_iq4_nl.gguf" | sha256sum -c -
```

### Ollama

```dockerfile
# Modelfile
FROM ./mark_engineer_v3_1_5b_iq4_nl.gguf
TEMPLATE """{{ if .System }}<|bos|>System: {{ .System }}<|eos|>
{{ end }}{{ if .Prompt }}<|bos|>User: {{ .Prompt }}<|eos|>
{{ end }}<|bos|>Assistant:
<think>
"""

PARAMETER stop "<|eos|>"
PARAMETER temperature 0.3
PARAMETER top_p 0.95
PARAMETER repeat_penalty 1.1

SYSTEM """You are Mark-Engineer-V3-1.5B, a systems-focused neural model."""
```

```bash
ollama create mark-engineer-v3-1.5b -f Modelfile
ollama run mark-engineer-v3-1.5b
```

### llama.cpp

```bash
# CLI generation
llama-cli -m mark_engineer_v3_1_5b_iq4_nl.gguf \
  -p "<|bos|>System: Systems mode active.<|eos|>\n<|bos|>User: Explain Linux memory barriers.<|eos|>\n<|bos|>Assistant:\n<think>\n" \
  -n 512 --temp 0.2

# HTTP server
llama-server -m mark_engineer_v3_1_5b_iq4_nl.gguf --port 8080 -c 32768
```

### Python Inspection

```python
import gguf

reader = gguf.GGUFReader("mark_engineer_v3_1_5b_iq4_nl.gguf")
print("Architecture:", reader.get_field("general.architecture"))
print("Total Tensors:", len(reader.tensors))
total_params = sum(int(t.shape.prod()) for t in reader.tensors)
print(f"Total Parameters: {total_params:,}")
```
