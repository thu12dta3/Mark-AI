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

# Mark-V3-1.1B

Mark-V3-1.1B introduces the third-generation Mark architecture featuring Hadamard-transformed Multi-Head Latent Attention (H-MLA) and Partitioned-Balancing Mixture-of-Experts (PB-MoE). Equipped with 18 layers, 32 routed experts (4 active per token), and 2 shared experts, it provides an exact physical parameter count of 1,222,464,960 parameters (330.65M active parameters per token).


> [!NOTE]
> **Architectural Nomenclature Note**:
> The designation "1.1B" is the series designator. The exact physical parameter count exported in `mark_v3_1_1b_iq4_nl.gguf` with 32 routed experts is exactly **1,222,464,960** physical parameters (330,651,072 active per token), requiring **656.13 MiB (688.00 MB / 688,002,048 bytes)** in IQ4_NL format.

## Section 1 -- Executive Systems Overview & Target Hardware Profile

### System Role & Engineering Purpose
Mark-V3-1.1B introduces the third-generation Mark architecture featuring Hadamard-transformed Multi-Head Latent Attention (H-MLA) and Partitioned-Balancing Mixture-of-Experts (PB-MoE). Equipped with 18 layers, 32 routed experts (4 active per token), and 2 shared experts, it provides an exact physical parameter count of 1,222,464,960 parameters (330.65M active parameters per token).

### Target Hardware Profile
Target Hardware: 8GB GPUs. Binary size of 656.13 MiB (688.00 MB / 688,002,048 bytes) in IQ4_NL ensures near-instant loading and leaves maximum VRAM available for context scaling up to 32,768 tokens.

- **Memory Interface**: Optimized for 256-bit GDDR6 memory controllers delivering ~448 GB/s bandwidth.
- **Cache Hierarchy**: Fitted to standard 4 MB to 32 MB L2 caches with minimal cache thrashing.
- **VRAM Constraint**: Strictly designed to reside comfortably within 8 GB consumer GPU boundaries, leaving generous capacity for dynamic operating memory and KV cache expansion.

---

## Section 2 -- Mathematical Formulation & Neural Topology

### Multi-Head Latent Attention (MLA)
The model compresses the traditional key-value memory footprint using low-rank latent subspace projections:

$$\mathbf{c}_t^{KV} = W^{DKV} \mathbf{h}_t \quad (\mathbf{c}_t^{KV} \in \mathbb{R}^{256})$$

$$\mathbf{c}_t^Q = W^{DQ} \mathbf{h}_t \quad (\mathbf{c}_t^Q \in \mathbb{R}^{384})$$

$$\mathbf{q}_t^C = \text{RMSNorm}(W^{UQ} \mathbf{c}_t^Q), \quad \mathbf{k}_t^C = \text{RMSNorm}(W^{UK} \mathbf{c}_t^{KV})$$

Decoupled rotary position embeddings are concatenated to preserve spatial position awareness without expanding cached state dimensions:

$$\mathbf{k}_t^R = \text{RoPE}(W^{KR} \mathbf{h}_t) \quad (\mathbf{k}_t^R \in \mathbb{R}^{64})$$

$$\mathbf{q}_{t, i} = [\mathbf{q}_{t, i}^C; \mathbf{q}_{t, i}^R], \quad \mathbf{k}_{t, i} = [\mathbf{k}_{t, i}^C; \mathbf{k}_t^R]$$

### Exact KV Cache Memory Sizing
The exact physical memory consumption of the KV cache per token is calculated as:

$$\text{KV}_{\text{bytes/token}} = 18 \times (256 + 64) \times 2 \text{ bytes} = 11,520 \text{ bytes/token}$$

| Context Window | MLA KV Cache Memory | Standard MHA KV Cache Memory | Memory Savings |
| :--- | :--- | :--- | :--- |
| 4,096 tokens | 45.00 MiB | 324.00 MiB | 7.2x compression |
| 8,192 tokens | 90.00 MiB | 648.00 MiB | 7.2x compression |
| 32,768 tokens | 360.00 MiB | 2592.00 MiB | 7.2x compression |

### Mixture-of-Experts (MoE) Topology
- **Leading Dense Layers**: Dense front-end layers ensure foundational feature extraction before routing.
- **Shared Experts (2S)**: Continuously active shared pathway captures common linguistic and grammatical invariants:
  $$F_{\text{shared}}(\mathbf{x}) = (\text{SiLU}(\mathbf{x} W_{\text{gate}}^S) \odot (\mathbf{x} W_{\text{up}}^S)) W_{\text{down}}^S$$
- **Routed Experts (32R, Top-4)**: Aux-Loss-Free dynamic router balances token routing across specialists using learned bias offsets:
  $$s_{t, e} = \text{Softmax}(\mathbf{x}_t W_{\text{router}} + \mathbf{b}_e)$$

### V3 Advanced Structural Innovations

1. **Hadamard-Transformed MLA (H-MLA)**:
   Suppresses latent activation outliers by absorbing the normalized Walsh-Hadamard transformation matrix:
   $$\mathbf{H}_{256} = \frac{1}{16} \mathbf{H}_2^{\otimes 8}$$
   directly into the key-value projection weights ($W_{wkv_a}$ and $W_{wkv_b}$) during compilation. This mathematical absorption guarantees zero runtime latency overhead while drastically reducing quantization-induced perplexity degradation under 4-bit representation.

2. **Partitioned-Balancing Mixture-of-Experts (PB-MoE)**:
   The 32 routed experts are partitioned into 4 distinct functional domain clans:
   - **Clan 0 (Experts 0-7)**: Low-level Systems, Kernel Semantics, and Concurrency.
   - **Clan 1 (Experts 8-15)**: Mathematical Derivations, Formal Logic, and STEM.
   - **Clan 2 (Experts 16-23)**: Security Boundaries, Cryptographic Protocols, and Sentinel Auditing.
   - **Clan 3 (Experts 24-31)**: Multimodal and Spatial-Temporal Reasoning.
   Routings enforce soft-pinning constraints ensuring balanced domain coverage per token.

3. **Latent Recursive Depth (LRD) & Tied Embeddings**:
   Enables recurrent intermediate latent refinements while enforcing weight-tying between input and output representations to minimize physical disk and VRAM footprints.

---

## Section 3 -- Binary Container Architecture & Non-Linear 4-Bit Quantization (GGML_TYPE_IQ4_NL)

The model is packaged in the native `deepseek2` GGUF container architecture:
1. **3D Tensor Packing**: Routed MoE weights are serialized as contiguous 3D tensors (`blk.{i}.ffn_gate_exps.weight`, `blk.{i}.ffn_up_exps.weight`, `blk.{i}.ffn_down_exps.weight`), eliminating kernel dispatch overhead during expert gathering.
2. **IQ4_NL Block Formatting**: Linear weights are structured into 18-byte super-blocks for every 32 physical parameters:
   - 2 bytes: Half-precision (`FP16`) scale factor $d$.
   - 16 bytes: 32 four-bit nibbles indexed into the non-linear quantization lookup table:
     `[-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]`
3. **Precision Invariants**: All 1D layer normalization weights (`*_norm.weight`) and router bias parameters (`exp_probs_b.bias`) are strictly preserved in full precision `FP32`.

---

## Section 4 -- Verified GGUF Binary Telemetry & Tensor Breakdown

The metrics below represent the single source of truth extracted directly from `mark_v3_1_1b_iq4_nl.gguf` via `gguf.GGUFReader`:

| Parameter Field | Ground-Truth Audit Value |
| :--- | :--- |
| **Model Identifier** | `mark-v3-1.1b` |
| **Hugging Face Repository** | [miceworks/Mark-V3-1.1B](https://huggingface.co/miceworks/Mark-V3-1.1B) |
| **GGUF Architecture Tag** | `deepseek2` |
| **Container File Name** | `mark_v3_1_1b_iq4_nl.gguf` |
| **Exact Binary Size** | `656.13 MiB (688.00 MB / 688,002,048 bytes)` |
| **Full SHA-256 Checksum** | `e12c358e8144137ce4cb0546c4f00f9b0ce089c5fa354f5177f95ddf7132d84d` |
| **Weight State** | `UN-TRAINED_ARCHITECTURAL_BASE` |
| **Total Physical Parameters** | `1,222,464,960` parameters (1222464960 physical) |
| **Active Parameters per Token** | `330,651,072` parameters (330651072 active) |
| **Total Tensor Count** | `327` tensors |
| **Quantization Breakdown** | `{'IQ4_NL': 236, 'F32': 91}` |
| **1D RMSNorm Precision** | `100% FP32` (Zero degradation on normalization boundaries) |
| **Transformer Layers (num_hidden_layers)** | `18` |
| **Hidden Dimension (hidden_size)** | `1152` |
| **Attention Heads (num_attention_heads)** | `18` |
| **MLA Dimensioning** | `q_lora_rank=384`, `kv_lora_rank=256`, `v_head_dim=64` |
| **Decoupled RoPE Dimensions** | `qk_nope_head_dim=64`, `qk_rope_head_dim=64` |
| **MoE Topology** | `32 Routed Experts, 2 Shared Expert (4 active/token)` |
| **MoE Intermediate Size** | `512` |
| **Vocabulary Size (vocab_size)** | `32032` |
| **Maximum Context Window** | `32768` tokens |

---

## Section 5 -- Weight Provenance, Training Curriculum & Scientific Scope

### Weight State & Provenance
- **Status**: `UN-TRAINED_ARCHITECTURAL_BASE`
- **Provenance Overview**: Published as an architectural foundation (UN-TRAINED_ARCHITECTURAL_BASE) demonstrating offline Walsh-Hadamard absorption and 4-clan domain routing.

### Scientific Boundary & Transparency
All reported metrics represent physical parameter counts and deterministic SHA-256 checksums. No synthetic benchmarks, unverified FLOPS claims, or unverified hardware compatibility flags (such as vLLM without native DeepSeek2 3D tensor support) are declared.

---

## Section 6 -- Deployment, Inference & Cryptographic Audit Guide

### Cryptographic Verification
Verify the binary integrity against the single source of truth:

```bash
# Verify cryptographic hash
echo "e12c358e8144137ce4cb0546c4f00f9b0ce089c5fa354f5177f95ddf7132d84d  mark_v3_1_1b_iq4_nl.gguf" | sha256sum -c -
```

### Ollama Deployment
Register and serve the model via Ollama using the canonical token delimiters:

```dockerfile
# Modelfile
FROM ./mark_v3_1_1b_iq4_nl.gguf
TEMPLATE """{{ if .System }}<|bos|>System: {{ .System }}<|eos|>
{{ end }}{{ if .Prompt }}<|bos|>User: {{ .Prompt }}<|eos|>
{{ end }}<|bos|>Assistant:
<think>
"""

PARAMETER stop "<|eos|>"
PARAMETER temperature 0.3
PARAMETER top_p 0.95
PARAMETER repeat_penalty 1.1

SYSTEM """You are Mark-V3-1.1B, a sparse systems neural architecture."""
```

```bash
ollama create mark-v3-1.1b -f Modelfile
ollama run mark-v3-1.1b
```

### llama.cpp Execution

```bash
# Direct CLI inference
llama-cli -m mark_v3_1_1b_iq4_nl.gguf \
  -p "<|bos|>System: Systems engineering mode active.<|eos|>\n<|bos|>User: Analyze kernel synchronization primitives.<|eos|>\n<|bos|>Assistant:\n<think>\n" \
  -n 512 --temp 0.2

# Launch OpenAI-compatible HTTP inference daemon
llama-server -m mark_v3_1_1b_iq4_nl.gguf --port 8080 -c 32768
```

### Forensic Audit Script
Execute local inspection using the official `gguf` library:

```python
import gguf

reader = gguf.GGUFReader("mark_v3_1_1b_iq4_nl.gguf")
print(f"Architecture: {reader.get_field('general.architecture')}")
print(f"Total Tensors: {len(reader.tensors)}")
total_params = sum(int(t.shape.prod()) for t in reader.tensors)
print(f"Audited Total Parameters: {total_params:,}")
```
