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

Mark-SLM-V2-302M is an advanced compact small language model combining an expansive 49,152 token vocabulary with a 16-layer DeepSeek2 MoE/MLA topology. The model features 16 routed experts and 1 shared expert (2 active routed experts per token), delivering high linguistic fidelity and reasoning capacity across both English and Portuguese while maintaining an active parameter footprint of only 155.72M parameters.


## Section 1 -- Executive Systems Overview & Target Hardware Profile

### System Role & Engineering Purpose
Mark-SLM-V2-302M is an advanced compact small language model combining an expansive 49,152 token vocabulary with a 16-layer DeepSeek2 MoE/MLA topology. The model features 16 routed experts and 1 shared expert (2 active routed experts per token), delivering high linguistic fidelity and reasoning capacity across both English and Portuguese while maintaining an active parameter footprint of only 155.72M parameters.

### Target Hardware Profile
Target Hardware: Consumer workstations and gaming laptops with 8GB VRAM (256-bit GDDR6 bus, 448 GB/s bandwidth). With a binary footprint of 177.76 MiB (186.39 MB), it requires minimal VRAM, allowing ultra-fast memory bus streaming and zero swap latency during interactive local execution.

- **Memory Interface**: Optimized for 256-bit GDDR6 memory controllers delivering ~448 GB/s bandwidth.
- **Cache Hierarchy**: Fitted to standard 4 MB to 32 MB L2 caches with minimal cache thrashing.
- **VRAM Constraint**: Strictly designed to reside comfortably within 8 GB consumer GPU boundaries, leaving generous capacity for dynamic operating memory and KV cache expansion.

---

## Section 2 -- Mathematical Formulation & Neural Topology

### Multi-Head Latent Attention (MLA)
The model compresses the traditional key-value memory footprint using low-rank latent subspace projections:

$$\mathbf{c}_t^{KV} = W^{DKV} \mathbf{h}_t \quad (\mathbf{c}_t^{KV} \in \mathbb{R}^{192})$$

$$\mathbf{c}_t^Q = W^{DQ} \mathbf{h}_t \quad (\mathbf{c}_t^Q \in \mathbb{R}^{384})$$

$$\mathbf{q}_t^C = \text{RMSNorm}(W^{UQ} \mathbf{c}_t^Q), \quad \mathbf{k}_t^C = \text{RMSNorm}(W^{UK} \mathbf{c}_t^{KV})$$

Decoupled rotary position embeddings are concatenated to preserve spatial position awareness without expanding cached state dimensions:

$$\mathbf{k}_t^R = \text{RoPE}(W^{KR} \mathbf{h}_t) \quad (\mathbf{k}_t^R \in \mathbb{R}^{32})$$

$$\mathbf{q}_{t, i} = [\mathbf{q}_{t, i}^C; \mathbf{q}_{t, i}^R], \quad \mathbf{k}_{t, i} = [\mathbf{k}_{t, i}^C; \mathbf{k}_t^R]$$

### Exact KV Cache Memory Sizing
The exact physical memory consumption of the KV cache per token is calculated as:

$$\text{KV}_{\text{bytes/token}} = 16 \times (192 + 32) \times 2 \text{ bytes} = 7,168 \text{ bytes/token}$$

| Context Window | MLA KV Cache Memory | Standard MHA KV Cache Memory | Memory Savings |
| :--- | :--- | :--- | :--- |
| 4,096 tokens | 28.00 MiB | 192.00 MiB | 6.9x compression |
| 8,192 tokens | 56.00 MiB | 384.00 MiB | 6.9x compression |
| 32,768 tokens | 224.00 MiB | 1536.00 MiB | 6.9x compression |

### Mixture-of-Experts (MoE) Topology
- **Leading Dense Layers**: Dense front-end layers ensure foundational feature extraction before routing.
- **Shared Experts (1S)**: Continuously active shared pathway captures common linguistic and grammatical invariants:
  $$F_{\text{shared}}(\mathbf{x}) = (\text{SiLU}(\mathbf{x} W_{\text{gate}}^S) \odot (\mathbf{x} W_{\text{up}}^S)) W_{\text{down}}^S$$
- **Routed Experts (16R, Top-2)**: Aux-Loss-Free dynamic router balances token routing across specialists using learned bias offsets:
  $$s_{t, e} = \text{Softmax}(\mathbf{x}_t W_{\text{router}} + \mathbf{b}_e)$$

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

The metrics below represent the single source of truth extracted directly from `mark_slm_v2_302m_iq4_nl.gguf` via `gguf.GGUFReader`:

| Parameter Field | Ground-Truth Audit Value |
| :--- | :--- |
| **Model Identifier** | `mark-slm-v2-302m` |
| **Hugging Face Repository** | [miceworks/Mark-SLM-V2-302M](https://huggingface.co/miceworks/Mark-SLM-V2-302M) |
| **GGUF Architecture Tag** | `deepseek2` |
| **Container File Name** | `mark_slm_v2_302m_iq4_nl.gguf` |
| **Exact Binary Size** | `177.76 MiB (186.39 MB / 186,394,944 bytes)` |
| **Full SHA-256 Checksum** | `34a3e0e39cd02cd6a12345b9558ea094877530ef3bde52e64552f20ca48ce38e` |
| **Weight State** | `TRAINED_CHECKPOINT_VALIDATED` |
| **Total Physical Parameters** | `329,132,000` parameters (329132000 physical) |
| **Active Parameters per Token** | `155,723,744` parameters (155723744 active) |
| **Total Tensor Count** | `281` tensors |
| **Quantization Breakdown** | `{'IQ4_NL': 202, 'F32': 79}` |
| **1D RMSNorm Precision** | `100% FP32` (Zero degradation on normalization boundaries) |
| **Transformer Layers (num_hidden_layers)** | `16` |
| **Hidden Dimension (hidden_size)** | `768` |
| **Attention Heads (num_attention_heads)** | `12` |
| **MLA Dimensioning** | `q_lora_rank=384`, `kv_lora_rank=192`, `v_head_dim=64` |
| **Decoupled RoPE Dimensions** | `qk_nope_head_dim=64`, `qk_rope_head_dim=32` |
| **MoE Topology** | `16 Routed Experts, 1 Shared Expert (2 active/token)` |
| **MoE Intermediate Size** | `384` |
| **Vocabulary Size (vocab_size)** | `49152` |
| **Maximum Context Window** | `4096` tokens |

---

## Section 5 -- Weight Provenance, Training Curriculum & Scientific Scope

### Weight State & Provenance
- **Status**: `TRAINED_CHECKPOINT_VALIDATED`
- **Provenance Overview**: Trained for 10,000 steps on miceworks/Mark-Ultra-Curriculum, achieving a converged validation loss of 0.263975. The expert dimensions were projected from 400 to 384 via truncated Singular Value Decomposition (SVD), preserving the 384 principal singular components to maintain strict divisibility by 32 for the IQ4_NL quantization super-blocks while keeping 100% of the trained MLA attention, embedding, and RMSNorm weights intact.

### Scientific Boundary & Transparency
All reported metrics represent physical parameter counts and deterministic SHA-256 checksums. No synthetic benchmarks, unverified FLOPS claims, or unverified hardware compatibility flags (such as vLLM without native DeepSeek2 3D tensor support) are declared.

---

## Section 6 -- Deployment, Inference & Cryptographic Audit Guide

### Cryptographic Verification
Verify the binary integrity against the single source of truth:

```bash
# Verify cryptographic hash
echo "34a3e0e39cd02cd6a12345b9558ea094877530ef3bde52e64552f20ca48ce38e  mark_slm_v2_302m_iq4_nl.gguf" | sha256sum -c -
```

### Ollama Deployment
Register and serve the model via Ollama using the canonical token delimiters:

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

SYSTEM """You are Mark-SLM-V2-302M, a sparse systems neural architecture."""
```

```bash
ollama create mark-slm-v2-302m -f Modelfile
ollama run mark-slm-v2-302m
```

### llama.cpp Execution

```bash
# Direct CLI inference
llama-cli -m mark_slm_v2_302m_iq4_nl.gguf \
  -p "<|bos|>System: Systems engineering mode active.<|eos|>\n<|bos|>User: Analyze kernel synchronization primitives.<|eos|>\n<|bos|>Assistant:\n<think>\n" \
  -n 512 --temp 0.2

# Launch OpenAI-compatible HTTP inference daemon
llama-server -m mark_slm_v2_302m_iq4_nl.gguf --port 8080 -c 4096
```

### Forensic Audit Script
Execute local inspection using the official `gguf` library:

```python
import gguf

reader = gguf.GGUFReader("mark_slm_v2_302m_iq4_nl.gguf")
print(f"Architecture: {reader.get_field('general.architecture')}")
print(f"Total Tensors: {len(reader.tensors)}")
total_params = sum(int(t.shape.prod()) for t in reader.tensors)
print(f"Audited Total Parameters: {total_params:,}")
```
