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

Mark-VL-2.2B is a native multimodal vision-language model pairing an AnyRes SigLIP vision backbone with a 14-layer DeepSeek2 MoE language model backbone. Utilizing 2D Rotary Position Embeddings (2D-RoPE) and a 2x2 PixelShuffle spatial projection adapter, it enables high-resolution visual comprehension of technical schematics, UI wireframes, architectural diagrams, and dense system telemetry.


### Multimodal Architecture & Artifact Breakdown

Mark-VL-2.2B provides a decoupled vision-language architecture distributed across three distinct artifacts:
1. `mark_vl_2_2b_iq4_nl.gguf`: Primary language backbone container (DeepSeek2 MoE + MLA, 14 physical layers, 24 routed experts, 1 shared expert).
2. `mmproj_mark_vl_2_2b.gguf`: Vision projector module embedding the AnyRes SigLIP visual feature encoder with 2D-RoPE and 2x2 PixelShuffle projection.
3. `mark_vl_2_2b_q4_k_m.gguf`: Legacy quantization artifact retained for compatibility with earlier runtimes.

## Section 1 -- Executive Systems Overview & Target Hardware Profile

### System Role & Engineering Purpose
Mark-VL-2.2B is a native multimodal vision-language model pairing an AnyRes SigLIP vision backbone with a 14-layer DeepSeek2 MoE language model backbone. Utilizing 2D Rotary Position Embeddings (2D-RoPE) and a 2x2 PixelShuffle spatial projection adapter, it enables high-resolution visual comprehension of technical schematics, UI wireframes, architectural diagrams, and dense system telemetry.

### Target Hardware Profile
Target Hardware: Consumer 8GB GPUs. The language backbone occupies 1166.72 MiB (1223.40 MB) and the vision projector mmproj occupies ~800 MiB, totaling under 2.0 GiB of VRAM and allowing seamless multimodal interaction on commodity desktop graphics cards.

- **Memory Interface**: Optimized for 256-bit GDDR6 memory controllers delivering ~448 GB/s bandwidth.
- **Cache Hierarchy**: Fitted to standard 4 MB to 32 MB L2 caches with minimal cache thrashing.
- **VRAM Constraint**: Strictly designed to reside comfortably within 8 GB consumer GPU boundaries, leaving generous capacity for dynamic operating memory and KV cache expansion.

---

## Section 2 -- Mathematical Formulation & Neural Topology

### Multi-Head Latent Attention (MLA)
The model compresses the traditional key-value memory footprint using low-rank latent subspace projections:

$$\mathbf{c}_t^{KV} = W^{DKV} \mathbf{h}_t \quad (\mathbf{c}_t^{KV} \in \mathbb{R}^{320})$$

$$\mathbf{c}_t^Q = W^{DQ} \mathbf{h}_t \quad (\mathbf{c}_t^Q \in \mathbb{R}^{640})$$

$$\mathbf{q}_t^C = \text{RMSNorm}(W^{UQ} \mathbf{c}_t^Q), \quad \mathbf{k}_t^C = \text{RMSNorm}(W^{UK} \mathbf{c}_t^{KV})$$

Decoupled rotary position embeddings are concatenated to preserve spatial position awareness without expanding cached state dimensions:

$$\mathbf{k}_t^R = \text{RoPE}(W^{KR} \mathbf{h}_t) \quad (\mathbf{k}_t^R \in \mathbb{R}^{32})$$

$$\mathbf{q}_{t, i} = [\mathbf{q}_{t, i}^C; \mathbf{q}_{t, i}^R], \quad \mathbf{k}_{t, i} = [\mathbf{k}_{t, i}^C; \mathbf{k}_t^R]$$

### Exact KV Cache Memory Sizing
The exact physical memory consumption of the KV cache per token is calculated as:

$$\text{KV}_{\text{bytes/token}} = 14 \times (320 + 32) \times 2 \text{ bytes} = 9,856 \text{ bytes/token}$$

| Context Window | MLA KV Cache Memory | Standard MHA KV Cache Memory | Memory Savings |
| :--- | :--- | :--- | :--- |
| 4,096 tokens | 38.50 MiB | 280.00 MiB | 7.3x compression |
| 8,192 tokens | 77.00 MiB | 560.00 MiB | 7.3x compression |
| 32,768 tokens | 308.00 MiB | 2240.00 MiB | 7.3x compression |

### Mixture-of-Experts (MoE) Topology
- **Leading Dense Layers**: Dense front-end layers ensure foundational feature extraction before routing.
- **Shared Experts (1S)**: Continuously active shared pathway captures common linguistic and grammatical invariants:
  $$F_{\text{shared}}(\mathbf{x}) = (\text{SiLU}(\mathbf{x} W_{\text{gate}}^S) \odot (\mathbf{x} W_{\text{up}}^S)) W_{\text{down}}^S$$
- **Routed Experts (24R, Top-2)**: Aux-Loss-Free dynamic router balances token routing across specialists using learned bias offsets:
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

The metrics below represent the single source of truth extracted directly from `mark_vl_2_2b_iq4_nl.gguf` via `gguf.GGUFReader`:

| Parameter Field | Ground-Truth Audit Value |
| :--- | :--- |
| **Model Identifier** | `mark-vl-2.2b` |
| **Hugging Face Repository** | [miceworks/Mark-VL-2.2B](https://huggingface.co/miceworks/Mark-VL-2.2B) |
| **GGUF Architecture Tag** | `deepseek2` |
| **Container File Name** | `mark_vl_2_2b_iq4_nl.gguf` |
| **Exact Binary Size** | `1166.72 MiB (1223.40 MB / 1,223,397,152 bytes)` |
| **Full SHA-256 Checksum** | `073d3c00eaadfb98ef7c8b5276e355db134b4b1282ea74d8ce3d6116535cf354` |
| **Weight State** | `TRAINED_CHECKPOINT_VALIDATED` |
| **Total Physical Parameters** | `2,174,310,096` parameters (2174310096 physical) |
| **Active Parameters per Token** | `395,499,216` parameters (395499216 active) |
| **Total Tensor Count** | `255` tensors |
| **Quantization Breakdown** | `{'IQ4_NL': 184, 'F32': 71}` |
| **1D RMSNorm Precision** | `100% FP32` (Zero degradation on normalization boundaries) |
| **Transformer Layers (num_hidden_layers)** | `14` |
| **Hidden Dimension (hidden_size)** | `1280` |
| **Attention Heads (num_attention_heads)** | `16` |
| **MLA Dimensioning** | `q_lora_rank=640`, `kv_lora_rank=320`, `v_head_dim=96` |
| **Decoupled RoPE Dimensions** | `qk_nope_head_dim=64`, `qk_rope_head_dim=32` |
| **MoE Topology** | `24 Routed Experts, 1 Shared Expert (2 active/token)` |
| **MoE Intermediate Size** | `1504` |
| **Vocabulary Size (vocab_size)** | `32000` |
| **Maximum Context Window** | `16384` tokens |

---

## Section 5 -- Weight Provenance, Training Curriculum & Scientific Scope

### Weight State & Provenance
- **Status**: `TRAINED_CHECKPOINT_VALIDATED`
- **Provenance Overview**: Trained on multimodal diagrammatic pairs, technical schematics, and structured documents from miceworks/Mark-Ultra-Curriculum. The repository provides the primary IQ4_NL language container (mark_vl_2_2b_iq4_nl.gguf), the vision adapter (mmproj_mark_vl_2_2b.gguf), and the legacy Q4_K_M artifact.

### Scientific Boundary & Transparency
All reported metrics represent physical parameter counts and deterministic SHA-256 checksums. No synthetic benchmarks, unverified FLOPS claims, or unverified hardware compatibility flags (such as vLLM without native DeepSeek2 3D tensor support) are declared.

---

## Section 6 -- Deployment, Inference & Cryptographic Audit Guide

### Cryptographic Verification
Verify the binary integrity against the single source of truth:

```bash
# Verify cryptographic hash
echo "073d3c00eaadfb98ef7c8b5276e355db134b4b1282ea74d8ce3d6116535cf354  mark_vl_2_2b_iq4_nl.gguf" | sha256sum -c -
```

### Ollama Deployment
Register and serve the model via Ollama using the canonical token delimiters:

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

SYSTEM """You are Mark-VL-2.2B, a sparse systems neural architecture."""
```

```bash
ollama create mark-vl-2.2b -f Modelfile
ollama run mark-vl-2.2b
```

### llama.cpp Execution

```bash
# Direct CLI inference
llama-cli -m mark_vl_2_2b_iq4_nl.gguf \
  -p "<|bos|>System: Systems engineering mode active.<|eos|>\n<|bos|>User: Analyze kernel synchronization primitives.<|eos|>\n<|bos|>Assistant:\n<think>\n" \
  -n 512 --temp 0.2

# Launch OpenAI-compatible HTTP inference daemon
llama-server -m mark_vl_2_2b_iq4_nl.gguf --port 8080 -c 16384
```

### Forensic Audit Script
Execute local inspection using the official `gguf` library:

```python
import gguf

reader = gguf.GGUFReader("mark_vl_2_2b_iq4_nl.gguf")
print(f"Architecture: {reader.get_field('general.architecture')}")
print(f"Total Tensors: {len(reader.tensors)}")
total_params = sum(int(t.shape.prod()) for t in reader.tensors)
print(f"Audited Total Parameters: {total_params:,}")
```
