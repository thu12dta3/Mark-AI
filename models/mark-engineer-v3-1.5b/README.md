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

Quantized in IQ4_NL, the container occupies 816.95 MiB (856.63 MB / 856,632,544 bytes), leaving ample VRAM on 8 GB GPUs for large source code files.

## Specifications

| Field | Value |
| :--- | :--- |
| Architecture | `deepseek2` |
| Weight State | `TRAINED_CHECKPOINT` |
| Binary Size | `816.95 MiB (856.63 MB / 856,632,544 bytes)` |
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
| SHA-256 | `9d005229429a0f55d61c66ecf2611bd5d03f2a8c76e20512999b8304abc686b3` |

## Official Benchmark Telemetry

### Official Benchmarks (EvalPlus, MultiPL-E, BFCL, MMLU)

Official evaluation results for **Mark-Engineer-V3-1.5B** (`1,520,496,960` physical parameters, `405,729,600` active parameters per token, evaluated on Kaggle Cloud 2x Tesla T4 GPUs in greedy decoding mode `temperature=0.0`, `top_p=1.0`):

| Benchmark Suite | Metric | Score | Target Domain |
| :--- | :--- | :---: | :--- |
| **EvalPlus HumanEval+** | `pass@1 (Base)` | **51.80%** | Python Unit Test Synthesis |
| **EvalPlus HumanEval+** | `pass@1 (Base + Extra)` | **46.30%** | Rigorous Contract & Edge-Case Validation |
| **EvalPlus MBPP+** | `pass@1 (Base)` | **58.50%** | Basic Programming Problems |
| **EvalPlus MBPP+** | `pass@1 (Base + Extra)` | **50.50%** | Contract-Reinforced Programming Synthesis |
| **MultiPL-E (Rust)** | `valid_pass@1` (`humaneval-rs`) | **43.60%** | Systems Code, Memory Safety & Concurrency |
| **MultiPL-E (C++)** | `valid_pass@1` (`humaneval-cpp`) | **47.80%** | High-Performance Systems & Algorithms |
| **MultiPL-E (Systems Avg)** | Average (Rust + C++) | **45.70%** | Multi-Language Systems Engineering |
| **BFCL AST Exact Match** | `ast_exact_match` | **71.00%** | Exact Tool Call & Parameter Matching |
| **BFCL Tool Selection** | `tool_selection_accuracy` | **78.00%** | Correct Function API Selection |
| **BFCL Schema Validity** | `schema_validity_rate` | **92.00%** | Syntactic JSON / Schema Conformance |
| **MMLU College CS** | `acc` | **20.00%** | University-Level Computer Science Theory |
| **MMLU Computer Security**| `acc` | **25.00%** | Vulnerabilities, Cryptography & Network Sec |
| **MMLU High School CS** | `acc` | **25.00%** | Fundamentals of Computation & Algorithms |
| **MMLU College Math** | `acc` | **24.00%** | Advanced Mathematics & Discrete Structures |
| **MMLU CS & Math Average**| Average | **23.50%** | Core Scientific & Computational Reasoning |

## Architecture

### Multi-Head Latent Attention (MLA)

The model projects key and value states into a shared low-rank latent representation to reduce KV cache memory:

```text
c_KV = h_t * W_DKV          (dim: 256)
c_Q  = h_t * W_DQ           (dim: 512)
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

`KV_bytes_per_token = 18 * (256 + 64) * 2 = 11,520 bytes/token`

| Context Window | MLA KV Cache | Standard MHA KV Cache | Memory Savings |
| :--- | :--- | :--- | :--- |
| 4,096 tokens | 45.00 MiB | 360.00 MiB | 8.0x |
| 8,192 tokens | 90.00 MiB | 720.00 MiB | 8.0x |
| 32,768 tokens | 360.00 MiB | 2880.00 MiB | 8.0x |

### Mixture-of-Experts (MoE)

- **Shared Experts (2S)**: Constantly active path for common features:
  `F_shared(x) = (SiLU(x * W_gate) * (x * W_up)) * W_down`
- **Routed Experts (32R, Top-4)**: Routed using learned bias offsets without auxiliary load balancing loss:
  `s_{t,e} = Softmax(x_t * W_router + b_e)`
  `y = x + F_shared(x) + sum_{i in TopK} g_i * F_routed_i(x)`

### V3 Architecture Details

1. **Hadamard-Transformed MLA (H-MLA)**:
   Suppresses latent activation outliers by absorbing the normalized Walsh-Hadamard matrix:
   `H_256 = (1/16) * H_2^(x8)`, `W_fused = H_256 * W_proj`
   directly into key-value projection weights (`W_wkv_a` and `W_wkv_b`) during compilation. This absorption eliminates activation spikes without adding runtime execution overhead.

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
  - 2 bytes: Half-precision (`FP16`) scale factor `d`.
  - 16 bytes: 32 four-bit nibbles referencing the non-linear lookup table `kvalues_iq4nl`:
    `[-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]`
- 1D normalization weights (`*_norm.weight`) and router bias vectors (`exp_probs_b.bias`) remain in full precision `FP32` to preserve numerical accuracy at layer boundaries.
- Routed MoE weights are serialized as contiguous 3D tensors (`blk.{i}.ffn_gate_exps.weight`, `blk.{i}.ffn_up_exps.weight`, `blk.{i}.ffn_down_exps.weight`), avoiding tensor splitting and kernel overhead.

## Training Status

Trained Checkpoint (TRAINED_CHECKPOINT). The model weights were initialized via SVD (Hybrid Polyglot Upcycling) and multi-stage alignment (SFT + Contrastive Length/Quality Tuning), and subsequently validated on Kaggle Cloud GPUs (dual Tesla T4).

Official evaluation executed deterministically in greedy mode (`temperature=0.0`, `top_p=1.0`), and benchmark results are saved in `engineer_v3_1_5b_benchmarks.json`.

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
echo "24093fd96a22c7b4b19e85df2fb0d4e1757361d6f58baa4f36ad0285fa29ac68  mark_engineer_v3_1_5b_iq4_nl.gguf" | sha256sum -c -
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
