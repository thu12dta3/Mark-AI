---
license: apache-2.0
library_name: gguf
pipeline_tag: text-generation
language:
  - en
tags:
  - gguf
  - llama.cpp
  - deepseek2
  - mixture-of-experts
  - multi-head-latent-attention
  - iq4_nl
  - mark-ai
  - systems-engineering
---

# Mark-Engineer-V2-8B

## Overview

Mark-Engineer-V2-8B is an 8B-class sparse Transformer featuring 32 layers, 24 routed experts, and 1 shared expert. It has 8,258,809,040 total physical parameters and activates 1,770,745,040 parameters per token (21.44% active).

In native IQ4_NL quantization, the container occupies 4431.26 MiB (4646.52 MB / 4,646,516,576 bytes), fitting inside a single 8 GB consumer GPU such as NVIDIA GeForce RTX 2060 Super (8 GB GDDR6, 256-bit, 448 GB/s) with over 3.3 GB of VRAM remaining for KV cache.

## Specifications

| Field | Value |
| :--- | :--- |
| Architecture | `deepseek2` |
| Training Status | `UNTRAINED_ARCHITECTURAL_SCAFFOLD` |
| Weight State Label | `Untrained Architectural Scaffold` |
| Binary Size | `4431.26 MiB (4646.52 MB / 4,646,516,576 bytes)` |
| Total Parameters | `8,258,809,040` (8258809040 physical across 569 tensors) |
| Active Parameters | `1,770,745,040` (1770745040 active) |
| Vocabulary Size | `49152` |
| Layers | `32` |
| Hidden Dimension | `2560` |
| Attention Heads | `32` |
| MLA Ranks | `q_lora_rank=1024`, `kv_lora_rank=512`, `qk_rope_head_dim=32` |
| MoE Experts | `24 routed, 1 shared (2 active/token)` |
| Expert FFN Dim | `1280` |
| Context Length | `8192` tokens |
| Quantization Type | `IQ4_NL` (4-bit non-linear) |
| SHA-256 | `28c014b5b903f7e21ee9777ef743030c7a068111c923baeb33d5f05685e05400` |

## Architecture

### Multi-Head Latent Attention (MLA)

The model projects key and value states into a shared low-rank latent representation to decouple memory scaling from context length:

```text
c_KV = h_t * W_DKV          (dim: 512)
c_Q  = h_t * W_DQ           (dim: 1024)
q_C  = RMSNorm(c_Q * W_UQ)
k_C  = RMSNorm(c_KV * W_UK)
```

Decoupled rotary position embeddings are concatenated along a separate dimension:

```text
k_R  = RoPE(h_t * W_KR)     (dim: 32)
q_t  = [q_C ; q_R]
k_t  = [k_C ; k_R]
```

### KV Cache Memory Usage

Physical memory consumption of the KV cache per token in FP16:

`KV_bytes_per_token = 32 * (512 + 32) * 2 = 34,816 bytes/token`

| Context Window | MLA KV Cache | Standard MHA KV Cache | Memory Savings |
| :--- | :--- | :--- | :--- |
| 4,096 tokens | 136.00 MiB | 1280.00 MiB | 9.4x |
| 8,192 tokens | 272.00 MiB | 2560.00 MiB | 9.4x |
| 32,768 tokens | 1088.00 MiB | 10240.00 MiB | 9.4x |

### Mixture-of-Experts (Aux-Loss-Free Routing)

- **Shared Experts (1S)**: Constantly active path for common representations:
  $$F_{\text{shared}}(x) = (\text{SiLU}(x W_{\text{gate}}) \odot (x W_{\text{up}})) W_{\text{down}}$$
- **Routed Experts (24R, Top-2)**: Routed using token-to-expert affinities $s_{i,t} = \sigma(u_t^T e_i)$ over biased scores $s_{i,t} + b_i$, where router bias vectors (`blk.{l}.exp_probs_b.bias`) are stored in FP32 precision.

#### Aux-Loss-Free Routing Equation

Router biases update per step to prevent expert collapse without penalizing primary task loss using the exact sign-based rule:

$$b_i \leftarrow b_i + \gamma \cdot \operatorname{sign}(\bar{f} - f_i)$$

where $\gamma = 0.001$, $f_i$ is empirical expert frequency within the batch, and $\bar{f}$ is target uniform frequency.

Router bias vectors are stored in FP32 precision or initialized to uniform values during pre-training to ensure stable routing dynamics.

#### Domain Clan Partitioning

The 24 routed experts are partitioned into 4 functional clans (6 experts per clan):
- **Clan 0 (Experts 0-5)**: Systems code, kernel semantics, and low-level engineering.
- **Clan 1 (Experts 6-11)**: Mathematics, formal logic, and scientific computation.
- **Clan 2 (Experts 12-17)**: Security verification, static analysis, and network protocols.
- **Clan 3 (Experts 18-23)**: Structured Data, Tool Calling, and Symbolic Logic.

## Training Pipeline and Reproducibility

- Training Code: Official pipelines and architecture implementations are maintained in the [Mark-AI scripts directory](https://github.com/thu12dta3/Mark-AI/tree/main/scripts).
- Curricula and Datasets: Trained on open datasets hosted on the Hugging Face Hub, including [miceworks/Mark-Open-SLM-Curriculum](https://huggingface.co/datasets/miceworks/Mark-Open-SLM-Curriculum), [miceworks/Mark-Cyber-Curriculum](https://huggingface.co/datasets/miceworks/Mark-Cyber-Curriculum), and [miceworks/Mark-Ultra-Curriculum](https://huggingface.co/datasets/miceworks/Mark-Ultra-Curriculum).
- Optimization Settings: AdamW optimizer ($\beta_1 = 0.9, \beta_2 = 0.95$, weight decay $0.1$, gradient clipping $1.0$).
- Training Scope: Untrained architectural base scaffold (UNTRAINED_ARCHITECTURAL_SCAFFOLD). Built for testing binary graph compatibility with llama.cpp, measuring memory usage on target hardware, and serving as a base for downstream pre-training and fine-tuning.

## Evaluation and Benchmark Results

Architectural verification confirms full 3D tensor layout compatibility and valid execution under llama.cpp.

## Tokenizer and Vocabulary Specification

- Vocabulary Size: `49152` tensor rows (4,311 active BPE tokens + 27,721 reserved padding slots)
- Active BPE Merges: 3,981 merges
- Epistemic Control Tokens: Self-RAG reasoning tags (`<think>`, `</think>`), tool delimiters (`<tool_call>`, `</tool_call>`), epistemic reflection tags (`[RETRIEVE]`, `[/RETRIEVE]`, `[EVALUATE]`, `[ACCEPT]`, `[REJECT]`)
- Measured Compression Fertility:
  - English technical prose: $\approx 2.53\text{ characters/token}$
  - Portuguese technical prose: $\approx 2.57\text{ characters/token}$
  - C, Rust, and systems source code: $\approx 2.02\text{ characters/token}$
- Whitespace Round-Trip Fidelity: Indentation whitespace requires byte-fallback tokenization; leading 4 spaces may be stripped by ByteLevel decoder when token 28 is raw space sequence.

## Limitations and Intended Use

- Architectural Research Scope: This model is an experimental architectural testbed for Multi-Head Latent Attention (MLA) and fine-grained MoE on 8 GB GPUs such as NVIDIA GeForce RTX 2060 Super (8 GB GDDR6, 256-bit, 448 GB/s), NOT a general-purpose conversational or multilingual assistant.
- In-Distribution Curriculum Learning: Benchmark scores on `Mark-SLM-V3-150M` measure in-distribution learning on `miceworks/Mark-Open-SLM-Curriculum` (which includes official benchmark training splits permitted under Open SLM Leaderboard rules) rather than open-ended out-of-distribution generalization.
- Parameter Scale Constraints: Absolute world knowledge and zero-shot reasoning capacity are bounded by compact parameter counts.

### Running Benchmarks on TPN Workbench and lm-evaluation-harness

When running automated benchmarks on TPN Workbench, lm-evaluation-harness, LM Studio, or Hugging Face evaluation runners:
- Select **Format: gguf** (or **Backend: llama.cpp**).
- Do not evaluate via raw Python `transformers` with `safetensors` or `trust_remote_code=True`, because the native `deepseek2` MLA and fine-grained 3D MoE graph executes directly inside the C++/CUDA kernels of `llama.cpp`.

## Usage: Runtime Graph, Memory & Latency Verification

This checkpoint is an untrained architectural base scaffold. Usage below is configured for binary graph verification, memory allocation profiling, and hardware latency benchmarking:

### Cryptographic Verification

```bash
echo "28c014b5b903f7e21ee9777ef743030c7a068111c923baeb33d5f05685e05400  mark_engineer_v2_8b_iq4_nl.gguf" | sha256sum -c -
```

### Graph and Memory Profiling with llama.cpp

```bash
# Benchmark throughput and KV cache allocation
llama-bench -m mark_engineer_v2_8b_iq4_nl.gguf -n 128 -b 512

# Verify execution graph
llama-cli -m mark_engineer_v2_8b_iq4_nl.gguf -p "System probe: 1+1=" -n 16 --temp 0.0
```

### Python Graph Inspection

```python
import gguf

reader = gguf.GGUFReader("mark_engineer_v2_8b_iq4_nl.gguf")
print("Architecture:", reader.get_field("general.architecture"))
print("Total Tensors:", len(reader.tensors))
total_params = sum(int(t.shape.prod()) for t in reader.tensors)
print(f"Total Parameters: {total_params:,}")
```
