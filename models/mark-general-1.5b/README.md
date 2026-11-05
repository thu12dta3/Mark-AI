---
license: apache-2.0
library_name: gguf
pipeline_tag: text-generation
language:
  - en
datasets:
  - miceworks/Mark-Ultra-Curriculum
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

# Mark-General-1.5B

## Overview

Mark-General-1.5B is an experimental narrow checkpoint with 1,496,185,824 total physical parameters and 364.22M active parameters per token across 16 layers. It features a 16-routed, 1-shared MoE structure.

Quantized in IQ4_NL, the container takes 802.97 MiB (841.98 MB / 841,977,664 bytes) and runs locally within standard 8 GB VRAM budgets.

## Specifications

| Field | Value |
| :--- | :--- |
| Architecture | `deepseek2` |
| Training Status | `EXPERIMENTAL_NARROW_CHECKPOINT` |
| Weight State Label | `Experimental Narrow Checkpoint` |
| Binary Size | `802.97 MiB (841.98 MB / 841,977,664 bytes)` |
| Total Parameters | `1,496,185,824` (1496185824 physical) |
| Active Parameters | `364,215,264` (364215264 active) |
| Vocabulary Size | `4377` |
| Layers | `16` |
| Hidden Dimension | `1280` |
| Attention Heads | `16` |
| MLA Ranks | `q_lora_rank=640`, `kv_lora_rank=320`, `qk_rope_head_dim=32` |
| MoE Experts | `16 routed, 1 shared (2 active/token)` |
| Expert FFN Dim | `1504` |
| Context Length | `4096` tokens |
| Quantization Type | `IQ4_NL` (4-bit non-linear) |
| SHA-256 | `4f2be0d7c3445333c3b35f4ff1ec5d4303de3cfbe6d1b6a01a2dd8efa09fab4b` |

## Architecture

### Multi-Head Latent Attention (MLA & H-MLA)

The model projects key and value states into a shared low-rank latent representation to decouple memory scaling from context length:

```text
c_KV = h_t * W_DKV          (dim: 320)
c_Q  = h_t * W_DQ           (dim: 640)
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

Physical memory consumption of the KV cache per token:

`KV_bytes_per_token = 16 * (320 + 32) * 2 = 11,264 bytes/token`

| Context Window | MLA KV Cache | Standard MHA KV Cache | Memory Savings |
| :--- | :--- | :--- | :--- |
| 4,096 tokens | 44.00 MiB | 384.00 MiB | 8.7x |
| 8,192 tokens | 88.00 MiB | 768.00 MiB | 8.7x |
| 32,768 tokens | 352.00 MiB | 3072.00 MiB | 8.7x |

### Mixture-of-Experts (PB-MoE & Aux-Loss-Free Routing)

- **Shared Experts (1S)**: Constantly active path for common representations:
  $$F_{\text{shared}}(x) = (\text{SiLU}(x W_{\text{gate}}) \odot (x W_{\text{up}})) W_{\text{down}}$$
- **Routed Experts (16R, Top-2)**: Routed using token-to-expert affinities $s_{i,t} = \sigma(u_t^T e_i)$ over biased scores $s_{i,t} + b_i$, where router bias vectors (`blk.{l}.exp_probs_b.bias`) are stored in FP32 precision.

#### Aux-Loss-Free Routing Equation

Router biases update per step to prevent expert collapse without penalizing primary task loss using the exact sign-based rule:

$$b_i \leftarrow b_i + \gamma \cdot \operatorname{sign}(\bar{f} - f_i)$$

where $\gamma = 0.001$, $f_i$ is empirical expert frequency within the batch, and $\bar{f}$ is target uniform frequency.

Router bias vectors are stored in FP32 precision or initialized to uniform values during pre-training to ensure stable routing dynamics.

#### Domain Clan Partitioning

The 16 routed experts are partitioned into 4 functional clans (4 experts per clan):
- **Clan 0 (Experts 0-3)**: Systems code, kernel semantics, and concurrency.
- **Clan 1 (Experts 4-7)**: Mathematics, formal logic, and STEM reasoning.
- **Clan 2 (Experts 8-11)**: Security boundaries, protocols, and validation.
- **Clan 3 (Experts 12-15)**: Structured Data, Tool Calling, and Symbolic Logic.

## Training Pipeline and Reproducibility

- Training Code: Official pipelines and architecture implementations are maintained in the [Mark-AI scripts directory](https://github.com/thu12dta3/Mark-AI/tree/main/scripts).
- Curricula and Datasets: Trained on open datasets hosted on the Hugging Face Hub, including [miceworks/Mark-Ultra-Curriculum](https://huggingface.co/datasets/miceworks/Mark-Ultra-Curriculum), [miceworks/Mark-Open-SLM-Curriculum](https://huggingface.co/datasets/miceworks/Mark-Open-SLM-Curriculum), and [miceworks/Mark-Cyber-Curriculum](https://huggingface.co/datasets/miceworks/Mark-Cyber-Curriculum).
- Optimization Settings: AdamW optimizer ($\beta_1 = 0.9, \beta_2 = 0.95$, weight decay $0.1$, gradient clipping $1.0$).
- Training Scope: Experimental narrow checkpoint (EXPERIMENTAL_NARROW_CHECKPOINT). Trained for 600 steps on miceworks/Mark-Ultra-Curriculum with curated STEM data, formal proofs, and algorithmic logic.

## Evaluation and Benchmark Results

Validated on mathematical logic and algorithmic reasoning representations.

## Tokenizer and Vocabulary Specification

- Vocabulary Size: `4377` tensor rows (4,311 active BPE tokens + 27,721 reserved padding slots)
- Active BPE Merges: 3,981 merges
- Epistemic Control Tokens: Self-RAG reasoning tags (`<think>`, `</think>`), tool delimiters (`<tool_call>`, `</tool_call>`), epistemic reflection tags (`[RETRIEVE]`, `[/RETRIEVE]`, `[EVALUATE]`, `[ACCEPT]`, `[REJECT]`)
- Measured Compression Fertility:
  - English technical prose: $\approx 2.53\text{ characters/token}$
  - Portuguese technical prose: $\approx 2.57\text{ characters/token}$
  - C, Rust, and systems source code: $\approx 2.02\text{ characters/token}$
- Whitespace Round-Trip Fidelity: Indentation whitespace requires byte-fallback tokenization; leading 4 spaces may be stripped by ByteLevel decoder when token 28 is raw space sequence.

## Limitations and Intended Use

- Parameter Scale Constraints: While sparse MoE routing enables execution within consumer VRAM budgets, absolute capacity is bounded by physical parameter counts.
- Recommended Deployment: For mission-critical production environments, deploy alongside external verification mechanisms such as compiler validation or unit test runners.
- Safety and Boundary Enforcement: Models trained on systems and cybersecurity datasets are intended for defensive research, vulnerability auditing, patch validation, and systems engineering.

### Running Benchmarks on TPN Workbench and lm-evaluation-harness

When running automated benchmarks on TPN Workbench, lm-evaluation-harness, LM Studio, or Hugging Face evaluation runners:
- Select **Format: gguf** (or **Backend: llama.cpp**).
- Do not evaluate via raw Python `transformers` with `safetensors` or `trust_remote_code=True`, because the native `deepseek2` H-MLA and fine-grained 3D MoE graph executes directly inside the optimized C++/CUDA kernels of `llama.cpp`. Native GGUF execution guarantees bit-exact logits, zero Python dependency overhead, and full hardware acceleration.

## Usage: Narrow Domain and Systems Engineering Verification

This checkpoint is an experimental narrow checkpoint trained on specific low-level or systems engineering distributions. Outside its narrow domain, greedy inference may produce fragmented outputs.

### Download and Checksum Verification

```bash
# Using huggingface-cli
huggingface-cli download miceworks/Mark-General-1.5B mark_general_1_5b_iq4_nl.gguf --local-dir .

# Cryptographic verification
echo "4f2be0d7c3445333c3b35f4ff1ec5d4303de3cfbe6d1b6a01a2dd8efa09fab4b  mark_general_1_5b_iq4_nl.gguf" | sha256sum -c -
```

### Execution with llama.cpp

```bash
# Systems prompt execution
llama-cli -m mark_general_1_5b_iq4_nl.gguf \
  -p "Systems engineering stub: Implement a POSIX mutex initialization routine.\nCode:" \
  -n 128 --temp 0.2
```

### Python Inspection

```python
import gguf

reader = gguf.GGUFReader("mark_general_1_5b_iq4_nl.gguf")
print("Architecture:", reader.get_field("general.architecture"))
print("Total Tensors:", len(reader.tensors))
total_params = sum(int(t.shape.prod()) for t in reader.tensors)
print(f"Total Parameters: {total_params:,}")
```
