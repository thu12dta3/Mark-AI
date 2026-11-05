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

# Mark-SLM-V2-302M

## Overview

Mark-SLM-V2-302M is an experimental narrow checkpoint in the 302M family naming convention. It contains 329,132,160 total physical parameters and activates 155.72M parameters per token across 16 layers. It features 16 routed experts and 1 shared expert (2 active routed experts per token).

Quantized in IQ4_NL, the binary occupies 177.76 MiB (186.39 MB / 186,394,944 bytes) and runs on 8 GB consumer GPUs with minimal VRAM consumption.

## Specifications

| Field | Value |
| :--- | :--- |
| Architecture | `deepseek2` |
| Training Status | `EXPERIMENTAL_NARROW_CHECKPOINT` |
| Weight State Label | `Experimental Narrow Checkpoint` |
| Binary Size | `177.76 MiB (186.39 MB / 186,394,944 bytes)` |
| Total Parameters | `329,132,000` (329132000 physical across 281 tensors) |
| Active Parameters | `155,723,744` (155723744 active) |
| Vocabulary Size | `49152` |
| Layers | `16` |
| Hidden Dimension | `768` |
| Attention Heads | `12` |
| MLA Ranks | `q_lora_rank=384`, `kv_lora_rank=192`, `qk_rope_head_dim=32` |
| MoE Experts | `16 routed, 1 shared (2 active/token)` |
| Expert FFN Dim | `384` |
| Context Length | `4096` tokens |
| Quantization Type | `IQ4_NL` (4-bit non-linear) |
| SHA-256 | `34a3e0e39cd02cd6a12345b9558ea094877530ef3bde52e64552f20ca48ce38e` |

## Architecture

### Multi-Head Latent Attention (MLA)

The model projects key and value states into a shared low-rank latent representation to decouple memory scaling from context length:

```text
c_KV = h_t * W_DKV          (dim: 192)
c_Q  = h_t * W_DQ           (dim: 384)
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

`KV_bytes_per_token = 16 * (192 + 32) * 2 = 7,168 bytes/token`

| Context Window | MLA KV Cache | Standard MHA KV Cache | Memory Savings |
| :--- | :--- | :--- | :--- |
| 4,096 tokens | 28.00 MiB | 192.00 MiB | 6.9x |
| 8,192 tokens | 56.00 MiB | 384.00 MiB | 6.9x |
| 32,768 tokens | 224.00 MiB | 1536.00 MiB | 6.9x |

### Mixture-of-Experts (Aux-Loss-Free Routing)

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
- Curricula and Datasets: Trained on open datasets hosted on the Hugging Face Hub, including [miceworks/Mark-Open-SLM-Curriculum](https://huggingface.co/datasets/miceworks/Mark-Open-SLM-Curriculum), [miceworks/Mark-Cyber-Curriculum](https://huggingface.co/datasets/miceworks/Mark-Cyber-Curriculum), and [miceworks/Mark-Ultra-Curriculum](https://huggingface.co/datasets/miceworks/Mark-Ultra-Curriculum).
- Optimization Settings: AdamW optimizer ($\beta_1 = 0.9, \beta_2 = 0.95$, weight decay $0.1$, gradient clipping $1.0$).
- Training Scope: Experimental narrow checkpoint (EXPERIMENTAL_NARROW_CHECKPOINT). Trained for 10,000 steps on miceworks/Mark-Ultra-Curriculum. The final training loss reached 0.263975, reflecting multi-epoch memorization on the small curriculum testbed dataset rather than broad open-domain generalization.

## Evaluation and Benchmark Results

Achieved a final validation loss of 0.263975 across 10,000 training steps on the narrow miceworks/Mark-Ultra-Curriculum distribution (memorization artifact).

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

## Usage: Narrow Domain and Systems Engineering Verification

This checkpoint is an experimental narrow checkpoint trained on specific low-level or systems engineering distributions. Outside its narrow domain, greedy inference produces fragmented outputs.

### Cryptographic Verification

```bash
echo "34a3e0e39cd02cd6a12345b9558ea094877530ef3bde52e64552f20ca48ce38e  mark_slm_v2_302m_iq4_nl.gguf" | sha256sum -c -
```

### Execution with llama.cpp

```bash
llama-cli -m mark_slm_v2_302m_iq4_nl.gguf \
  -p "Systems engineering stub: Implement a POSIX mutex initialization routine.\nCode:" \
  -n 128 --temp 0.2
```

### Python Inspection

```python
import gguf

reader = gguf.GGUFReader("mark_slm_v2_302m_iq4_nl.gguf")
print("Architecture:", reader.get_field("general.architecture"))
print("Total Tensors:", len(reader.tensors))
total_params = sum(int(t.shape.prod()) for t in reader.tensors)
print(f"Total Parameters: {total_params:,}")
```
