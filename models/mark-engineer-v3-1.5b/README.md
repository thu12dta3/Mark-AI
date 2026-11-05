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

# Mark-Engineer-V3-1.5B

## Overview

Mark-Engineer-V3-1.5B is an experimental narrow checkpoint with 1,520,496,960 total physical parameters and 405.73M active parameters per token across 18 layers. It uses 32 routed experts and 2 shared experts.

Quantized in IQ4_NL, the container occupies 816.95 MiB (856.63 MB / 856,632,832 bytes). Note: This is an experimental checkpoint trained briefly with a 4,311-token base vocabulary; outside its narrow systems domain, greedy inference produces incoherent fragments.

## Specifications

| Field | Value |
| :--- | :--- |
| Architecture | `deepseek2` |
| Training Status | `EXPERIMENTAL_NARROW_CHECKPOINT` |
| Weight State Label | `Experimental Narrow Checkpoint` |
| Binary Size | `816.95 MiB (856.63 MB / 856,632,832 bytes)` |
| Total Parameters | `1,520,496,960` (1520496960 physical across 327 tensors) |
| Active Parameters | `405,729,600` (405729600 active) |
| Vocabulary Size | `32032` |
| Layers | `18` |
| Hidden Dimension | `1280` |
| Attention Heads | `20` |
| MLA Ranks | `q_lora_rank=512`, `kv_lora_rank=256`, `qk_rope_head_dim=64` |
| MoE Experts | `32 routed, 2 shared (4 active/token)` |
| Expert FFN Dim | `576` |
| Context Length | `32768` tokens |
| Quantization Type | `IQ4_NL` (4-bit non-linear) |
| SHA-256 | `24093fd96a22c7b4b19e85df2fb0d4e1757361d6f58baa4f36ad0285fa29ac68` |

## Architecture

### Multi-Head Latent Attention (MLA)

The model projects key and value states into a shared low-rank latent representation to decouple memory scaling from context length:

```text
c_KV = h_t * W_DKV          (dim: 256)
c_Q  = h_t * W_DQ           (dim: 512)
q_C  = RMSNorm(c_Q * W_UQ)
k_C  = RMSNorm(c_KV * W_UK)
```

Decoupled rotary position embeddings are concatenated along a separate dimension:

```text
k_R  = RoPE(h_t * W_KR)     (dim: 64)
q_t  = [q_C ; q_R]
k_t  = [k_C ; k_R]
```

### KV Cache Memory Usage

Physical memory consumption of the KV cache per token in FP16:

`KV_bytes_per_token = 18 * (256 + 64) * 2 = 11,520 bytes/token`

| Context Window | MLA KV Cache | Standard MHA KV Cache | Memory Savings |
| :--- | :--- | :--- | :--- |
| 4,096 tokens | 45.00 MiB | 360.00 MiB | 8.0x |
| 8,192 tokens | 90.00 MiB | 720.00 MiB | 8.0x |
| 32,768 tokens | 360.00 MiB | 2880.00 MiB | 8.0x |

### Mixture-of-Experts (Aux-Loss-Free Routing)

- **Shared Experts (2S)**: Constantly active path for common representations:
  $$F_{\text{shared}}(x) = (\text{SiLU}(x W_{\text{gate}}) \odot (x W_{\text{up}})) W_{\text{down}}$$
- **Routed Experts (32R, Top-4)**: Routed using token-to-expert affinities $s_{i,t} = \sigma(u_t^T e_i)$ over biased scores $s_{i,t} + b_i$, where router bias vectors (`blk.{l}.exp_probs_b.bias`) are stored in FP32 precision.

#### Aux-Loss-Free Routing Equation

Router biases update per step to prevent expert collapse without penalizing primary task loss using the exact sign-based rule:

$$b_i \leftarrow b_i + \gamma \cdot \operatorname{sign}(\bar{f} - f_i)$$

where $\gamma = 0.001$, $f_i$ is empirical expert frequency within the batch, and $\bar{f}$ is target uniform frequency.

Router bias vectors (`blk.{l}.exp_probs_b.bias`) are stored in FP32 precision. Measured disk telemetry from the IQ4_NL container:
- Bias Parameters: 576 FP32 parameters across 18 MoE layers (32 routed experts each)
- Measured Values: min = 0.8760, mean = 2.2283, max = 2.9219

#### Domain Clan Partitioning

The 32 routed experts are partitioned into 4 functional clans (8 experts per clan):
- **Clan 0 (Experts 0-7)**: Systems code, kernel internals, concurrency, and low-level engineering.
- **Clan 1 (Experts 8-15)**: Mathematics, formal reasoning, algorithmic logic, and STEM.
- **Clan 2 (Experts 16-23)**: Cybersecurity, vulnerability assessment, protocol enforcement, and defense.
- **Clan 3 (Experts 24-31)**: Structured Data, Tool Calling, and Symbolic Logic.

### Experimental PyTorch Research Extensions (Not Active in llama.cpp GGUF Runtime)

The following architectural mechanisms are research extensions implemented in the PyTorch reference codebase (`src/model/mark_v3_arch.py`) and are not active in the compiled C++/CUDA feed-forward DAG of the `llama.cpp` GGUF runtime:

1. **Latent Recursive Depth (LRD)**:
   In PyTorch, intermediate core layers (layers $s..e-1$) execute across $L_{\text{core}} = 2$ recurrent iterations using residual anchoring:
   $$h^{(r+1)}_l = \text{Block}_l(h^{(r)}_l + 0.5 \cdot h_{\text{stem}})$$
   In `llama.cpp`, GGUF execution is strictly feed-forward (single sequential pass through layers $0..N-1$).

2. **PB-MoE Soft-Pinning Constraints**:
   During PyTorch curriculum training, clan logit constraints reserve routing slots for domain specialists. In `llama.cpp`, routing executes via standard Top-K selection over the learned router bias affinities:
   $$s_{i,t} = \sigma(u_t^T e_i) + b_i$$

## Training Pipeline and Reproducibility

- Training Code: Official pipelines and architecture implementations are maintained in the [Mark-AI scripts directory](https://github.com/thu12dta3/Mark-AI/tree/main/scripts).
- Curricula and Datasets: Trained on open datasets hosted on the Hugging Face Hub, including [miceworks/Mark-Open-SLM-Curriculum](https://huggingface.co/datasets/miceworks/Mark-Open-SLM-Curriculum), [miceworks/Mark-Cyber-Curriculum](https://huggingface.co/datasets/miceworks/Mark-Cyber-Curriculum), and [miceworks/Mark-Ultra-Curriculum](https://huggingface.co/datasets/miceworks/Mark-Ultra-Curriculum).
- Optimization Settings: AdamW optimizer ($\beta_1 = 0.9, \beta_2 = 0.95$, weight decay $0.1$, gradient clipping $1.0$).
- Training Scope: Experimental narrow checkpoint (EXPERIMENTAL_NARROW_CHECKPOINT). Trained briefly on systems engineering stubs from miceworks/Mark-Ultra-Curriculum. Unverified claims regarding upstream training priors and synthetic benchmark evaluations have been formally retracted.

## Evaluation and Benchmark Results

Experimental checkpoint; does not generalize to open-ended coding benchmarks or general conversation. Usage is intended for architecture inspection and narrow experimental research.

## Tokenizer and Vocabulary Specification

- Vocabulary Size: `32032` tensor rows (4,311 active BPE tokens + 27,721 reserved padding slots)
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
echo "24093fd96a22c7b4b19e85df2fb0d4e1757361d6f58baa4f36ad0285fa29ac68  mark_engineer_v3_1_5b_iq4_nl.gguf" | sha256sum -c -
```

### Execution with llama.cpp

```bash
llama-cli -m mark_engineer_v3_1_5b_iq4_nl.gguf \
  -p "Systems engineering stub: Implement a POSIX mutex initialization routine.\nCode:" \
  -n 128 --temp 0.2
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
