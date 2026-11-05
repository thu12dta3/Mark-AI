---
license: apache-2.0
library_name: gguf
pipeline_tag: text-generation
language:
  - en
datasets:
  - miceworks/Mark-Open-SLM-Curriculum
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

# Mark-SLM-V3-150M

## Overview

Mark-SLM-V3-150M is a compact edge model engineered strictly under the 150M parameter ceiling (Rule 01 of Open SLM Leaderboard). By using tied word embeddings (reusing token_embd.weight as the language model output projection), the GGUF file contains 272 tensors and exactly 146,899,248 parameters (54.00M active per token across 15 layers).

Quantized in IQ4_NL, the container occupies 90.37 MiB (94.76 MB / 94,764,576 bytes), making it suitable for low-power edge microcontrollers, mobile devices, and embedded systems.

## Specifications

| Field | Value |
| :--- | :--- |
| Architecture | `deepseek2` |
| Training Status | `TRAINED_CHECKPOINT` |
| Weight State Label | `Trained Checkpoint (Model Soup 50/50)` |
| Binary Size | `90.37 MiB (94.76 MB / 94,764,576 bytes)` |
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
| Quantization Type | `IQ4_NL` (4-bit non-linear) |
| SHA-256 | `4bbfe2dc3fc7a6ac895d56e30352bda0de1322f9826afa28cafe7d24b0e99632` |

## Architecture

### Multi-Head Latent Attention (MLA & H-MLA)

The model projects key and value states into a shared low-rank latent representation to decouple memory scaling from context length:

```text
c_KV = h_t * W_DKV          (dim: 128)
c_Q  = h_t * W_DQ           (dim: 256)
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

Physical memory consumption of the KV cache per token:

`KV_bytes_per_token = 15 * (128 + 64) * 2 = 5,760 bytes/token`

| Context Window | MLA KV Cache | Standard MHA KV Cache | Memory Savings |
| :--- | :--- | :--- | :--- |
| 4,096 tokens | 22.50 MiB | 135.00 MiB | 6.0x |
| 8,192 tokens | 45.00 MiB | 270.00 MiB | 6.0x |
| 32,768 tokens | 180.00 MiB | 1080.00 MiB | 6.0x |

### Mixture-of-Experts (PB-MoE & Aux-Loss-Free Routing)

- **Shared Experts (1S)**: Constantly active path for common representations:
  $$F_{\text{shared}}(x) = (\text{SiLU}(x W_{\text{gate}}) \odot (x W_{\text{up}})) W_{\text{down}}$$
- **Routed Experts (16R, Top-2)**: Routed using token-to-expert affinities $s_{i,t} = \sigma(u_t^T e_i)$ over biased scores $s_{i,t} + b_i$, where router bias vectors (`blk.{l}.exp_probs_b.bias`) are stored in FP32 precision.

#### Aux-Loss-Free Routing Equation

Router biases update per step to prevent expert collapse without penalizing primary task loss using the exact sign-based rule:

$$b_i \leftarrow b_i + \gamma \cdot \operatorname{sign}(\bar{f} - f_i)$$

where $\gamma = 0.001$, $f_i$ is empirical expert frequency within the batch, and $\bar{f}$ is target uniform frequency.

Router bias vectors (`blk.{l}.exp_probs_b.bias`) are stored in FP32 precision. Measured disk telemetry from the IQ4_NL container:
- Bias Parameters: 240 FP32 parameters across 15 MoE layers (16 routed experts each)
- Measured Values: min = 0.8677, mean = 1.3500, max = 2.0547

#### Domain Clan Partitioning

The 16 routed experts are partitioned into 4 functional clans (4 experts per clan):
- **Clan 0 (Experts 0-3)**: Systems code, kernel semantics, and concurrency.
- **Clan 1 (Experts 4-7)**: Mathematics, formal logic, and STEM reasoning.
- **Clan 2 (Experts 8-11)**: Security boundaries, protocols, and validation.
- **Clan 3 (Experts 12-15)**: Structured Data, Tool Calling, and Symbolic Logic.

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
- Curricula and Datasets: Trained on open datasets hosted on the Hugging Face Hub, including [miceworks/Mark-Ultra-Curriculum](https://huggingface.co/datasets/miceworks/Mark-Ultra-Curriculum), [miceworks/Mark-Open-SLM-Curriculum](https://huggingface.co/datasets/miceworks/Mark-Open-SLM-Curriculum), and [miceworks/Mark-Cyber-Curriculum](https://huggingface.co/datasets/miceworks/Mark-Cyber-Curriculum).
- Optimization Settings: AdamW optimizer ($\beta_1 = 0.9, \beta_2 = 0.95$, weight decay $0.1$, gradient clipping $1.0$).
- Training Scope: Trained model checkpoint (TRAINED_CHECKPOINT). The model was trained through 25,000 optimization steps of pre-training followed by 4,000 steps of Phase 2 Cooldown on miceworks/Mark-Open-SLM-Curriculum, consolidated via 50/50 Model Soup (Weight Averaging). Note: The training dataset miceworks/Mark-Open-SLM-Curriculum contains benchmark training splits (in-distribution learning). Evaluated on the official Open SLM Leaderboard benchmark suite (Rule 01 and Rule 02 compliant), achieving an official AxiomicLabs Intelligence Index of 16.36, merged in PR #291 (https://github.com/AxiomicLabs/open-slm-leaderboard/pull/291).

## Evaluation and Benchmark Results

Official Open SLM Leaderboard Scores (PR #291 merged, Rule 01 and Rule 02 compliant):
- PIQA: 62.50% (Norm: 25.00%)
- ARC-Easy: 40.80% (Norm: 21.07%)
- HellaSwag: 37.10% (Norm: 16.13%)
- ArithMark-3.0 Official: 33.90% (Norm: 11.87%)
- ARC-Challenge: 25.50% (Norm: 0.67%)
- ARC-Average (Norm): 10.87%
- AxiomicLabs Intelligence Index: 16.36

Checkpoint Revision Changelog:
- Revision 1 (Step 25,000 Base Checkpoint): Intelligence Index 10.04
- Revision 2 (Step 4,000 Cooldown Checkpoint): Intelligence Index 14.35
- Revision 3 (Model Soup 50/50 Final): Intelligence Index 16.36 (Official submission merged in PR #291)

## Tokenizer and Vocabulary Specification

- Vocabulary Size: `32032` active BPE tokens (58,980 merges, 0 padding tokens)
- Architecture Backend: Byte-level Byte-Pair Encoding (BPE) with GPT-2 byte fallbacks
- Epistemic Control Tokens: Self-RAG reasoning tags (`<think>`, `</think>`), tool delimiters (`<tool_call>`, `</tool_call>`), epistemic reflection tags (`[RETRIEVE]`, `[/RETRIEVE]`, `[EVALUATE]`, `[ACCEPT]`, `[REJECT]`)
- Measured Compression Fertility:
  - English technical prose: $\approx 5.85\text{ characters/token}$
  - Portuguese technical prose: $\approx 3.17\text{ characters/token}$
  - C, Rust, and systems source code: $\approx 2.23\text{ characters/token}$
- Whitespace Round-Trip Fidelity: 100% lossless preservation of leading spaces, indentation tabs, and byte streams.

## Limitations and Intended Use

- Parameter Scale Constraints: While sparse MoE routing enables execution within consumer VRAM budgets, absolute capacity is bounded by physical parameter counts.
- Recommended Deployment: For mission-critical production environments, deploy alongside external verification mechanisms such as compiler validation or unit test runners.
- Safety and Boundary Enforcement: Models trained on systems and cybersecurity datasets are intended for defensive research, vulnerability auditing, patch validation, and systems engineering.

### Running Benchmarks on TPN Workbench and lm-evaluation-harness

When running automated benchmarks on TPN Workbench, lm-evaluation-harness, LM Studio, or Hugging Face evaluation runners:
- Select **Format: gguf** (or **Backend: llama.cpp**).
- Do not evaluate via raw Python `transformers` with `safetensors` or `trust_remote_code=True`, because the native `deepseek2` H-MLA and fine-grained 3D MoE graph executes directly inside the optimized C++/CUDA kernels of `llama.cpp`. Native GGUF execution guarantees bit-exact logits, zero Python dependency overhead, and full hardware acceleration.

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
echo "4bbfe2dc3fc7a6ac895d56e30352bda0de1322f9826afa28cafe7d24b0e99632  mark_slm_v3_150m_iq4_nl.gguf" | sha256sum -c -
```

### llama.cpp

```bash
# CLI text completion
llama-cli -m mark_slm_v3_150m_iq4_nl.gguf \
  -p "Question: What is 12 * 12?\nAnswer:" \
  -n 128 --temp 0.2

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
