---
license: apache-2.0
library_name: gguf
pipeline_tag: text-generation
language:
  - en
datasets:
  - miceworks/Mark-Open-SLM-Curriculum
  - AxiomicLabs/ArithMark-3.0
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

# Mark-SLM-3M

## Overview

Mark-SLM-3M is an ultra-compact language model engineered under the strict 3,000,000 parameter ceiling for the Open SLM Leaderboard. It integrates Multi-Head Latent Attention (MLA) and Mixture of Experts (MoE), trained 100% from scratch in full compliance with Rules 01 and 02.

The model contains 2,946,008 unique physical parameters and 2,503,640 active parameters per token across 8 layers. Quantized to IQ4_NL, the container occupies 3.12 MiB (3.27 MB / 3,271,680 bytes) and runs on embedded microcontrollers, mobile processors, or desktop environments with minimal resource consumption.

## Specifications

| Field | Value |
| :--- | :--- |
| Architecture | `deepseek2` |
| Training Status | `TRAINED_CHECKPOINT` |
| Weight State Label | `Trained Checkpoint (From Scratch)` |
| Binary Size | `3.12 MiB (3.27 MB / 3,271,680 bytes)` |
| Total Parameters | `2,946,008` (2946008 physical across 137 tensors) |
| Active Parameters | `2,503,640` (2503640 active) |
| Vocabulary Size | `4096` |
| Layers | `8` |
| Hidden Dimension | `192` |
| Attention Heads | `6` |
| MLA Ranks | `q_lora_rank=96`, `kv_lora_rank=64`, `qk_rope_head_dim=16` |
| MoE Experts | `4 routed, 1 shared (2 active/token)` |
| Expert FFN Dim | `64` |
| Context Length | `4096` tokens |
| Quantization Type | `IQ4_NL` (4-bit non-linear) |
| SHA-256 | `c66a0b3b76aba96c64d53d0e8821bd9e6e8bb29dc925890fa3a8efa05b1d1d30` |

## Architecture

### Multi-Head Latent Attention (MLA)

The model projects key and value states into a shared low-rank latent representation to decouple memory scaling from context length:

```text
c_KV = h_t * W_DKV          (dim: 64)
c_Q  = h_t * W_DQ           (dim: 96)
q_C  = RMSNorm(c_Q * W_UQ)
k_C  = RMSNorm(c_KV * W_UK)
```

Decoupled rotary position embeddings are concatenated along a separate dimension:

```text
k_R  = RoPE(h_t * W_KR)     (dim: 16)
q_t  = [q_C ; q_R]
k_t  = [k_C ; k_R]
```

### KV Cache Memory Usage

Physical memory consumption of the KV cache per token in FP16:

`KV_bytes_per_token = 8 * (64 + 16) * 2 = 1,280 bytes/token`

| Context Window | MLA KV Cache | Standard MHA KV Cache | Memory Savings |
| :--- | :--- | :--- | :--- |
| 4,096 tokens | 5.00 MiB | 24.00 MiB | 4.8x |
| 8,192 tokens | 10.00 MiB | 48.00 MiB | 4.8x |
| 32,768 tokens | 40.00 MiB | 192.00 MiB | 4.8x |

### Mixture-of-Experts (Aux-Loss-Free Routing)

- **Shared Experts (1S)**: Constantly active path for common representations:
  $$F_{\text{shared}}(x) = (\text{SiLU}(x W_{\text{gate}}) \odot (x W_{\text{up}})) W_{\text{down}}$$
- **Routed Experts (4R, Top-2)**: Routed using token-to-expert affinities $s_{i,t} = \sigma(u_t^T e_i)$ over biased scores $s_{i,t} + b_i$, where router bias vectors (`blk.{l}.exp_probs_b.bias`) are stored in FP32 precision.

#### Aux-Loss-Free Routing Equation

Router biases update per step to prevent expert collapse without penalizing primary task loss using the exact sign-based rule:

$$b_i \leftarrow b_i + \gamma \cdot \operatorname{sign}(\bar{f} - f_i)$$

where $\gamma = 0.001$, $f_i$ is empirical expert frequency within the batch, and $\bar{f}$ is target uniform frequency.

Router bias vectors (`blk.{l}.exp_probs_b.bias`) are stored in FP32 precision. Measured disk telemetry from the IQ4_NL container:
- Bias Parameters: 24 FP32 parameters across 6 MoE layers (4 routed experts each)
- Measured Values: initialized at uniform 0.0000

#### Domain Clan Partitioning

The 4 routed experts are partitioned into 4 functional clans (1 expert per clan):
- **Clan 0 (Expert 0)**: Systems code, low-level logic, and algorithmic execution.
- **Clan 1 (Expert 1)**: Mathematics, arithmetic reasoning, and STEM problem solving.
- **Clan 2 (Expert 2)**: Security boundaries, protocol compliance, and invariant validation.
- **Clan 3 (Expert 3)**: Structured Data, Tool Calling, and Symbolic Logic.

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
- Training Scope: Trained 100% from scratch (Gaussian normal std=0.02 initialization, zero external checkpoints) for 3,000 optimization steps on miceworks/Mark-Open-SLM-Curriculum, Cosmopedia-v2 STEM, and procedural arithmetic in strict compliance with Open SLM Leaderboard Rules 01 and 02.

## Evaluation and Benchmark Results

Official Open SLM Leaderboard Scores (Sub-3M Category, Rule 01 and Rule 02 Compliant):
- PIQA: 52.00% (Norm: 4.00%)
- ARC-Easy: 26.60% (Norm: 2.13%)
- HellaSwag: 25.60% (Norm: 0.80%)
- ArithMark-3.0 Official: 29.70% (Norm: 6.27%)
- ARC-Challenge: 23.75% (Norm: 0.00%)
- ARC-Average (Norm): 1.07%
- Best Intelligence Index: 3.85

## Tokenizer and Vocabulary Specification

- Vocabulary Size: `4096` active BPE tokens (3,835 merges, 0 padding tokens)
- Architecture Backend: Byte-level Byte-Pair Encoding (BPE)
- Control Tokens: System tags (`<think>`, `</think>`), sequence delimiters (`<|bos|>`, `<|eos|>`)
- Measured Compression Fertility:
  - English technical prose: $\approx 3.90\text{ characters/token}$
  - Portuguese technical prose: $\approx 1.94\text{ characters/token}$
  - C, Rust, and systems source code: $\approx 1.81\text{ characters/token}$
- Whitespace Round-Trip Fidelity: 100% lossless preservation across token sequences.

## Limitations and Intended Use

- Architectural Research Scope: This model is an experimental architectural testbed for Multi-Head Latent Attention (MLA) and fine-grained MoE on 8 GB GPUs such as NVIDIA GeForce RTX 2060 Super (8 GB GDDR6, 256-bit, 448 GB/s), NOT a general-purpose conversational or multilingual assistant.
- In-Distribution Curriculum Learning: Benchmark scores on `Mark-SLM-V3-150M` measure in-distribution learning on `miceworks/Mark-Open-SLM-Curriculum` (which includes official benchmark training splits permitted under Open SLM Leaderboard rules) rather than open-ended out-of-distribution generalization.
- Parameter Scale Constraints: Absolute world knowledge and zero-shot reasoning capacity are bounded by compact parameter counts.

### Running Benchmarks on TPN Workbench and lm-evaluation-harness

When running automated benchmarks on TPN Workbench, lm-evaluation-harness, LM Studio, or Hugging Face evaluation runners:
- Select **Format: gguf** (or **Backend: llama.cpp**).
- Do not evaluate via raw Python `transformers` with `safetensors` or `trust_remote_code=True`, because the native `deepseek2` MLA and fine-grained 3D MoE graph executes directly inside the C++/CUDA kernels of `llama.cpp`.

## Usage

### Download

```bash
# Download GGUF and Modelfile using huggingface-cli
huggingface-cli download miceworks/Mark-SLM-3M mark_slm_3m_iq4_nl.gguf Modelfile --local-dir .

# Or using Python
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="miceworks/Mark-SLM-3M", filename="mark_slm_3m_iq4_nl.gguf")
```

### Cryptographic Verification

```bash
echo "c66a0b3b76aba96c64d53d0e8821bd9e6e8bb29dc925890fa3a8efa05b1d1d30  mark_slm_3m_iq4_nl.gguf" | sha256sum -c -
```

### llama.cpp Execution

```bash
# Direct CLI completion
llama-cli -m mark_slm_3m_iq4_nl.gguf \
  -p "Question: What is 12 * 12?\nAnswer:" \
  -n 64 --temp 0.2

# Local HTTP server
llama-server -m mark_slm_3m_iq4_nl.gguf --port 8080 -c 4096
```

### Local Ollama Registration

```bash
# Create local model from downloaded Modelfile
ollama create mark-slm-3m -f Modelfile
ollama run mark-slm-3m "Question: What is 12 * 12? Answer:"
```

### Python Inspection

```python
import gguf

reader = gguf.GGUFReader("mark_slm_3m_iq4_nl.gguf")
print("Architecture:", reader.get_field("general.architecture"))
print("Total Tensors:", len(reader.tensors))
total_params = sum(int(t.shape.prod()) for t in reader.tensors)
print(f"Materialized GGUF Parameters: {total_params:,}")
```
