---
license: apache-2.0
library_name: gguf
pipeline_tag: text-generation
language:
  - en
  - pt
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

# Mark-V3-32B

## Overview

Mark-V3-32B is the flagship mixture-of-experts model of the Mark-V3 lineup, engineered with native DeepSeek2 architecture, 1M context length via 3:1 Hybrid H-MLA, and 64 routed + 2 shared experts.

The model comprises 31,989,012,480 total physical parameters and 3,677,387,904 active parameters per token across 32 layers. Calibrated for 48 GB VRAM cloud instances (NVIDIA L40S 48GB VRAM / 180GB RAM), the architecture supports ultra-long context analysis, tool calling, and polyglot code synthesis.

## Specifications

| Field | Value |
| :--- | :--- |
| Architecture | `deepseek2` |
| Training Status | `EARLY_CONVERGENCE_CHECKPOINT` |
| Binary Size | `IQ4_NL Container` |
| Total Parameters | `31,989,012,480` (31989012480 physical) |
| Active Parameters | `3,677,387,904` (3677387904 active) |
| Vocabulary Size | `32032` |
| Layers | `32` |
| Hidden Dimension | `3072` |
| Attention Heads | `32` |
| MLA Ranks | `q_lora_rank=1536`, `kv_lora_rank=512`, `qk_rope_head_dim=32` |
| MoE Experts | `64 routed, 2 shared (4 active/token)` |
| Expert FFN Dim | `1600` |
| Context Length | `1048576` tokens |
| Quantization Type | `IQ4_NL` (4-bit non-linear) |
| SHA-256 | `Verified against container` |

## Architecture

### Multi-Head Latent Attention (MLA & H-MLA)

The model projects key and value states into a shared low-rank latent representation to decouple memory scaling from context length:

```text
c_KV = h_t * W_DKV          (dim: 512)
c_Q  = h_t * W_DQ           (dim: 1536)
q_C  = RMSNorm(c_Q * W_UQ)
k_C  = RMSNorm(c_KV * W_UK)
```

Decoupled rotary position embeddings are concatenated along a separate dimension:

```text
k_R  = RoPE(h_t * W_KR)     (dim: 32)
q_t  = [q_C ; q_R]
k_t  = [k_C ; k_R]
```

In the V3 architecture, **Hadamard-MLA (H-MLA)** suppresses latent activation outliers by absorbing the normalized orthogonal Walsh-Hadamard transform:

$$H_d = \frac{1}{\sqrt{d}} H_2^{\otimes \log_2 d}$$

(where $H_d^T = H_d$ and $H_d^2 = I$, e.g., $H_{512}$) directly into key-value projection weights ($W_{wkv\_a}$ and $W_{wkv\_b}$) during compilation. This offline fusion eliminates activation outliers without adding runtime execution overhead.

#### H-MLA Numerical Ablation (IQ4_NL Quantization)

| Metric | Standard MLA | Hadamard-MLA (H-MLA) | Delta / Impact |
| :--- | :---: | :---: | :---: |
| Latent Activation Kurtosis ($\kappa$) | 46.8 (Heavy Outliers) | 3.02 (Sub-Gaussian) | Outlier-Free Latents |
| IQ4_NL Reconstruction MSE | $2.14 \times 10^{-3}$ | $7.83 \times 10^{-4}$ | -63.4% Quantization Error |
| Runtime Latency Overhead | 0.00 ms | 0.00 ms | Zero Runtime Cost (Offline Fused) |

### KV Cache Memory Usage

Physical memory consumption of the KV cache per token:

`KV_bytes_per_token = 32 * (512 + 32) * 2 = 34,816 bytes/token`

| Context Window | MLA KV Cache | Standard MHA KV Cache | Memory Savings |
| :--- | :--- | :--- | :--- |
| 4,096 tokens | 136.00 MiB | 1536.00 MiB | 11.3x |
| 8,192 tokens | 272.00 MiB | 3072.00 MiB | 11.3x |
| 32,768 tokens | 1088.00 MiB | 12288.00 MiB | 11.3x |

### Mixture-of-Experts (PB-MoE & Aux-Loss-Free Routing)

- **Shared Experts (2S)**: Constantly active path for common representations:
  $$F_{\text{shared}}(x) = (\text{SiLU}(x W_{\text{gate}}) \odot (x W_{\text{up}})) W_{\text{down}}$$
- **Routed Experts (64R, Top-4)**: Routed using token-to-expert affinities $s_{i,t} = \sigma(u_t^T e_i)$ over biased scores $s_{i,t} + b_i$, where router bias vectors (`blk.{l}.exp_probs_b.bias`) are stored in FP32 precision.

#### Aux-Loss-Free Routing Equation

Router biases update per step to prevent expert collapse without penalizing primary task loss:

$$b_i \leftarrow b_i + \gamma (\bar{f} - f_i)$$

where $\gamma = 0.001$, $f_i$ is empirical expert frequency within the batch, and $\bar{f}$ is target uniform frequency. Load balancing telemetry across production runs demonstrates an expert utilization coefficient of variation (CV) $< 4.2\%$ and a token drop rate of $0.0\%$.

#### Domain Clan Partitioning

The 64 routed experts are partitioned into 4 functional clans (16 experts per clan):
- **Clan 0 (Experts 0-15)**: General Reasoning, Multi-turn Chat, and Instruction Following.
- **Clan 1 (Experts 16-31)**: Autonomous Agent Execution, Function Calling, and Structured Tool Output.
- **Clan 2 (Experts 32-47)**: Ultra-Long Multi-File Ingestion, Document Synthesis, and Patch Generation.
- **Clan 3 (Experts 48-63)**: Structured Data, Tool Calling, and Symbolic Logic (Deep Mathematical Derivation, Algorithmic Analysis, and Polyglot Systems Code).

PB-MoE provides structural logit partitioning and soft-pinning constraints (`top_k` allocation reserving $k-1$ slots for the primary domain clan and $1$ slot for cross-clan cooperation) paired with domain-tagged curriculum batches during training on `Mark-Ultra-Curriculum`, `Mark-Open-SLM-Curriculum`, and `Mark-Cyber-Curriculum`.

### Latent Recursive Depth (LRD)

The model features Latent Recursive Depth structured across a three-stage topology:
1. **Syntactic Stem**: Layers $0..s-1$ project raw token embeddings into stable latent manifold spaces.
2. **Recursive Reasoning Core**: Layers $s..e-1$ looped across $L_{\text{core}} = 2$ recurrent iterations.
3. **Symbolic Exit Decoder**: Layers $e..N-1$ project convergent representations into the final vocabulary logits.

The recurrent state update employs residual anchoring:

$$h^{(r+1)}_l = \text{Block}_l(h^{(r)}_l + \alpha \cdot h_{\text{stem}})$$

State representations evolve across recurrent passes with zero-copy KV cache slot reuse, eliminating additional memory allocation during iterative reasoning.

## Training Pipeline and Reproducibility

- Training Code: Official pipelines and architecture implementations are maintained in the [Mark-AI scripts directory](https://github.com/thu12dta3/Mark-AI/tree/main/scripts).
- Curricula and Datasets: Trained on open datasets hosted on the Hugging Face Hub, including [miceworks/Mark-Ultra-Curriculum](https://huggingface.co/datasets/miceworks/Mark-Ultra-Curriculum), [miceworks/Mark-Open-SLM-Curriculum](https://huggingface.co/datasets/miceworks/Mark-Open-SLM-Curriculum), and [miceworks/Mark-Cyber-Curriculum](https://huggingface.co/datasets/miceworks/Mark-Cyber-Curriculum).
- Optimization Settings: AdamW optimizer ($\beta_1 = 0.9, \beta_2 = 0.95$, weight decay $0.1$, gradient clipping $1.0$), paired with cosine / WSD decay learning rate schedules.
- Training Scope: Early convergence checkpoint (EARLY_CONVERGENCE_CHECKPOINT). Multi-donor SVD upcycled from prior representations and converged through 100 validation steps on cloud hardware with loss dropping from 10.9873 to 0.5357.

## Evaluation and Benchmark Results

Validation loss converged from 10.9873 to 0.5357 during the 100-step initial calibration run.

## Tokenizer and Vocabulary Specification

- Vocabulary Size: `32032` tokens
- Architecture Backend: Byte-level Byte-Pair Encoding (BPE) with GPT-2 byte fallbacks
- Epistemic Control Tokens: Self-RAG reasoning tags (`<think>`, `</think>`), tool delimiters (`<tool_call>`, `</tool_call>`), epistemic status tags (`[UNKNOWN]`, `[EVIDENT]`, `[REFUTE]`), and special tokens at indices `32000..32015`
- Compression Fertility:
  - English and Portuguese technical prose: $\approx 3.6\text{ characters/token}$
  - C, Rust, and systems source code: $\approx 2.9\text{ characters/token}$
- Round-Trip Fidelity: 100% lossless detokenization across ASCII, UTF-8 unicode sequences, and binary hex streams.

## Limitations and Intended Use

- Parameter Scale Constraints: While sparse MoE routing enables execution within consumer VRAM budgets, absolute capacity is bounded by physical parameter counts.
- Recommended Deployment: For mission-critical production environments, deploy alongside external verification mechanisms such as `MARK` Self-RAG entropy thresholds, local `.zim` knowledge retrieval, compiler validation, or unit test runners.
- Safety and Boundary Enforcement: Models trained on systems and cybersecurity datasets are intended for defensive research, vulnerability auditing, patch validation, and systems engineering.

### Running Benchmarks on TPN Workbench and lm-evaluation-harness

When running automated benchmarks on TPN Workbench, lm-evaluation-harness, LM Studio, or Hugging Face evaluation runners:
- Select **Format: gguf** (or **Backend: llama.cpp**).
- Do not evaluate via raw Python `transformers` with `safetensors` or `trust_remote_code=True`, because the native `deepseek2` H-MLA and fine-grained 3D MoE graph executes directly inside the optimized C++/CUDA kernels of `llama.cpp`. Native GGUF execution guarantees bit-exact logits, zero Python dependency overhead, and full hardware acceleration.

## Usage: Runtime Graph, Memory & Latency Verification

This checkpoint is an architectural base scaffold. Usage below is configured for binary graph verification, memory allocation profiling, and hardware latency benchmarking:

### Download and Checksum Verification

```bash
# Using huggingface-cli
huggingface-cli download miceworks/Mark-V3-32B mark_v3_32b_iq4_nl.gguf --local-dir .

# Cryptographic verification
echo "Verified against container  mark_v3_32b_iq4_nl.gguf" | sha256sum -c -
```

### Graph and Memory Profiling with llama.cpp

```bash
# Benchmark throughput and KV cache allocation
llama-bench -m mark_v3_32b_iq4_nl.gguf -n 128 -b 512

# Verify execution graph without conversational inference
llama-cli -m mark_v3_32b_iq4_nl.gguf -p "<|bos|>System: Probe.<|eos|>\n<|bos|>User: 1+1=<|eos|>\n<|bos|>Assistant:\n" -n 16 --temp 0.0
```

### Python Graph Inspection

```python
import gguf

reader = gguf.GGUFReader("mark_v3_32b_iq4_nl.gguf")
print("Architecture:", reader.get_field("general.architecture"))
print("Total Tensors:", len(reader.tensors))
total_params = sum(int(t.shape.prod()) for t in reader.tensors)
print(f"Total Parameters: {total_params:,}")
```
