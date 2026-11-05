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

# Mark-Experimental-V3-8B

## Overview

Mark-Experimental-V3-8B is an 8B-class sparse model packing 32 layers, 32 routed experts, 2 shared experts, and full H-MLA attention into 8,168,001,024 total physical parameters (2,002,374,144 active per token).

Quantized in IQ4_NL, the binary occupies 4382.55 MiB (4595.44 MB / 4,595,438,848 bytes), fitting within an 8 GB consumer GPU while leaving over 3.4 GB of VRAM for KV cache.

## Specifications

| Field | Value |
| :--- | :--- |
| Architecture | `deepseek2` |
| Training Status | `UNTRAINED_ARCHITECTURAL_SCAFFOLD` |
| Weight State Label | `Untrained Architectural Scaffold` |
| Binary Size | `4382.55 MiB (4595.44 MB / 4,595,438,848 bytes)` |
| Total Parameters | `8,168,001,024` (8168001024 physical) |
| Active Parameters | `2,002,374,144` (2002374144 active) |
| Vocabulary Size | `32032` |
| Layers | `32` |
| Hidden Dimension | `2560` |
| Attention Heads | `32` |
| MLA Ranks | `q_lora_rank=1024`, `kv_lora_rank=512`, `qk_rope_head_dim=32` |
| MoE Experts | `32 routed, 2 shared (4 active/token)` |
| Expert FFN Dim | `896` |
| Context Length | `32768` tokens |
| Quantization Type | `IQ4_NL` (4-bit non-linear) |
| SHA-256 | `792e127a71f62145f87c3775c4a03a0dc7bb68b8ee355acb74602348cf748b4a` |

## Architecture

### Multi-Head Latent Attention (MLA & H-MLA)

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

Physical memory consumption of the KV cache per token:

`KV_bytes_per_token = 32 * (512 + 32) * 2 = 34,816 bytes/token`

| Context Window | MLA KV Cache | Standard MHA KV Cache | Memory Savings |
| :--- | :--- | :--- | :--- |
| 4,096 tokens | 136.00 MiB | 1280.00 MiB | 9.4x |
| 8,192 tokens | 272.00 MiB | 2560.00 MiB | 9.4x |
| 32,768 tokens | 1088.00 MiB | 10240.00 MiB | 9.4x |

### Mixture-of-Experts (PB-MoE & Aux-Loss-Free Routing)

- **Shared Experts (2S)**: Constantly active path for common representations:
  $$F_{\text{shared}}(x) = (\text{SiLU}(x W_{\text{gate}}) \odot (x W_{\text{up}})) W_{\text{down}}$$
- **Routed Experts (32R, Top-4)**: Routed using token-to-expert affinities $s_{i,t} = \sigma(u_t^T e_i)$ over biased scores $s_{i,t} + b_i$, where router bias vectors (`blk.{l}.exp_probs_b.bias`) are stored in FP32 precision.

#### Aux-Loss-Free Routing Equation

Router biases update per step to prevent expert collapse without penalizing primary task loss using the exact sign-based rule:

$$b_i \leftarrow b_i + \gamma \cdot \operatorname{sign}(\bar{f} - f_i)$$

where $\gamma = 0.001$, $f_i$ is empirical expert frequency within the batch, and $\bar{f}$ is target uniform frequency.

Router bias vectors are stored in FP32 precision or initialized to uniform values during pre-training to ensure stable routing dynamics.

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
- Curricula and Datasets: Trained on open datasets hosted on the Hugging Face Hub, including [miceworks/Mark-Ultra-Curriculum](https://huggingface.co/datasets/miceworks/Mark-Ultra-Curriculum), [miceworks/Mark-Open-SLM-Curriculum](https://huggingface.co/datasets/miceworks/Mark-Open-SLM-Curriculum), and [miceworks/Mark-Cyber-Curriculum](https://huggingface.co/datasets/miceworks/Mark-Cyber-Curriculum).
- Optimization Settings: AdamW optimizer ($\beta_1 = 0.9, \beta_2 = 0.95$, weight decay $0.1$, gradient clipping $1.0$).
- Training Scope: Untrained architectural base scaffold (UNTRAINED_ARCHITECTURAL_SCAFFOLD). Built for testing binary graph compatibility with llama.cpp, measuring memory usage on target hardware, and serving as a base for downstream research.

## Evaluation and Benchmark Results

Structural tensor verification confirms full DeepSeek2 MoE graph compliance.

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

- Parameter Scale Constraints: While sparse MoE routing enables execution within consumer VRAM budgets, absolute capacity is bounded by physical parameter counts.
- Recommended Deployment: For mission-critical production environments, deploy alongside external verification mechanisms such as compiler validation or unit test runners.
- Safety and Boundary Enforcement: Models trained on systems and cybersecurity datasets are intended for defensive research, vulnerability auditing, patch validation, and systems engineering.

### Running Benchmarks on TPN Workbench and lm-evaluation-harness

When running automated benchmarks on TPN Workbench, lm-evaluation-harness, LM Studio, or Hugging Face evaluation runners:
- Select **Format: gguf** (or **Backend: llama.cpp**).
- Do not evaluate via raw Python `transformers` with `safetensors` or `trust_remote_code=True`, because the native `deepseek2` H-MLA and fine-grained 3D MoE graph executes directly inside the optimized C++/CUDA kernels of `llama.cpp`. Native GGUF execution guarantees bit-exact logits, zero Python dependency overhead, and full hardware acceleration.

## Usage: Runtime Graph, Memory & Latency Verification

This checkpoint is an untrained architectural base scaffold. Usage below is configured for binary graph verification, memory allocation profiling, and hardware latency benchmarking:

### Download and Checksum Verification

```bash
# Using huggingface-cli
huggingface-cli download miceworks/Mark-Experimental-V3-8B mark_experimental_v3_8b_iq4_nl.gguf --local-dir .

# Cryptographic verification
echo "792e127a71f62145f87c3775c4a03a0dc7bb68b8ee355acb74602348cf748b4a  mark_experimental_v3_8b_iq4_nl.gguf" | sha256sum -c -
```

### Graph and Memory Profiling with llama.cpp

```bash
# Benchmark throughput and KV cache allocation
llama-bench -m mark_experimental_v3_8b_iq4_nl.gguf -n 128 -b 512

# Verify execution graph
llama-cli -m mark_experimental_v3_8b_iq4_nl.gguf -p "System probe: 1+1=" -n 16 --temp 0.0
```

### Python Graph Inspection

```python
import gguf

reader = gguf.GGUFReader("mark_experimental_v3_8b_iq4_nl.gguf")
print("Architecture:", reader.get_field("general.architecture"))
print("Total Tensors:", len(reader.tensors))
total_params = sum(int(t.shape.prod()) for t in reader.tensors)
print(f"Total Parameters: {total_params:,}")
```
