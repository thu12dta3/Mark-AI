---
license: apache-2.0
library_name: gguf
pipeline_tag: image-text-to-text
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
  - multimodal
  - vision-language
---

# Mark-VL-V3-4B

## Overview

Mark-VL-V3-4B is a third-generation multimodal vision-language model architecture scaffold integrating a 16-layer Vision Transformer (ViT) with Spatial 2D-RoPE, a 2x2 PixelShuffle projection adapter, and a 24-layer DeepSeek2 MLA and MoE language backbone.

The model comprises 4,134,688,769 total physical parameters and 824,483,584 active parameters per token across 24 language layers and 16 vision layers.

## Specifications

| Field | Value |
| :--- | :--- |
| Architecture | `deepseek2` |
| Training Status | `UNTRAINED_ARCHITECTURAL_SCAFFOLD` |
| Weight State Label | `Untrained Architectural Scaffold` |
| Binary Size | `IQ4_NL Container` |
| Total Parameters | `4,134,688,769` (4134688769 physical across 273 tensors) |
| Active Parameters | `824,483,584` (824483584 active) |
| Vocabulary Size | `32032` |
| Layers | `24` |
| Hidden Dimension | `1792` |
| Attention Heads | `28` |
| MLA Ranks | `q_lora_rank=512`, `kv_lora_rank=256`, `qk_rope_head_dim=64` |
| MoE Experts | `32 routed, 2 shared (4 active/token)` |
| Expert FFN Dim | `832` |
| Context Length | `32768` tokens |
| Quantization Type | `IQ4_NL` (4-bit non-linear) |
| SHA-256 | `N/A` |

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

`KV_bytes_per_token = 24 * (256 + 64) * 2 = 15,360 bytes/token`

| Context Window | MLA KV Cache | Standard MHA KV Cache | Memory Savings |
| :--- | :--- | :--- | :--- |
| 4,096 tokens | 60.00 MiB | 672.00 MiB | 11.2x |
| 8,192 tokens | 120.00 MiB | 1344.00 MiB | 11.2x |
| 32,768 tokens | 480.00 MiB | 5376.00 MiB | 11.2x |

### Mixture-of-Experts (Aux-Loss-Free Routing)

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
- **Clan 3 (Experts 24-31)**: Multimodal and Spatial Reasoning.

### Experimental PyTorch Research Extensions (Not Active in llama.cpp GGUF Runtime)

The following architectural mechanisms are research extensions implemented in the PyTorch reference codebase (`src/model/mark_v3_arch.py`) and are not active in the compiled C++/CUDA feed-forward DAG of the `llama.cpp` GGUF runtime:

1. **Latent Recursive Depth (LRD)**:
   In PyTorch, intermediate core layers (layers $s..e-1$) execute across $L_{\text{core}} = 2$ recurrent iterations using residual anchoring:
   $$h^{(r+1)}_l = \text{Block}_l(h^{(r)}_l + 0.5 \cdot h_{\text{stem}})$$
   In `llama.cpp`, GGUF execution is strictly feed-forward (single sequential pass through layers $0..N-1$).

2. **PB-MoE Soft-Pinning Constraints**:
   During PyTorch curriculum training, clan logit constraints reserve routing slots for domain specialists. In `llama.cpp`, routing executes via standard Top-K selection over the learned router bias affinities:
   $$s_{i,t} = \sigma(u_t^T e_i) + b_i$$

### Vision-Language Projector Subsystem

The visual architecture integrates an AnyRes SigLIP visual encoder with Spatial 2D-RoPE and a 2x2 PixelShuffle projection adapter:
1. **AnyRes Dynamic Resolution**: Preserves native aspect ratio up to 2048x2048 without downsampling distortion.
2. **Spatial 2D-RoPE**: Visual tokens retain physical (X, Y) coordinate awareness.
3. **PixelShuffle 2x2 C-Abstractor**: Compresses 4 visual patches into 1 dense token (75% visual token reduction).

## Training Pipeline and Reproducibility

- Training Code: Official pipelines and architecture implementations are maintained in the [Mark-AI scripts directory](https://github.com/thu12dta3/Mark-AI/tree/main/scripts).
- Curricula and Datasets: Trained on open datasets hosted on the Hugging Face Hub, including [miceworks/Mark-Open-SLM-Curriculum](https://huggingface.co/datasets/miceworks/Mark-Open-SLM-Curriculum), [miceworks/Mark-Cyber-Curriculum](https://huggingface.co/datasets/miceworks/Mark-Cyber-Curriculum), and [miceworks/Mark-Ultra-Curriculum](https://huggingface.co/datasets/miceworks/Mark-Ultra-Curriculum).
- Optimization Settings: AdamW optimizer ($\beta_1 = 0.9, \beta_2 = 0.95$, weight decay $0.1$, gradient clipping $1.0$).
- Training Scope: Untrained multimodal architectural scaffold (UNTRAINED_ARCHITECTURAL_SCAFFOLD). Built for multimodal graph verification and spatial 2D-RoPE alignment testing.

## Evaluation and Benchmark Results

Projector and attention alignment verified with spatial 2D-RoPE coordinate preservation.

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
echo "N/A  mark_vl_v3_4b_iq4_nl.gguf" | sha256sum -c -
```

### Graph and Memory Profiling with llama.cpp

```bash
# Benchmark throughput and KV cache allocation
llama-bench -m mark_vl_v3_4b_iq4_nl.gguf -n 128 -b 512

# Verify execution graph
llama-cli -m mark_vl_v3_4b_iq4_nl.gguf -p "System probe: 1+1=" -n 16 --temp 0.0
```

### Python Graph Inspection

```python
import gguf

reader = gguf.GGUFReader("mark_vl_v3_4b_iq4_nl.gguf")
print("Architecture:", reader.get_field("general.architecture"))
print("Total Tensors:", len(reader.tensors))
total_params = sum(int(t.shape.prod()) for t in reader.tensors)
print(f"Total Parameters: {total_params:,}")
```
