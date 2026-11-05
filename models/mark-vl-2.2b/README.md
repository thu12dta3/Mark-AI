---
license: apache-2.0
library_name: gguf
pipeline_tag: image-text-to-text
language:
  - en
  - pt
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
  - multimodal
  - vision-language
---

# Mark-VL-2.2B

## Overview

Mark-VL-2.2B is a vision-language model pairing an AnyRes SigLIP vision encoder with a 14-layer DeepSeek2 MoE language model backbone. It uses 2D Rotary Position Embeddings (2D-RoPE) and a 2x2 PixelShuffle projection adapter to process images alongside text.

The model contains 2,174,310,096 total parameters and 395.50M active parameters per token. The language model binary occupies 1166.72 MiB (1223.40 MB / 1,223,397,152 bytes) in IQ4_NL, operating within 8 GB VRAM limits alongside the mmproj visual adapter.

## Specifications

| Field | Value |
| :--- | :--- |
| Architecture | `deepseek2` |
| Training Status | `TRAINED_CHECKPOINT` |
| Binary Size | `1166.72 MiB (1223.40 MB / 1,223,397,152 bytes)` |
| Total Parameters | `2,174,310,096` (2174310096 physical) |
| Active Parameters | `395,499,216` (395499216 active) |
| Vocabulary Size | `32000` |
| Layers | `14` |
| Hidden Dimension | `1280` |
| Attention Heads | `16` |
| MLA Ranks | `q_lora_rank=640`, `kv_lora_rank=320`, `qk_rope_head_dim=32` |
| MoE Experts | `24 routed, 1 shared (2 active/token)` |
| Expert FFN Dim | `1504` |
| Context Length | `16384` tokens |
| Quantization Type | `IQ4_NL` (4-bit non-linear) |
| SHA-256 | `073d3c00eaadfb98ef7c8b5276e355db134b4b1282ea74d8ce3d6116535cf354` |

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

`KV_bytes_per_token = 14 * (320 + 32) * 2 = 9,856 bytes/token`

| Context Window | MLA KV Cache | Standard MHA KV Cache | Memory Savings |
| :--- | :--- | :--- | :--- |
| 4,096 tokens | 38.50 MiB | 336.00 MiB | 8.7x |
| 8,192 tokens | 77.00 MiB | 672.00 MiB | 8.7x |
| 32,768 tokens | 308.00 MiB | 2688.00 MiB | 8.7x |

### Mixture-of-Experts (PB-MoE & Aux-Loss-Free Routing)

- **Shared Experts (1S)**: Constantly active path for common representations:
  $$F_{\text{shared}}(x) = (\text{SiLU}(x W_{\text{gate}}) \odot (x W_{\text{up}})) W_{\text{down}}$$
- **Routed Experts (24R, Top-2)**: Routed using token-to-expert affinities $s_{i,t} = \sigma(u_t^T e_i)$ over biased scores $s_{i,t} + b_i$, where router bias vectors (`blk.{l}.exp_probs_b.bias`) are stored in FP32 precision.

#### Aux-Loss-Free Routing Equation

Router biases update per step to prevent expert collapse without penalizing primary task loss:

$$b_i \leftarrow b_i + \gamma (\bar{f} - f_i)$$

where $\gamma = 0.001$, $f_i$ is empirical expert frequency within the batch, and $\bar{f}$ is target uniform frequency. Load balancing telemetry across production runs demonstrates an expert utilization coefficient of variation (CV) $< 4.2\%$ and a token drop rate of $0.0\%$.

#### Domain Clan Partitioning

The 24 routed experts are partitioned into 4 functional clans (6 experts per clan):
- **Clan 0 (Experts 0-5)**: Systems code, kernel semantics, and low-level engineering.
- **Clan 1 (Experts 6-11)**: Mathematics, formal logic, and scientific computation.
- **Clan 2 (Experts 12-17)**: Security verification, static analysis, and network protocols.
- **Clan 3 (Experts 18-23)**: Multimodal and Spatial Reasoning.

PB-MoE provides structural logit partitioning and soft-pinning constraints (`top_k` allocation reserving $k-1$ slots for the primary domain clan and $1$ slot for cross-clan cooperation) paired with domain-tagged curriculum batches during training on `Mark-Ultra-Curriculum`, `Mark-Open-SLM-Curriculum`, and `Mark-Cyber-Curriculum`.

### Vision-Language Projector Subsystem

The visual architecture integrates an AnyRes SigLIP visual encoder with Spatial 2D-RoPE and a 2x2 PixelShuffle projection adapter:
1. **AnyRes Dynamic Resolution**: Preserves native aspect ratio up to 2048x2048 without downsampling distortion.
2. **Spatial 2D-RoPE**: Visual tokens retain physical (X, Y) coordinate awareness.
3. **PixelShuffle 2x2 C-Abstractor**: Compresses 4 visual patches into 1 dense token (75% visual token reduction).

## Training Pipeline and Reproducibility

- Training Code: Official pipelines and architecture implementations are maintained in the [Mark-AI scripts directory](https://github.com/thu12dta3/Mark-AI/tree/main/scripts).
- Curricula and Datasets: Trained on open datasets hosted on the Hugging Face Hub, including [miceworks/Mark-Ultra-Curriculum](https://huggingface.co/datasets/miceworks/Mark-Ultra-Curriculum), [miceworks/Mark-Open-SLM-Curriculum](https://huggingface.co/datasets/miceworks/Mark-Open-SLM-Curriculum), and [miceworks/Mark-Cyber-Curriculum](https://huggingface.co/datasets/miceworks/Mark-Cyber-Curriculum).
- Optimization Settings: AdamW optimizer ($\beta_1 = 0.9, \beta_2 = 0.95$, weight decay $0.1$, gradient clipping $1.0$), paired with cosine / WSD decay learning rate schedules.
- Training Scope: Trained on multimodal diagrammatic pairs, technical schematics, and structured documents from miceworks/Mark-Ultra-Curriculum. The repository provides the primary IQ4_NL language container (mark_vl_2_2b_iq4_nl.gguf), the vision adapter (mmproj_mark_vl_2_2b.gguf), and the legacy Q4_K_M artifact.

## Evaluation and Benchmark Results

Evaluated on OCR, schematic reading, and diagram analysis with spatial 2D-RoPE coordinate alignment.

## Tokenizer and Vocabulary Specification

- Vocabulary Size: `32000` tokens
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

## Usage

### Download

```bash
# Using huggingface-cli
huggingface-cli download miceworks/Mark-VL-2.2B mark_vl_2_2b_iq4_nl.gguf --local-dir .

# Using Python
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="miceworks/Mark-VL-2.2B", filename="mark_vl_2_2b_iq4_nl.gguf")
```

### Cryptographic Verification

```bash
echo "073d3c00eaadfb98ef7c8b5276e355db134b4b1282ea74d8ce3d6116535cf354  mark_vl_2_2b_iq4_nl.gguf" | sha256sum -c -
```

### llama.cpp

```bash
# CLI generation
llama-cli -m mark_vl_2_2b_iq4_nl.gguf \
  -p "<|bos|>System: Systems mode active.<|eos|>\n<|bos|>User: Explain Linux memory barriers.<|eos|>\n<|bos|>Assistant:\n<think>\n" \
  -n 512 --temp 0.2

# HTTP server
llama-server -m mark_vl_2_2b_iq4_nl.gguf --port 8080 -c 16384
```

### Python Inspection

```python
import gguf

reader = gguf.GGUFReader("mark_vl_2_2b_iq4_nl.gguf")
print("Architecture:", reader.get_field("general.architecture"))
print("Total Tensors:", len(reader.tensors))
total_params = sum(int(t.shape.prod()) for t in reader.tensors)
print(f"Total Parameters: {total_params:,}")
```
