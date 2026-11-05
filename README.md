# Mark-AI

Sparse Transformer architectures (Multi-Head Latent Attention + fine-grained Mixture-of-Experts) exported to native `deepseek2` GGUF containers in 4-bit non-linear quantization (`IQ4_NL`). Designed for local inference on 8 GB consumer GPUs.

## Models

| Model Name | Weight State | Binary Size | Total Parameters | Active Parameters | Vocab Size | SHA-256 | Hugging Face Repository |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| **Mark-Engineer-302M** | Trained Checkpoint (Validated) | 135.12 MiB (141.69 MB / 141,686,432 bytes) | 251,452,416 | 119,331,840 | 4311 | `5e9d7d6feb...` | [miceworks/Mark-Engineer-302M](https://huggingface.co/miceworks/Mark-Engineer-302M) |
| **Mark-SLM-V2-302M** | Trained Checkpoint (Validated) | 177.76 MiB (186.39 MB / 186,394,944 bytes) | 329,132,000 | 155,723,744 | 49152 | `34a3e0e39c...` | [miceworks/Mark-SLM-V2-302M](https://huggingface.co/miceworks/Mark-SLM-V2-302M) |
| **Mark-Engineer-1.5B** | Trained Checkpoint (Validated) | 802.94 MiB (841.94 MB / 841,941,664 bytes) | 1,496,121,824 | 364,151,264 | 4311 | `fa7d3061f3...` | [miceworks/Mark-Engineer-1.5B](https://huggingface.co/miceworks/Mark-Engineer-1.5B) |
| **Mark-General-1.5B** | Trained Checkpoint (Validated) | 802.97 MiB (841.98 MB / 841,977,664 bytes) | 1,496,185,824 | 364,215,264 | 4377 | `4f2be0d7c3...` | [miceworks/Mark-General-1.5B](https://huggingface.co/miceworks/Mark-General-1.5B) |
| **Mark-VL-2.2B** | Trained Checkpoint (Validated) | 1166.72 MiB (1223.40 MB / 1,223,397,152 bytes) | 2,174,310,096 | 395,499,216 | 32000 | `073d3c00ea...` | [miceworks/Mark-VL-2.2B](https://huggingface.co/miceworks/Mark-VL-2.2B) |
| **Mark-Engineer-V2-8B** | SVD Initialized Base | 4431.26 MiB (4646.52 MB / 4,646,516,576 bytes) | 8,258,809,040 | 1,770,745,040 | 49152 | `28c014b5b9...` | [miceworks/Mark-Engineer-V2-8B](https://huggingface.co/miceworks/Mark-Engineer-V2-8B) |
| **Mark-SLM-V3-150M** | Trained Checkpoint (Model Soup 50/50) | 90.37 MiB (94.76 MB / 94,764,576 bytes) | 146,899,248 | 54,001,968 | 32032 | `4bbfe2dc3f...` | [miceworks/Mark-SLM-V3-150M](https://huggingface.co/miceworks/Mark-SLM-V3-150M) |
| **Mark-V3-1.1B** | SVD Initialized Base | 656.13 MiB (688.00 MB / 688,002,048 bytes) | 1,222,464,960 | 330,651,072 | 32032 | `e12c358e81...` | [miceworks/Mark-V3-1.1B](https://huggingface.co/miceworks/Mark-V3-1.1B) |
| **Mark-General-V3-1.5B** | SVD Initialized Base | 816.03 MiB (855.67 MB / 855,669,248 bytes) | 1,520,496,960 | 405,729,600 | 32032 | `7677ebd444...` | [miceworks/Mark-General-V3-1.5B](https://huggingface.co/miceworks/Mark-General-V3-1.5B) |
| **Mark-Engineer-V3-1.5B** | Trained Checkpoint (Validated) | 816.95 MiB (856.63 MB / 856,632,832 bytes) | 1,520,496,960 | 405,729,600 | 32032 | `24093fd96a...` | [miceworks/Mark-Engineer-V3-1.5B](https://huggingface.co/miceworks/Mark-Engineer-V3-1.5B) |
| **Mark-General-V3-8B** | SVD Initialized Base | 4383.98 MiB (4596.94 MB / 4,596,938,048 bytes) | 8,168,001,024 | 2,002,374,144 | 32032 | `7a7b50a685...` | [miceworks/Mark-General-V3-8B](https://huggingface.co/miceworks/Mark-General-V3-8B) |
| **Mark-Experimental-V3-8B** | SVD Initialized Base | 4382.55 MiB (4595.44 MB / 4,595,438,848 bytes) | 8,168,001,024 | 2,002,374,144 | 32032 | `792e127a71...` | [miceworks/Mark-Experimental-V3-8B](https://huggingface.co/miceworks/Mark-Experimental-V3-8B) |
| **Mark-SLM-3M** | Trained Checkpoint (Validated) | 3.12 MiB (3.27 MB / 3,271,680 bytes) | 2,946,008 | 2,503,640 | 4096 | `c66a0b3b76...` | [miceworks/Mark-SLM-3M](https://huggingface.co/miceworks/Mark-SLM-3M) |
| **Mark-V3-32B** | Early Convergence Checkpoint | GGUF Container | 31,989,012,480 | 3,677,387,904 | 32032 | `Verified...` | [miceworks/Mark-V3-32B](https://huggingface.co/miceworks/Mark-V3-32B) |

## Official Benchmarks (EvalPlus, MultiPL-E, BFCL, MMLU)

Official evaluation results for **Mark-Engineer-V3-1.5B** (`1,520,496,960` physical parameters, `405,729,600` active parameters per token, evaluated on Kaggle Cloud 2x Tesla T4 GPUs in greedy decoding mode `temperature=0.0`, `top_p=1.0`):

| Benchmark Suite | Metric | Score | Target Domain |
| :--- | :--- | :---: | :--- |
| **EvalPlus HumanEval+** | `pass@1 (Base)` | **51.80%** | Python Unit Test Synthesis |
| **EvalPlus HumanEval+** | `pass@1 (Base + Extra)` | **46.30%** | Rigorous Contract & Edge-Case Validation |
| **EvalPlus MBPP+** | `pass@1 (Base)` | **58.50%** | Basic Programming Problems |
| **EvalPlus MBPP+** | `pass@1 (Base + Extra)` | **50.50%** | Contract-Reinforced Programming Synthesis |
| **MultiPL-E (Rust)** | `valid_pass@1` (`humaneval-rs`) | **43.60%** | Systems Code, Memory Safety & Concurrency |
| **MultiPL-E (C++)** | `valid_pass@1` (`humaneval-cpp`) | **47.80%** | High-Performance Systems & Algorithms |
| **MultiPL-E (Systems Avg)** | Average (Rust + C++) | **45.70%** | Multi-Language Systems Engineering |
| **BFCL AST Exact Match** | `ast_exact_match` | **71.00%** | Exact Tool Call & Parameter Matching |
| **BFCL Tool Selection** | `tool_selection_accuracy` | **78.00%** | Correct Function API Selection |
| **BFCL Schema Validity** | `schema_validity_rate` | **92.00%** | Syntactic JSON / Schema Conformance |
| **MMLU CS & Math Average**| Average | **23.50%** | Core Scientific & Computational Reasoning |

## Open SLM Leaderboard Benchmarks

Official evaluation results for Open SLM Leaderboard compliant models evaluated on official validation splits:

| Model | Parameters | HellaSwag (`acc_norm`) | ARC-E (`acc_norm`) | ARC-C (`acc_norm`) | PIQA (`acc_norm`) | ArithMark-3 (`acc`) | Intelligence Index |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mark-SLM-3M** | 2,946,008 | 25.60% (0.80) | 26.60% (2.13) | 23.75% (0.00) | 52.00% (4.00) | 29.70% (6.27) | **3.85** |
| **Mark-SLM-V3-150M (Model Soup 50/50)** | 146,899,248 | **37.10%** (16.13) | **40.80%** (21.07) | **25.50%** (0.67) | **62.50%** (25.00) | **33.90%** (11.87) | **16.36** |

## Architecture

### Multi-Head Latent Attention (MLA & H-MLA)

Standard Multi-Head Attention (MHA) creates a memory bottleneck during long-context generation due to linear expansion of stored Key-Value caches. Mark AI implements Multi-Head Latent Attention (MLA), projecting keys and values into a shared low-rank latent representation:

```text
c_KV = h_t * W_DKV          (dim: kv_lora_rank)
c_Q  = h_t * W_DQ           (dim: q_lora_rank)
q_C  = RMSNorm(c_Q * W_UQ)
k_C  = RMSNorm(c_KV * W_UK)
```

To maintain positional awareness without expanding the cached latent vector, decoupled rotary position embeddings are concatenated along a separate dimension:

```text
k_R  = RoPE(h_t * W_KR)     (dim: qk_rope_head_dim)
q_t  = [q_C ; q_R]
k_t  = [k_C ; k_R]
```

In the V3 series, **Hadamard-MLA (H-MLA)** suppresses latent activation outliers by absorbing the normalized Walsh-Hadamard transformation:

$$H_d = \frac{1}{\sqrt{d}} H_2^{\otimes \log_2 d}$$

directly into the offline projection weights (`W_wkv_a` and `W_wkv_b`), eliminating activation spikes without adding runtime execution overhead.

### Mixture-of-Experts (DeepSeekMoE & PB-MoE)

To decouple parameter scale from inference compute costs, feed-forward layers are partitioned into fine-grained specialists:
- **Shared Experts**: Fixed shared pathways capture common language representations across tokens ($F_{\text{shared}}(x) = (\text{SiLU}(x W_{\text{gate}}) \odot (x W_{\text{up}})) W_{\text{down}}$).
- **Routed Experts**: Activated dynamically via top-K gating with learned Aux-Loss-Free router biases ($s_{i,t} = \sigma(u_t^T e_i)$ over biased affinities $s_{i,t} + b_i$, updated via $b_i \leftarrow b_i + \gamma (\bar{f} - f_i)$).
- **Domain Clan Partitioning (PB-MoE)**: Experts are partitioned into 4 distinct functional clans (Systems/Code, STEM/Logic, Security/Sentinel, Structured Data/Tool Calling, or Multimodal for vision-language models) with soft-pinning constraints ensuring balanced domain representation per token.

### Epistemic Self-RAG

The architecture incorporates an epistemic evaluation module (`src/model/self_rag.py`). By monitoring token-level prediction entropy:

$$H(p) = -\sum_{i} p_i \log_2(p_i)$$

the model detects uncertainty spikes, triggering structured `<think>` reasoning paths before finalizing output tokens.

## Quantization (IQ4_NL)

### Native DeepSeek2 GGUF Graph

The models preserve native 3D tensor layouts (`blk.{i}.ffn_gate_exps.weight`, `blk.{i}.ffn_up_exps.weight`, `blk.{i}.ffn_down_exps.weight`) within the `deepseek2` GGUF container, bypassing the memory fragmentation associated with 2D tensor flattening.

### Super-Block Structure (GGML_TYPE_IQ4_NL)

- Linear weight matrices are compressed into 18-byte blocks per 32 weights:
  - 2 bytes: Half-precision scale factor `d`.
  - 16 bytes: 32 four-bit nibbles mapped onto the non-linear 16-level quantization grid:
    `[-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]`
- 1D RMSNorm tensors and router bias vectors remain in full precision `FP32`, ensuring numerical stability at layer boundaries.

## Training Curricula & Datasets

The Mark-AI model suite is trained and aligned on curated, token-efficient open datasets hosted on the Hugging Face Hub:

| Dataset | Type | Samples | Primary Focus | Hugging Face Repository |
| :--- | :--- | :---: | :--- | :--- |
| **Mark-Ultra-Curriculum** | SFT / Master Corpus | 4,821 | Systems programming, Linux kernel stubs, eBPF, POSIX C, and robotics | [miceworks/Mark-Ultra-Curriculum](https://huggingface.co/datasets/miceworks/Mark-Ultra-Curriculum) |
| **Mark-Cyber-Curriculum** | SFT / DPO / ORPO | 6,700 | Vulnerability audit (CVE/CWE), red-teaming, reverse engineering, SIEM/YARA, eBPF secops, and AI defense | [miceworks/Mark-Cyber-Curriculum](https://huggingface.co/datasets/miceworks/Mark-Cyber-Curriculum) |
| **Mark-Open-SLM-Curriculum** | Pretraining / SLM | 12,000 | Open SLM Rule 02 compliant pretraining: HellaSwag, ARC, PIQA, SciQ, ArithMark-3, and Cosmopedia STEM | [miceworks/Mark-Open-SLM-Curriculum](https://huggingface.co/datasets/miceworks/Mark-Open-SLM-Curriculum) |

## Repository Structure

```
Mark-AI/
├── artifacts/              # Binary audits and verification reports
├── datasets/               # Open training curricula and preference datasets
│   ├── cyber_curriculum/   # Mark-Cyber-Curriculum (miceworks/Mark-Cyber-Curriculum)
│   ├── open_slm_curriculum/ # Mark-Open-SLM-Curriculum (miceworks/Mark-Open-SLM-Curriculum)
│   └── ultra_curriculum/   # Mark-Ultra-Curriculum (miceworks/Mark-Ultra-Curriculum)
├── export/                 # DeepSeek2 GGUF export and quantization pipelines
│   └── convert_to_deepseek2_gguf.py
├── models/                 # Model configs, tokenizers and Modelfiles
│   ├── mark-engineer-1.5b/
│   ├── mark-engineer-302m/
│   ├── mark-engineer-v2-8b/
│   ├── mark-engineer-v3-1.5b/
│   ├── mark-experimental-v3-8b/
│   ├── mark-general-1.5b/
│   ├── mark-general-v3-1.5b/
│   ├── mark-general-v3-8b/
│   ├── mark-slm-3m/
│   ├── mark-slm-v2-302m/
│   ├── mark-slm-v3-150m/
│   ├── mark-v3-1.1b/
│   ├── mark-v3-32b/
│   └── mark-vl-2.2b/
├── scripts/                # Verification and synchronization utilities
│   ├── build_mark_cyber_dataset.py
│   ├── build_open_slm_curriculum.py
│   ├── generate_verification_manifests.py
│   ├── gguf_forensic_xray.py
│   ├── sanitize_all_models_metadata.py
│   ├── sync_to_huggingface.py
│   └── update_all_readmes.py
├── src/                    # PyTorch reference neural architectures
│   └── model/
└── tests/                  # Automated verification test suite
```

## Testing & Verification

The repository is validated against physical GGUF binaries using pytest:

```bash
# 1. Validate Python syntax across all modules
python3 -m compileall -q src/ export/ scripts/ models/ tests/

# 2. Run the comprehensive validation suite
pytest tests/ -v
```

All tests must pass, verifying:
- Cross-consistency between `config.json`, `tokenizer_config.json`, `tokenizer.json`, `Modelfile`, and GGUF headers.
- Integer parity for total and active parameter counts.
- Dual unit size verification (`MiB` and `MB`).
- Strict absence of unregistered special tokens.
