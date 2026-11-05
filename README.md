# Mark-AI

Sparse Transformer research codebase exploring Multi-Head Latent Attention (MLA) and fine-grained Mixture-of-Experts (MoE) exported to native `deepseek2` GGUF containers in 4-bit non-linear quantization (`IQ4_NL`). Engineered and profiled on consumer 8 GB hardware (`NVIDIA GeForce RTX 2060 Super (8 GB GDDR6, 256-bit, 448 GB/s)`).

## Public Research Checkpoint: Mark-SLM-V3-150M

**`Mark-SLM-V3-150M` is a 150M-parameter experimental sparse language model trained on 3.47M tokens (`miceworks/Mark-Open-SLM-Curriculum`), achieving an in-distribution Intelligence Index of `16.36` on the Open SLM Leaderboard ([PR #291](https://huggingface.co/spaces/AxiomicLabs/Open_SLM_Leaderboard/discussions/291)). It is an architectural research testbed for Multi-Head Latent Attention (MLA) and fine-grained MoE on 8 GB GPUs, NOT a general-purpose conversational or multilingual assistant.**

- **Public Hugging Face Repository:** [`miceworks/Mark-SLM-V3-150M`](https://huggingface.co/miceworks/Mark-SLM-V3-150M)
- **GGUF Container:** `mark_slm_v3_150m_iq4_nl.gguf` (`90.37 MiB / 94.76 MB / 94,764,576 bytes`)
- **SHA-256:** `4bbfe2dc3fc7a6ac895d56e30352bda0de1322f9826afa28cafe7d24b0e99632`
- **Unique Architectural Parameters:** `146,899,248` (`54,001,968` active per token across 15 layers: 2 dense + 13 MoE)
- **Materialized GGUF Parameters:** `165,349,680` across `273` tensors (including materialized `output.weight`)

### Open SLM Leaderboard Evaluation (In-Distribution Curriculum Learning)

| Model | Unique Parameters | HellaSwag (`acc_norm`) | ARC-E (`acc_norm`) | ARC-C (`acc_norm`) | PIQA (`acc_norm`) | ArithMark-3 (`acc`) | Intelligence Index |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mark-SLM-V3-150M (Model Soup 50/50)** | 146,899,248 | 37.10% (16.13) | 40.80% (21.07) | 25.50% (0.67) | 62.50% (25.00) | 33.90% (11.87) | **16.36** |
| **Mark-SLM-3M (Local Sub-3M Experiment)** | 2,946,008 | 25.60% (0.80) | 26.60% (2.13) | 23.75% (0.00) | 52.00% (4.00) | 29.70% (6.27) | **3.85** |

Disclosure: `miceworks/Mark-Open-SLM-Curriculum` incorporates official training splits from HellaSwag, PIQA, ARC, SciQ, and procedural ArithMark (permitted under Open SLM Leaderboard Rules 01 and 02), so these scores measure in-distribution curriculum learning rather than broad out-of-distribution generalization.

## Public Datasets on Hugging Face

All three public training datasets have been deduplicated, decontaminated against evaluation splits, stripped of local filesystem paths, and released under the `apache-2.0` license:

| Dataset | Type | Unique Samples | Primary Focus | Hugging Face Repository |
| :--- | :--- | :---: | :--- | :--- |
| **Mark-Open-SLM-Curriculum** | Pretraining / SLM | 11,990 (3.47M tokens) | Rule 02 compliant pretraining splits from HellaSwag, ARC, PIQA, SciQ, ArithMark-3, and Cosmopedia STEM (decontaminated against ARC val/test) | [miceworks/Mark-Open-SLM-Curriculum](https://huggingface.co/datasets/miceworks/Mark-Open-SLM-Curriculum) |
| **Mark-Cyber-Curriculum** | SFT / DPO | 4,613 | Strictly 3-field deduplicated vulnerability research (CWE/CVE), Red/Blue Team SecOps, and eBPF security (`CyberNative`, `Trendyol`, `neuralchemy`) | [miceworks/Mark-Cyber-Curriculum](https://huggingface.co/datasets/miceworks/Mark-Cyber-Curriculum) |
| **Mark-Ultra-Curriculum** | Systems SFT | 1,131 | Deduplicated Rust/C++ concurrency, Linux eBPF, robotics kinematics, and STEM reasoning | [miceworks/Mark-Ultra-Curriculum](https://huggingface.co/datasets/miceworks/Mark-Ultra-Curriculum) |

## Architectural Definitions and Local Configurations (`src/model/` & `models/`)

Alongside the public `Mark-SLM-V3-150M` checkpoint, this repository defines local PyTorch reference modules (`src/model/`) and GGUF export configurations (`models/`) used to test `deepseek2` tensor packing, KV cache scaling, and narrow curriculum convergence:

| Architecture Configuration | Status | GGUF Footprint | Unique Parameters | Active Parameters | Vocab Size | Availability |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
| **Mark-Engineer-302M** | `EXPERIMENTAL_NARROW_CHECKPOINT` | 135.12 MiB (141.69 MB / 141,686,432 bytes) | 251,452,416 | 119,331,840 | 4311 | Local Config (`models/mark-engineer-302m/`) |
| **Mark-SLM-V2-302M** | `EXPERIMENTAL_NARROW_CHECKPOINT` | 177.76 MiB (186.39 MB / 186,394,944 bytes) | 329,132,000 | 155,723,744 | 49152 | Local Config (`models/mark-slm-v2-302m/`) |
| **Mark-Engineer-1.5B** | `EXPERIMENTAL_NARROW_CHECKPOINT` | 802.94 MiB (841.94 MB / 841,941,664 bytes) | 1,496,121,824 | 364,151,264 | 4311 | Local Config (`models/mark-engineer-1.5b/`) |
| **Mark-General-1.5B** | `EXPERIMENTAL_NARROW_CHECKPOINT` | 802.97 MiB (841.98 MB / 841,977,664 bytes) | 1,496,185,824 | 364,215,264 | 4377 | Local Config (`models/mark-general-1.5b/`) |
| **Mark-VL-2.2B** | `EXPERIMENTAL_NARROW_CHECKPOINT` | 1166.72 MiB (1223.40 MB / 1,223,397,152 bytes) | 2,174,310,096 | 395,499,216 | 32000 | Local Config (`models/mark-vl-2.2b/`) |
| **Mark-Engineer-V2-8B** | `UNTRAINED_ARCHITECTURAL_SCAFFOLD` | 4431.26 MiB (4646.52 MB / 4,646,516,576 bytes) | 8,258,809,040 | 1,770,745,040 | 49152 | Local Config (`models/mark-engineer-v2-8b/`) |
| **Mark-SLM-V3-150M** | `TRAINED_CHECKPOINT` | 90.37 MiB (94.76 MB / 94,764,576 bytes) | 146,899,248 | 54,001,968 | 32032 | [Public Checkpoint (`miceworks/Mark-SLM-V3-150M`)](https://huggingface.co/miceworks/Mark-SLM-V3-150M) |
| **Mark-V3-1.1B** | `UNTRAINED_ARCHITECTURAL_SCAFFOLD` | 656.13 MiB (688.00 MB / 688,002,048 bytes) | 1,222,464,960 | 330,651,072 | 32032 | Local Config (`models/mark-v3-1.1b/`) |
| **Mark-General-V3-1.5B** | `UNTRAINED_ARCHITECTURAL_SCAFFOLD` | 816.03 MiB (855.67 MB / 855,669,248 bytes) | 1,520,496,960 | 405,729,600 | 32032 | Local Config (`models/mark-general-v3-1.5b/`) |
| **Mark-Engineer-V3-1.5B** | `EXPERIMENTAL_NARROW_CHECKPOINT` | 816.95 MiB (856.63 MB / 856,632,832 bytes) | 1,520,496,960 | 405,729,600 | 32032 | Local Config (`models/mark-engineer-v3-1.5b/`) |
| **Mark-General-V3-8B** | `UNTRAINED_ARCHITECTURAL_SCAFFOLD` | 4383.98 MiB (4596.94 MB / 4,596,938,048 bytes) | 8,168,001,024 | 2,002,374,144 | 32032 | Local Config (`models/mark-general-v3-8b/`) |
| **Mark-Experimental-V3-8B** | `UNTRAINED_ARCHITECTURAL_SCAFFOLD` | 4382.55 MiB (4595.44 MB / 4,595,438,848 bytes) | 8,168,001,024 | 2,002,374,144 | 32032 | Local Config (`models/mark-experimental-v3-8b/`) |
| **Mark-SLM-3M** | `TRAINED_CHECKPOINT` | 3.12 MiB (3.27 MB / 3,271,680 bytes) | 2,946,008 | 2,503,640 | 4096 | Local Config (`models/mark-slm-3m/`) |
| **Mark-V3-32B** | `UNTRAINED_ARCHITECTURAL_SCAFFOLD` | PyTorch / GGUF Config | 31,989,012,480 | 3,677,387,904 | 32032 | Local Config (`models/mark-v3-32b/`) |
| **Mark-VL-V3-4B** | `UNTRAINED_ARCHITECTURAL_SCAFFOLD` | PyTorch / GGUF Config | 4,134,688,769 | 824,483,584 | 32032 | Local Config (`models/mark-vl-v3-4b/`) |
| **Mark-General-2.2B** | `EXPERIMENTAL_NARROW_CHECKPOINT` | PyTorch / GGUF Config | 2,223,906,816 | 375M FLOPs (Top-2 routed + 1 shared) | 32000 | Local Config (`models/mark-general-2.2b/`) |

## Architecture Details

### Multi-Head Latent Attention (MLA)

To reduce Key-Value cache memory during long-context inference on 8 GB GPUs (`NVIDIA GeForce RTX 2060 Super (8 GB GDDR6, 256-bit, 448 GB/s)`), keys and values are compressed into a shared low-rank latent vector:

```text
c_KV = h_t * W_DKV          (dim: kv_lora_rank)
c_Q  = h_t * W_DQ           (dim: q_lora_rank)
q_C  = RMSNorm(c_Q * W_UQ)
k_C  = RMSNorm(c_KV * W_UK)
k_R  = RoPE(h_t * W_KR)     (dim: qk_rope_head_dim)
q_t  = [q_C ; q_R]
k_t  = [k_C ; k_R]
```

### Mixture-of-Experts and Sign-Based Router Bias

Feed-forward layers combine shared experts with top-K routed experts gated via sigmoid affinities $s_{i,t} = \sigma(u_t^T e_i)$ plus learned FP32 bias $b_i$ (`blk.{l}.exp_probs_b.bias`), updated during training via the sign-based load-balancing rule:

$$b_i \leftarrow b_i + \gamma \cdot \operatorname{sign}(\bar{f} - f_i)$$

### Experimental PyTorch Research Extensions (Not Active in llama.cpp GGUF Runtime)

Research extensions implemented in `src/model/mark_v3_arch.py` (such as Latent Recursive Depth when `core_loops > 1` and domain clan soft-pinning during PyTorch training) are PyTorch-only modules; exported GGUF files execute as standard sequential `deepseek2` graphs in `llama.cpp`.

## Testing and Verification

```bash
# 1. Validate Python syntax across all modules
python3 -m compileall -q src/ export/ scripts/ models/ tests/

# 2. Run the automated verification suite
pytest tests/ -v
```
