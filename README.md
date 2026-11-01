# Mark AI -- Foundational Neural Architecture & Systems Engine

The central cognitive, polyglot systems engineering, and multimodal engine of the MARK ecosystem. Built on native DeepSeek2 architecture featuring Multi-Head Latent Attention (MLA) and 3D Mixture-of-Experts (MoE) in non-linear 4-bit quantization (IQ4_NL).

## Ecosystem Model Matrix (11 Models - GGUF Binary Ground Truth)

| Model Name | Weight State | Binary Size | Total Parameters | Active Parameters | Vocab Size | SHA-256 | Hugging Face Repository |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| **Mark-Engineer-302M** | Trained Checkpoint (Validated) | 135.12 MiB (141.69 MB / 141,686,432 bytes) | 251,452,416 | 119,331,840 | 4311 | `5e9d7d6feb...` | [miceworks/Mark-Engineer-302M](https://huggingface.co/miceworks/Mark-Engineer-302M) |
| **Mark-SLM-V2-302M** | Trained Checkpoint (Validated) | 177.76 MiB (186.39 MB / 186,394,944 bytes) | 329,132,000 | 155,723,744 | 49152 | `34a3e0e39c...` | [miceworks/Mark-SLM-V2-302M](https://huggingface.co/miceworks/Mark-SLM-V2-302M) |
| **Mark-Engineer-1.5B** | Trained Checkpoint (Validated) | 802.94 MiB (841.94 MB / 841,941,664 bytes) | 1,496,121,824 | 364,151,264 | 4311 | `fa7d3061f3...` | [miceworks/Mark-Engineer-1.5B](https://huggingface.co/miceworks/Mark-Engineer-1.5B) |
| **Mark-General-1.5B** | Trained Checkpoint (Validated) | 802.97 MiB (841.98 MB / 841,977,664 bytes) | 1,496,185,824 | 364,215,264 | 4377 | `4f2be0d7c3...` | [miceworks/Mark-General-1.5B](https://huggingface.co/miceworks/Mark-General-1.5B) |
| **Mark-VL-2.2B** | Trained Checkpoint (Validated) | 1166.72 MiB (1223.40 MB / 1,223,397,152 bytes) | 2,174,310,096 | 395,499,216 | 32000 | `073d3c00ea...` | [miceworks/Mark-VL-2.2B](https://huggingface.co/miceworks/Mark-VL-2.2B) |
| **Mark-Engineer-V2-8B** | Untrained Architectural Base | 4431.26 MiB (4646.52 MB / 4,646,516,576 bytes) | 8,258,809,040 | 1,770,745,040 | 49152 | `28c014b5b9...` | [miceworks/Mark-Engineer-V2-8B](https://huggingface.co/miceworks/Mark-Engineer-V2-8B) |
| **Mark-SLM-V3-150M** | Untrained Architectural Base | 79.97 MiB (83.85 MB / 83,849,984 bytes) | 146,899,248 | 54,001,968 | 32032 | `6308c8c152...` | [miceworks/Mark-SLM-V3-150M](https://huggingface.co/miceworks/Mark-SLM-V3-150M) |
| **Mark-V3-1.1B** | Untrained Architectural Base | 656.13 MiB (688.00 MB / 688,002,048 bytes) | 1,222,464,960 | 330,651,072 | 32032 | `e12c358e81...` | [miceworks/Mark-V3-1.1B](https://huggingface.co/miceworks/Mark-V3-1.1B) |
| **Mark-General-V3-1.5B** | Untrained Architectural Base | 816.03 MiB (855.67 MB / 855,669,248 bytes) | 1,520,496,960 | 405,729,600 | 32032 | `7677ebd444...` | [miceworks/Mark-General-V3-1.5B](https://huggingface.co/miceworks/Mark-General-V3-1.5B) |
| **Mark-Engineer-V3-1.5B** | Untrained Architectural Base | 816.03 MiB (855.67 MB / 855,669,248 bytes) | 1,520,496,960 | 405,729,600 | 32032 | `62b8bd003a...` | [miceworks/Mark-Engineer-V3-1.5B](https://huggingface.co/miceworks/Mark-Engineer-V3-1.5B) |
| **Mark-Experimental-V3-8B** | Untrained Architectural Base | 4382.55 MiB (4595.44 MB / 4,595,438,848 bytes) | 8,168,001,024 | 2,002,374,144 | 32032 | `792e127a71...` | [miceworks/Mark-Experimental-V3-8B](https://huggingface.co/miceworks/Mark-Experimental-V3-8B) |

## Rigorous Measurement Conventions

All numerical metrics in the Mark-AI repository are verified against physical binary files on disk via `gguf.GGUFReader`:
1. **Size Units**: Binary sizes are strictly reported in both mebibytes and megabytes with exact byte counts: `X.XX MiB (Y.YY MB / Z,ZZZ,ZZZ bytes)`, where `1 MiB = 1,048,576 bytes` and `1 MB = 1,000,000 bytes`.
2. **Parameter Counts**: Parameter values represent physical scalar elements across all tensors in the GGUF container. Routed expert parameters are weighted by `num_experts_per_tok / n_routed_experts` to yield exact active parameters per token.
3. **Weight State**:
   - `TRAINED_CHECKPOINT_VALIDATED`: Models trained and verified against ground truth checkpoints on disk.
   - `UN-TRAINED_ARCHITECTURAL_BASE`: Models exported as architectural foundations with clean initialization, pending large-scale pre-training or fine-tuning. No synthetic or unverified benchmarks are reported.

## Architectural Highlights

1. **Native DeepSeek2 Format**: Preserves 3D expert tensors (`ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`) and Multi-Head Latent Attention compression (`attn_q_a`, `attn_q_b`, `attn_kv_a_mqa`, `attn_kv_b`) without dense degradation.
2. **IQ4_NL Quantization**: Non-linear 4-bit quantization for linear and MoE layers, retaining FP32 precision for RMSNorm and Aux-Loss-Free router bias.
3. **Reproducibility**: Each model directory contains an auditable `gguf_verification_manifest.json` with exact checksums, tensor breakdowns, and runtime execution logs.

## Quickstart

```bash
# Clone the repository
git clone https://github.com/thu12dta3/Mark-AI.git
cd Mark-AI

# Run metadata consistency and verification tests
python3 -m pytest tests/ -v
```
