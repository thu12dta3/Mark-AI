# Mark AI -- Foundational Neural Architecture & Systems Engine

The central cognitive, polyglot systems engineering, and multimodal engine of the MARK ecosystem. Built on native DeepSeek2 architecture featuring Multi-Head Latent Attention (MLA) and 3D Mixture-of-Experts (MoE) in non-linear 4-bit quantization (IQ4_NL).

## Ecosystem Model Matrix (11 Models - Forensic GGUF Verification)

| Model Name | Status | Size (MB) | Total Params | Active Params | Layers | MoE (R/S/U) | SHA-256 | Hugging Face |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **Mark-Engineer-302M** | Trained (Valid) | **135.12** | 251.45M | 119.33M | 8 | 16/1/2 | `5e9d7d6feb...` | [miceworks/Mark-Engineer-302M](https://huggingface.co/miceworks/Mark-Engineer-302M) |
| **Mark-SLM-V2-302M** | Untrained Base | **176.84** | 329.13M | 155.72M | 16 | 16/1/2 | `388237d49e...` | [miceworks/Mark-SLM-V2-302M](https://huggingface.co/miceworks/Mark-SLM-V2-302M) |
| **Mark-Engineer-1.5B** | Trained (Valid) | **802.94** | 1.50B | 364.15M | 16 | 16/1/2 | `fa7d3061f3...` | [miceworks/Mark-Engineer-1.5B](https://huggingface.co/miceworks/Mark-Engineer-1.5B) |
| **Mark-General-1.5B** | Trained (Valid) | **802.97** | 1.50B | 364.22M | 16 | 16/1/2 | `4f2be0d7c3...` | [miceworks/Mark-General-1.5B](https://huggingface.co/miceworks/Mark-General-1.5B) |
| **Mark-VL-2.2B** | Trained (Valid) | **1166.72** | 2.17B | 395.50M | 14 | 24/1/2 | `073d3c00ea...` | [miceworks/Mark-VL-2.2B](https://huggingface.co/miceworks/Mark-VL-2.2B) |
| **Mark-Engineer-V2-8B** | Untrained Base | **4431.26** | 8.26B | 1.77B | 32 | 24/1/2 | `28c014b5b9...` | [miceworks/Mark-Engineer-V2-8B](https://huggingface.co/miceworks/Mark-Engineer-V2-8B) |
| **Mark-SLM-V3-150M** | Untrained Base | **88.94** | 165.35M | 72.45M | 15 | 16/1/2 | `c10b0cb0e1...` | [miceworks/Mark-SLM-V3-150M](https://huggingface.co/miceworks/Mark-SLM-V3-150M) |
| **Mark-V3-1.1B** | Untrained Base | **656.13** | 1.22B | 330.65M | 18 | 32/2/4 | `e12c358e81...` | [miceworks/Mark-V3-1.1B](https://huggingface.co/miceworks/Mark-V3-1.1B) |
| **Mark-General-V3-1.5B** | Untrained Base | **816.03** | 1.52B | 405.73M | 18 | 32/2/4 | `7677ebd444...` | [miceworks/Mark-General-V3-1.5B](https://huggingface.co/miceworks/Mark-General-V3-1.5B) |
| **Mark-Engineer-V3-1.5B** | Untrained Base | **816.03** | 1.52B | 405.73M | 18 | 32/2/4 | `62b8bd003a...` | [miceworks/Mark-Engineer-V3-1.5B](https://huggingface.co/miceworks/Mark-Engineer-V3-1.5B) |
| **Mark-Experimental-V3-8B** | Untrained Base | **4382.55** | 8.17B | 2.00B | 32 | 32/2/4 | `792e127a71...` | [miceworks/Mark-Experimental-V3-8B](https://huggingface.co/miceworks/Mark-Experimental-V3-8B) |

## Architectural Highlights

1. **Native DeepSeek2 Format**: Preserves 3D expert tensors (`ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`) and Multi-Head Latent Attention compression (`attn_q_a`, `attn_q_b`, `attn_kv_a_mqa`, `attn_kv_b`) without dense degradation.
2. **IQ4_NL Quantization**: Non-linear 4-bit quantization for all linear and MoE layers, retaining FP32 precision for RMSNorm and Aux-Loss-Free router bias.
3. **Transparent Scientific Classification**:
   - Models with verified checkpoints on disk are marked as *Trained Checkpoint (Validated)*.
   - Newly initialized models are marked strictly as *Untrained Architectural Base*. No unverified benchmark figures are reported.

## Quickstart

```bash
# Clone the repository
git clone https://github.com/thu12dta3/Mark-AI.git
cd Mark-AI

# Run validation suite
python3 -m pytest tests/ -q
```
