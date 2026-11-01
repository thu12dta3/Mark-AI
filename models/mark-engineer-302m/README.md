# Mark-Engineer-302M

Ultra-lightweight systems engineering model for kernel-level code and low-resource edge devices.

## Forensic Binary Specifications (GGUFReader Verified)

| Property | Value |
| :--- | :--- |
| Architecture | `deepseek2` |
| Quantization Type | `IQ4_NL` (4-bit Non-Linear Weights, FP32 Norms/Bias) |
| Container File | `mark_engineer_302m_iq4_nl.gguf` |
| File Size | `135.12 MiB (141.69 MB / 141,686,432 bytes)` |
| SHA-256 Checksum | `5e9d7d6feb9bc0b6ad5da98e444e3e083a1936e58ad6091230ee674d1ef714b0` |
| Total Parameters | `251,452,416` physical |
| Active Parameters / Token | `119,331,840` active |
| Total Tensors | `115` ({'IQ4_NL': 82, 'F32': 33}) |
| 1D RMSNorm Precision | `100% FP32` |
| Hidden Dimension (`hidden_size`) | `1024` |
| Transformer Layers | `8` |
| Attention Heads | `16` |
| Multi-Head Latent Attention (MLA) | `q_lora_rank=512`, `kv_lora_rank=256` |
| MoE Routed Experts | `16` |
| MoE Shared Experts | `1` |
| Active Experts / Token | `2` |
| MoE Intermediate Size | `1536` |
| Context Window | `4096` |
| Vocabulary Size | `4311` |
| Weight State | `TRAINED_CHECKPOINT_VALIDATED` |

## Quantization Architecture

This model is quantized in native non-linear 4-bit (`IQ4_NL`):
- Linear weight matrices and 3D MoE expert tensors are quantized to 4-bit non-linear integer tables with FP16 scale factors per 32 weights.
- All layer normalization tensors (`*_norm.weight`) and router biases are preserved in full precision `FP32`.

## Local Deployment & Inference

### Ollama

Create a `Modelfile` containing:

```dockerfile
FROM ./mark_engineer_302m_iq4_nl.gguf

TEMPLATE """{{ if .System }}<|bos|>System: {{ .System }}<|eos|>
{{ end }}{{ if .Prompt }}<|bos|>User: {{ .Prompt }}<|eos|>
{{ end }}<|bos|>Assistant:
<think>
"""

PARAMETER stop "<|eos|>"
PARAMETER temperature 0.2
PARAMETER top_p 0.95
PARAMETER repeat_penalty 1.1

SYSTEM """You are Mark-Engineer-302M, a model built on native DeepSeek2 MoE and MLA architecture."""
```

Build and run:

```bash
ollama create mark-engineer-302m -f Modelfile
ollama run mark-engineer-302m
```

### llama.cpp

```bash
# Run CLI inference
llama-cli -m mark_engineer_302m_iq4_nl.gguf -p "<|bos|>System: You are an expert system.<|eos|>\n<|bos|>User: Status report?<|eos|>\n<|bos|>Assistant:\n<think>\n" -n 256

# Start local OpenAI-compatible server
llama-server -m mark_engineer_302m_iq4_nl.gguf --port 8080 -c 4096
```

## Artifact Provenance & Verification Manifest

This repository includes an auditable verification manifest: `gguf_verification_manifest.json`.

To audit the binary checksum and vocabulary on your local machine:

```bash
sha256sum mark_engineer_302m_iq4_nl.gguf && python3 -c "import gguf; r = gguf.GGUFReader('mark_engineer_302m_iq4_nl.gguf'); print('vocab:', r.get_field('deepseek2.vocab_size').parts[-1].tolist()[0])"
```
