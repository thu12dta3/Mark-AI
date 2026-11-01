# Mark-Engineer-1.5B

Foundational systems engineering model trained on low-level kernels, concurrency primitives, and systems code.

## Forensic Binary Specifications (GGUFReader Verified)

| Property | Value |
| :--- | :--- |
| Architecture | `deepseek2` |
| Quantization Type | `IQ4_NL` (4-bit Non-Linear Weights, FP32 Norms/Bias) |
| Container File | `mark_engineer_1_5b_iq4_nl.gguf` |
| File Size | `802.94 MiB (841.94 MB / 841,941,664 bytes)` |
| SHA-256 Checksum | `fa7d3061f33a20beaae5c2b85aceaf720c75b811efd8bceae0c57c57037e838e` |
| Total Parameters | `1,496,121,824` physical |
| Active Parameters / Token | `364,151,264` active |
| Total Tensors | `281` ({'IQ4_NL': 202, 'F32': 79}) |
| 1D RMSNorm Precision | `100% FP32` |
| Hidden Dimension (`hidden_size`) | `1280` |
| Transformer Layers | `16` |
| Attention Heads | `16` |
| Multi-Head Latent Attention (MLA) | `q_lora_rank=640`, `kv_lora_rank=320` |
| MoE Routed Experts | `16` |
| MoE Shared Experts | `1` |
| Active Experts / Token | `2` |
| MoE Intermediate Size | `1504` |
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
FROM ./mark_engineer_1_5b_iq4_nl.gguf

TEMPLATE """{{ if .System }}<|bos|>System: {{ .System }}<|eos|>
{{ end }}{{ if .Prompt }}<|bos|>User: {{ .Prompt }}<|eos|>
{{ end }}<|bos|>Assistant:
<think>
"""

PARAMETER stop "<|eos|>"
PARAMETER temperature 0.2
PARAMETER top_p 0.95
PARAMETER repeat_penalty 1.1

SYSTEM """You are Mark-Engineer-1.5B, a model built on native DeepSeek2 MoE and MLA architecture."""
```

Build and run:

```bash
ollama create mark-engineer-1.5b -f Modelfile
ollama run mark-engineer-1.5b
```

### llama.cpp

```bash
# Run CLI inference
llama-cli -m mark_engineer_1_5b_iq4_nl.gguf -p "<|bos|>System: You are an expert system.<|eos|>\n<|bos|>User: Status report?<|eos|>\n<|bos|>Assistant:\n<think>\n" -n 256

# Start local OpenAI-compatible server
llama-server -m mark_engineer_1_5b_iq4_nl.gguf --port 8080 -c 4096
```

## Artifact Provenance & Verification Manifest

This repository includes an auditable verification manifest: `gguf_verification_manifest.json`.

To audit the binary checksum and vocabulary on your local machine:

```bash
sha256sum mark_engineer_1_5b_iq4_nl.gguf && python3 -c "import gguf; r = gguf.GGUFReader('mark_engineer_1_5b_iq4_nl.gguf'); print('vocab:', r.get_field('deepseek2.vocab_size').parts[-1].tolist()[0])"
```
