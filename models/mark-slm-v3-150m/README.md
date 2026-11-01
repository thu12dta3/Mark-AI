# Mark-SLM-V3-150M

Sub-200M parameter edge model built on native DeepSeek2 MoE and MLA architecture with 32K vocabulary.

## Forensic Binary Specifications (GGUFReader Verified)

| Property | Value |
| :--- | :--- |
| Architecture | `deepseek2` |
| Quantization Type | `IQ4_NL` (4-bit Non-Linear Weights, FP32 Norms/Bias) |
| Container File | `mark_slm_v3_150m_iq4_nl.gguf` |
| File Size | `79.97 MiB (83.85 MB / 83,849,984 bytes)` |
| SHA-256 Checksum | `6308c8c1529b79e3965131693dbbf46937950b38f88d9791aa6d97a4b87b9a34` |
| Total Parameters | `146,899,248` physical |
| Active Parameters / Token | `54,001,968` active |
| Total Tensors | `272` ({'IQ4_NL': 196, 'F32': 76}) |
| 1D RMSNorm Precision | `100% FP32` |
| Hidden Dimension (`hidden_size`) | `576` |
| Transformer Layers | `15` |
| Attention Heads | `9` |
| Multi-Head Latent Attention (MLA) | `q_lora_rank=256`, `kv_lora_rank=128` |
| MoE Routed Experts | `16` |
| MoE Shared Experts | `1` |
| Active Experts / Token | `2` |
| MoE Intermediate Size | `256` |
| Context Window | `8192` |
| Vocabulary Size | `32032` |
| Weight State | `UN-TRAINED_ARCHITECTURAL_BASE` |

## Quantization Architecture

This model is quantized in native non-linear 4-bit (`IQ4_NL`):
- Linear weight matrices and 3D MoE expert tensors are quantized to 4-bit non-linear integer tables with FP16 scale factors per 32 weights.
- All layer normalization tensors (`*_norm.weight`) and router biases are preserved in full precision `FP32`.

## Local Deployment & Inference

### Ollama

Create a `Modelfile` containing:

```dockerfile
FROM ./mark_slm_v3_150m_iq4_nl.gguf

TEMPLATE """{{ if .System }}<|bos|>System: {{ .System }}<|eos|>
{{ end }}{{ if .Prompt }}<|bos|>User: {{ .Prompt }}<|eos|>
{{ end }}<|bos|>Assistant:
<think>
"""

PARAMETER stop "<|eos|>"
PARAMETER temperature 0.2
PARAMETER top_p 0.95
PARAMETER repeat_penalty 1.1

SYSTEM """You are Mark-SLM-V3-150M, a model built on native DeepSeek2 MoE and MLA architecture."""
```

Build and run:

```bash
ollama create mark-slm-v3-150m -f Modelfile
ollama run mark-slm-v3-150m
```

### llama.cpp

```bash
# Run CLI inference
llama-cli -m mark_slm_v3_150m_iq4_nl.gguf -p "<|bos|>System: You are an expert system.<|eos|>\n<|bos|>User: Status report?<|eos|>\n<|bos|>Assistant:\n<think>\n" -n 256

# Start local OpenAI-compatible server
llama-server -m mark_slm_v3_150m_iq4_nl.gguf --port 8080 -c 8192
```

## Artifact Provenance & Verification Manifest

This repository includes an auditable verification manifest: `gguf_verification_manifest.json`.

To audit the binary checksum and vocabulary on your local machine:

```bash
sha256sum mark_slm_v3_150m_iq4_nl.gguf && python3 -c "import gguf; r = gguf.GGUFReader('mark_slm_v3_150m_iq4_nl.gguf'); print('vocab:', r.get_field('deepseek2.vocab_size').parts[-1].tolist()[0])"
```
