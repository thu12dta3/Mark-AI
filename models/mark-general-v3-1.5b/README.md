# Mark-General-V3-1.5B

V3 architecture 1.5B class general reasoning model with 32 routed experts and MLA.

## Forensic Binary Specifications (GGUFReader Verified)

| Property | Value |
| :--- | :--- |
| Architecture | `deepseek2` |
| Quantization Type | `IQ4_NL` (4-bit Non-Linear Weights, FP32 Norms/Bias) |
| Container File | `mark_general_v3_1_5b_iq4_nl.gguf` |
| File Size | `816.03 MiB (855.67 MB / 855,669,248 bytes)` |
| SHA-256 Checksum | `7677ebd44467eecba594c2f41e8a9031590c3fdbb692089011eee7578c87c55f` |
| Total Parameters | `1,520,496,960` physical |
| Active Parameters / Token | `405,729,600` active |
| Total Tensors | `327` ({'IQ4_NL': 236, 'F32': 91}) |
| 1D RMSNorm Precision | `100% FP32` |
| Hidden Dimension (`hidden_size`) | `1280` |
| Transformer Layers | `18` |
| Attention Heads | `20` |
| Multi-Head Latent Attention (MLA) | `q_lora_rank=512`, `kv_lora_rank=256` |
| MoE Routed Experts | `32` |
| MoE Shared Experts | `2` |
| Active Experts / Token | `4` |
| MoE Intermediate Size | `576` |
| Context Window | `32768` |
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
FROM ./mark_general_v3_1_5b_iq4_nl.gguf

TEMPLATE """{{ if .System }}<|bos|>System: {{ .System }}<|eos|>
{{ end }}{{ if .Prompt }}<|bos|>User: {{ .Prompt }}<|eos|>
{{ end }}<|bos|>Assistant:
<think>
"""

PARAMETER stop "<|eos|>"
PARAMETER temperature 0.2
PARAMETER top_p 0.95
PARAMETER repeat_penalty 1.1

SYSTEM """You are Mark-General-V3-1.5B, a model built on native DeepSeek2 MoE and MLA architecture."""
```

Build and run:

```bash
ollama create mark-general-v3-1.5b -f Modelfile
ollama run mark-general-v3-1.5b
```

### llama.cpp

```bash
# Run CLI inference
llama-cli -m mark_general_v3_1_5b_iq4_nl.gguf -p "<|bos|>System: You are an expert system.<|eos|>\n<|bos|>User: Status report?<|eos|>\n<|bos|>Assistant:\n<think>\n" -n 256

# Start local OpenAI-compatible server
llama-server -m mark_general_v3_1_5b_iq4_nl.gguf --port 8080 -c 32768
```

## Artifact Provenance & Verification Manifest

This repository includes an auditable verification manifest: `gguf_verification_manifest.json`.

To audit the binary checksum and vocabulary on your local machine:

```bash
sha256sum mark_general_v3_1_5b_iq4_nl.gguf && python3 -c "import gguf; r = gguf.GGUFReader('mark_general_v3_1_5b_iq4_nl.gguf'); print('vocab:', r.get_field('deepseek2.vocab_size').parts[-1].tolist()[0])"
```
