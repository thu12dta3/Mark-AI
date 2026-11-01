# Mark-Experimental-V3-8B

V3 architecture 8B class research model with 32 routed experts, 2 shared experts, and MLA.

## Forensic Binary Specifications (GGUFReader Verified)

| Property | Value |
| :--- | :--- |
| Architecture | `deepseek2` |
| Quantization Type | `IQ4_NL` (4-bit Non-Linear Weights, FP32 Norms/Bias) |
| Container File | `mark_experimental_v3_8b_iq4_nl.gguf` |
| File Size | `4382.55 MiB (4595.44 MB / 4,595,438,848 bytes)` |
| SHA-256 Checksum | `792e127a71f62145f87c3775c4a03a0dc7bb68b8ee355acb74602348cf748b4a` |
| Total Parameters | `8,168,001,024` physical |
| Active Parameters / Token | `2,002,374,144` active |
| Total Tensors | `579` ({'IQ4_NL': 418, 'F32': 161}) |
| 1D RMSNorm Precision | `100% FP32` |
| Hidden Dimension (`hidden_size`) | `2560` |
| Transformer Layers | `32` |
| Attention Heads | `32` |
| Multi-Head Latent Attention (MLA) | `q_lora_rank=1024`, `kv_lora_rank=512` |
| MoE Routed Experts | `32` |
| MoE Shared Experts | `2` |
| Active Experts / Token | `4` |
| MoE Intermediate Size | `896` |
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
FROM ./mark_experimental_v3_8b_iq4_nl.gguf

TEMPLATE """{{ if .System }}<|bos|>System: {{ .System }}<|eos|>
{{ end }}{{ if .Prompt }}<|bos|>User: {{ .Prompt }}<|eos|>
{{ end }}<|bos|>Assistant:
<think>
"""

PARAMETER stop "<|eos|>"
PARAMETER temperature 0.2
PARAMETER top_p 0.95
PARAMETER repeat_penalty 1.1

SYSTEM """You are Mark-Experimental-V3-8B, a model built on native DeepSeek2 MoE and MLA architecture."""
```

Build and run:

```bash
ollama create mark-experimental-v3-8b -f Modelfile
ollama run mark-experimental-v3-8b
```

### llama.cpp

```bash
# Run CLI inference
llama-cli -m mark_experimental_v3_8b_iq4_nl.gguf -p "<|bos|>System: You are an expert system.<|eos|>\n<|bos|>User: Status report?<|eos|>\n<|bos|>Assistant:\n<think>\n" -n 256

# Start local OpenAI-compatible server
llama-server -m mark_experimental_v3_8b_iq4_nl.gguf --port 8080 -c 32768
```

## Artifact Provenance & Verification Manifest

This repository includes an auditable verification manifest: `gguf_verification_manifest.json`.

To audit the binary checksum and vocabulary on your local machine:

```bash
sha256sum mark_experimental_v3_8b_iq4_nl.gguf && python3 -c "import gguf; r = gguf.GGUFReader('mark_experimental_v3_8b_iq4_nl.gguf'); print('vocab:', r.get_field('deepseek2.vocab_size').parts[-1].tolist()[0])"
```
