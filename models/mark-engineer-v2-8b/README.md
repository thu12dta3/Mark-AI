# Mark-Engineer-V2-8B

Flagship 8B architectural foundation featuring 24 MoE experts, Multi-Head Latent Attention, and 49K vocabulary.

## Forensic Binary Specifications (GGUFReader Verified)

| Property | Value |
| :--- | :--- |
| Architecture | `deepseek2` |
| Quantization Type | `IQ4_NL` (4-bit Non-Linear Weights, FP32 Norms/Bias) |
| Container File | `mark_engineer_v2_8b_iq4_nl.gguf` |
| File Size | `4431.26 MiB (4646.52 MB / 4,646,516,576 bytes)` |
| SHA-256 Checksum | `28c014b5b903f7e21ee9777ef743030c7a068111c923baeb33d5f05685e05400` |
| Total Parameters | `8,258,809,040` physical |
| Active Parameters / Token | `1,770,745,040` active |
| Total Tensors | `569` ({'IQ4_NL': 410, 'F32': 159}) |
| 1D RMSNorm Precision | `100% FP32` |
| Hidden Dimension (`hidden_size`) | `2560` |
| Transformer Layers | `32` |
| Attention Heads | `32` |
| Multi-Head Latent Attention (MLA) | `q_lora_rank=1024`, `kv_lora_rank=512` |
| MoE Routed Experts | `24` |
| MoE Shared Experts | `1` |
| Active Experts / Token | `2` |
| MoE Intermediate Size | `1280` |
| Context Window | `8192` |
| Vocabulary Size | `49152` |
| Weight State | `UN-TRAINED_ARCHITECTURAL_BASE` |

## Quantization Architecture

This model is quantized in native non-linear 4-bit (`IQ4_NL`):
- Linear weight matrices and 3D MoE expert tensors are quantized to 4-bit non-linear integer tables with FP16 scale factors per 32 weights.
- All layer normalization tensors (`*_norm.weight`) and router biases are preserved in full precision `FP32`.

## Local Deployment & Inference

### Ollama

Create a `Modelfile` containing:

```dockerfile
FROM ./mark_engineer_v2_8b_iq4_nl.gguf

TEMPLATE """{{ if .System }}<|bos|>System: {{ .System }}<|eos|>
{{ end }}{{ if .Prompt }}<|bos|>User: {{ .Prompt }}<|eos|>
{{ end }}<|bos|>Assistant:
<think>
"""

PARAMETER stop "<|eos|>"
PARAMETER temperature 0.2
PARAMETER top_p 0.95
PARAMETER repeat_penalty 1.1

SYSTEM """You are Mark-Engineer-V2-8B, a model built on native DeepSeek2 MoE and MLA architecture."""
```

Build and run:

```bash
ollama create mark-engineer-v2-8b -f Modelfile
ollama run mark-engineer-v2-8b
```

### llama.cpp

```bash
# Run CLI inference
llama-cli -m mark_engineer_v2_8b_iq4_nl.gguf -p "<|bos|>System: You are an expert system.<|eos|>\n<|bos|>User: Status report?<|eos|>\n<|bos|>Assistant:\n<think>\n" -n 256

# Start local OpenAI-compatible server
llama-server -m mark_engineer_v2_8b_iq4_nl.gguf --port 8080 -c 8192
```

## Artifact Provenance & Verification Manifest

This repository includes an auditable verification manifest: `gguf_verification_manifest.json`.

To audit the binary checksum and vocabulary on your local machine:

```bash
sha256sum mark_engineer_v2_8b_iq4_nl.gguf && python3 -c "import gguf; r = gguf.GGUFReader('mark_engineer_v2_8b_iq4_nl.gguf'); print('vocab:', r.get_field('deepseek2.vocab_size').parts[-1].tolist()[0])"
```
