# Mark-SLM-V2-302M

Compact small language model with an expanded 49,152 token vocabulary and DeepSeek2 MoE/MLA topology.

## Forensic Binary Specifications (GGUFReader Verified)

| Property | Value |
| :--- | :--- |
| Architecture | `deepseek2` |
| Quantization Type | `IQ4_NL` (4-bit Non-Linear Weights, FP32 Norms/Bias) |
| Container File | `mark_slm_v2_302m_iq4_nl.gguf` |
| File Size | `176.84 MiB (185.43 MB / 185,430,656 bytes)` |
| SHA-256 Checksum | `388237d49ea776c955851e575bb15e30a63b3e8218fda3d23cd690188bbb6b3a` |
| Total Parameters | `329,132,000` physical |
| Active Parameters / Token | `155,723,744` active |
| Total Tensors | `281` ({'IQ4_NL': 202, 'F32': 79}) |
| 1D RMSNorm Precision | `100% FP32` |
| Hidden Dimension (`hidden_size`) | `768` |
| Transformer Layers | `16` |
| Attention Heads | `12` |
| Multi-Head Latent Attention (MLA) | `q_lora_rank=384`, `kv_lora_rank=192` |
| MoE Routed Experts | `16` |
| MoE Shared Experts | `1` |
| Active Experts / Token | `2` |
| MoE Intermediate Size | `384` |
| Context Window | `4096` |
| Vocabulary Size | `49152` |
| Weight State | `TRAINED_CHECKPOINT_VALIDATED` |

## Quantization Architecture

This model is quantized in native non-linear 4-bit (`IQ4_NL`):
- Linear weight matrices and 3D MoE expert tensors are quantized to 4-bit non-linear integer tables with FP16 scale factors per 32 weights.
- All layer normalization tensors (`*_norm.weight`) and router biases are preserved in full precision `FP32`.

## Local Deployment & Inference

### Ollama

Create a `Modelfile` containing:

```dockerfile
FROM ./mark_slm_v2_302m_iq4_nl.gguf

TEMPLATE """{{ if .System }}<|bos|>System: {{ .System }}<|eos|>
{{ end }}{{ if .Prompt }}<|bos|>User: {{ .Prompt }}<|eos|>
{{ end }}<|bos|>Assistant:
<think>
"""

PARAMETER stop "<|eos|>"
PARAMETER temperature 0.2
PARAMETER top_p 0.95
PARAMETER repeat_penalty 1.1

SYSTEM """You are Mark-SLM-V2-302M, a model built on native DeepSeek2 MoE and MLA architecture."""
```

Build and run:

```bash
ollama create mark-slm-v2-302m -f Modelfile
ollama run mark-slm-v2-302m
```

### llama.cpp

```bash
# Run CLI inference
llama-cli -m mark_slm_v2_302m_iq4_nl.gguf -p "<|bos|>System: You are an expert system.<|eos|>\n<|bos|>User: Status report?<|eos|>\n<|bos|>Assistant:\n<think>\n" -n 256

# Start local OpenAI-compatible server
llama-server -m mark_slm_v2_302m_iq4_nl.gguf --port 8080 -c 4096
```

## Artifact Provenance & Verification Manifest

This repository includes an auditable verification manifest: `gguf_verification_manifest.json`.

To audit the binary checksum and vocabulary on your local machine:

```bash
sha256sum mark_slm_v2_302m_iq4_nl.gguf && python3 -c "import gguf; r = gguf.GGUFReader('mark_slm_v2_302m_iq4_nl.gguf'); print('vocab:', r.get_field('deepseek2.vocab_size').parts[-1].tolist()[0])"
```
