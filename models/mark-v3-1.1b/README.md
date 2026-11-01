# Mark-V3-1.1B

V3 series model with 32 routed experts (4 active per token), 2 shared experts, and MLA. Total physical parameters: 1,222,464,960.

## Forensic Binary Specifications (GGUFReader Verified)

| Property | Value |
| :--- | :--- |
| Architecture | `deepseek2` |
| Quantization Type | `IQ4_NL` (4-bit Non-Linear Weights, FP32 Norms/Bias) |
| Container File | `mark_v3_1_1b_iq4_nl.gguf` |
| File Size | `656.13 MiB (688.00 MB / 688,002,048 bytes)` |
| SHA-256 Checksum | `e12c358e8144137ce4cb0546c4f00f9b0ce089c5fa354f5177f95ddf7132d84d` |
| Total Parameters | `1,222,464,960` physical |
| Active Parameters / Token | `330,651,072` active |
| Total Tensors | `327` ({'IQ4_NL': 236, 'F32': 91}) |
| 1D RMSNorm Precision | `100% FP32` |
| Hidden Dimension (`hidden_size`) | `1152` |
| Transformer Layers | `18` |
| Attention Heads | `18` |
| Multi-Head Latent Attention (MLA) | `q_lora_rank=384`, `kv_lora_rank=256` |
| MoE Routed Experts | `32` |
| MoE Shared Experts | `2` |
| Active Experts / Token | `4` |
| MoE Intermediate Size | `512` |
| Context Window | `32768` |
| Vocabulary Size | `32032` |
| Weight State | `UN-TRAINED_ARCHITECTURAL_BASE` |

## Architectural Designation Note

The designation "1.1B" is the series nomenclature. The exact physical parameter count exported in `mark_v3_1_1b_iq4_nl.gguf` with 32 routed experts (moe_intermediate_size=512, kv_lora_rank=256) is **1,222,464,960** total parameters (330,651,072 active per token), requiring **656.13 MiB (688.00 MB / 688,002,048 bytes)** in IQ4_NL container format.

## Quantization Architecture

This model is quantized in native non-linear 4-bit (`IQ4_NL`):
- Linear weight matrices and 3D MoE expert tensors are quantized to 4-bit non-linear integer tables with FP16 scale factors per 32 weights.
- All layer normalization tensors (`*_norm.weight`) and router biases are preserved in full precision `FP32`.

## Local Deployment & Inference

### Ollama

Create a `Modelfile` containing:

```dockerfile
FROM ./mark_v3_1_1b_iq4_nl.gguf

TEMPLATE """{{ if .System }}<|bos|>System: {{ .System }}<|eos|>
{{ end }}{{ if .Prompt }}<|bos|>User: {{ .Prompt }}<|eos|>
{{ end }}<|bos|>Assistant:
<think>
"""

PARAMETER stop "<|eos|>"
PARAMETER temperature 0.2
PARAMETER top_p 0.95
PARAMETER repeat_penalty 1.1

SYSTEM """You are Mark-V3-1.1B, a model built on native DeepSeek2 MoE and MLA architecture."""
```

Build and run:

```bash
ollama create mark-v3-1.1b -f Modelfile
ollama run mark-v3-1.1b
```

### llama.cpp

```bash
# Run CLI inference
llama-cli -m mark_v3_1_1b_iq4_nl.gguf -p "<|bos|>System: You are an expert system.<|eos|>\n<|bos|>User: Status report?<|eos|>\n<|bos|>Assistant:\n<think>\n" -n 256

# Start local OpenAI-compatible server
llama-server -m mark_v3_1_1b_iq4_nl.gguf --port 8080 -c 32768
```

## Artifact Provenance & Verification Manifest

This repository includes an auditable verification manifest: `gguf_verification_manifest.json`.

To audit the binary checksum and vocabulary on your local machine:

```bash
sha256sum mark_v3_1_1b_iq4_nl.gguf && python3 -c "import gguf; r = gguf.GGUFReader('mark_v3_1_1b_iq4_nl.gguf'); print('vocab:', r.get_field('deepseek2.vocab_size').parts[-1].tolist()[0])"
```
