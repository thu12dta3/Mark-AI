# Mark-VL-2.2B

Multimodal vision-language model pairing a native DeepSeek2 MoE backbone with a specialized vision projector adapter.

## Forensic Binary Specifications (GGUFReader Verified)

| Property | Value |
| :--- | :--- |
| Architecture | `deepseek2` |
| Quantization Type | `IQ4_NL` (4-bit Non-Linear Weights, FP32 Norms/Bias) |
| Container File | `mark_vl_2_2b_iq4_nl.gguf` |
| File Size | `1166.72 MiB (1223.40 MB / 1,223,397,152 bytes)` |
| SHA-256 Checksum | `073d3c00eaadfb98ef7c8b5276e355db134b4b1282ea74d8ce3d6116535cf354` |
| Total Parameters | `2,174,310,096` physical |
| Active Parameters / Token | `395,499,216` active |
| Total Tensors | `255` ({'IQ4_NL': 184, 'F32': 71}) |
| 1D RMSNorm Precision | `100% FP32` |
| Hidden Dimension (`hidden_size`) | `1280` |
| Transformer Layers | `14` |
| Attention Heads | `16` |
| Multi-Head Latent Attention (MLA) | `q_lora_rank=640`, `kv_lora_rank=320` |
| MoE Routed Experts | `24` |
| MoE Shared Experts | `1` |
| Active Experts / Token | `2` |
| MoE Intermediate Size | `1504` |
| Context Window | `16384` |
| Vocabulary Size | `32000` |
| Weight State | `TRAINED_CHECKPOINT_VALIDATED` |

## Repository Binary Artifacts

This repository provides three distinct binary artifacts for multimodal inference:
1. `mark_vl_2_2b_iq4_nl.gguf`: The primary language model backbone quantized in IQ4_NL (DeepSeek2 MoE + MLA, 14 layers, 24 routed experts, 1 shared expert).
2. `mmproj_mark_vl_2_2b.gguf`: Multimodal vision projector adapter responsible for processing spatial image patches into the text embedding space.
3. `mark_vl_2_2b_q4_k_m.gguf`: Legacy Q4_K_M quantization checkpoint preserved for compatibility with older runtimes.

## Quantization Architecture

This model is quantized in native non-linear 4-bit (`IQ4_NL`):
- Linear weight matrices and 3D MoE expert tensors are quantized to 4-bit non-linear integer tables with FP16 scale factors per 32 weights.
- All layer normalization tensors (`*_norm.weight`) and router biases are preserved in full precision `FP32`.

## Local Deployment & Inference

### Ollama

Create a `Modelfile` containing:

```dockerfile
FROM ./mark_vl_2_2b_iq4_nl.gguf
ADAPTER ./mmproj_mark_vl_2_2b.gguf

TEMPLATE """{{ if .System }}<|bos|>System: {{ .System }}<|eos|>
{{ end }}{{ if .Prompt }}<|bos|>User: {{ .Prompt }}<|eos|>
{{ end }}<|bos|>Assistant:
<think>
"""

PARAMETER stop "<|eos|>"
PARAMETER temperature 0.2
PARAMETER top_p 0.95
PARAMETER repeat_penalty 1.1

SYSTEM """You are Mark-VL-2.2B, a model built on native DeepSeek2 MoE and MLA architecture."""
```

Build and run:

```bash
ollama create mark-vl-2.2b -f Modelfile
ollama run mark-vl-2.2b
```

### llama.cpp

```bash
# Run CLI inference
llama-cli -m mark_vl_2_2b_iq4_nl.gguf -p "<|bos|>System: You are an expert system.<|eos|>\n<|bos|>User: Status report?<|eos|>\n<|bos|>Assistant:\n<think>\n" -n 256

# Start local OpenAI-compatible server
llama-server -m mark_vl_2_2b_iq4_nl.gguf --port 8080 -c 16384
```

## Artifact Provenance & Verification Manifest

This repository includes an auditable verification manifest: `gguf_verification_manifest.json`.

To audit the binary checksum and vocabulary on your local machine:

```bash
sha256sum mark_vl_2_2b_iq4_nl.gguf && python3 -c "import gguf; r = gguf.GGUFReader('mark_vl_2_2b_iq4_nl.gguf'); print('vocab:', r.get_field('deepseek2.vocab_size').parts[-1].tolist()[0])"
```
