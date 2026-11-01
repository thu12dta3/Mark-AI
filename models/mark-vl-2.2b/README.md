# Mark-VL-2.2B

Vision-language model combining DeepSeek2 language backbone with multimodal projector (mmproj).

## Forensic Binary Specifications (GGUFReader Verified)

| Property | Value |
| :--- | :--- |
| Architecture | `deepseek2` |
| Quantization Type | `IQ4_NL` (4-bit Non-Linear Weights, FP32 Norms/Bias) |
| Container File | `mark_vl_2_2b_iq4_nl.gguf` |
| File Size | `1166.72 MB` |
| SHA-256 Checksum | `073d3c00eaadfb98ef7c8b5276e355db134b4b1282ea74d8ce3d6116535cf354` |
| Total Parameters | `2.17B` (2,174,310,096 physical) |
| Active Parameters / Token | `395.50M` (395,499,216 active) |
| Total Tensors | `255` (`184` IQ4_NL, `71` FP32) |
| Hidden Dimension (`dim`) | `1280` |
| Transformer Layers | `14` |
| Attention Query Heads | `16` |
| Multi-Head Latent Attention (MLA) | `q_lora_rank=640`, `kv_lora_rank=320` |
| MoE Routed Experts | `24` |
| MoE Shared Experts | `1` |
| Active Experts / Token | `2` |
| MoE Intermediate Size | `1504` |
| Context Window | `16384` |
| Vocabulary Size | `32000` |
| Training Status | **Trained Checkpoint (Validated)** |

## Quantization Details

This model is quantized in native non-linear 4-bit (`IQ4_NL`) according to ggml specifications:
- 2D linear weight matrices and 3D MoE expert tensors are packed into 18-byte blocks for each 32 weights (2-byte FP16 scale + 16 bytes mapping to non-linear lookup table `kvalues_iq4nl = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]`).
- Normalization weights (`*_norm.weight`) and router bias (`exp_probs_b.bias`) are preserved in full precision `FP32`.

## Usage & Deployment

### llama.cpp

```bash
# Run CLI inference
llama-cli -m mark_vl_2_2b_iq4_nl.gguf -p "System architecture analysis:" -n 256

# Start local OpenAI-compatible server
llama-server -m mark_vl_2_2b_iq4_nl.gguf --port 8080 -c 16384
```

### Ollama

Create a `Modelfile` containing:

```dockerfile
FROM ./mark_vl_2_2b_iq4_nl.gguf

TEMPLATE """<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
<think>
"""

PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
PARAMETER temperature 0.2
PARAMETER top_p 0.95
```

Register and execute:

```bash
ollama create mark-vl-2.2b -f Modelfile
ollama run mark-vl-2.2b
```

### Python (llama-cpp-python)

```python
from llama_cpp import Llama

llm = Llama(
    model_path="mark_vl_2_2b_iq4_nl.gguf",
    n_ctx=16384,
    n_threads=8,
)

response = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are a concise software engineering assistant."},
        {"role": "user", "content": "Explain lock-free atomic CAS loops."},
    ]
)
print(response["choices"][0]["message"]["content"])
```
