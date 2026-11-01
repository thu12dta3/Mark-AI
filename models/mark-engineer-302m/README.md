# Mark-Engineer-302M

Compact edge testbed featuring DeepSeek2 native MoE (16 routed, 1 shared, top-2 active) and Multi-Head Latent Attention (MLA).

## Forensic Binary Specifications (GGUFReader Verified)

| Property | Value |
| :--- | :--- |
| Architecture | `deepseek2` |
| Quantization Type | `IQ4_NL` (4-bit Non-Linear Weights, FP32 Norms/Bias) |
| Container File | `mark_engineer_302m_iq4_nl.gguf` |
| File Size | `135.12 MB` |
| SHA-256 Checksum | `5e9d7d6feb9bc0b6ad5da98e444e3e083a1936e58ad6091230ee674d1ef714b0` |
| Total Parameters | `251.45M` (251,452,416 physical) |
| Active Parameters / Token | `119.33M` (119,331,840 active) |
| Total Tensors | `115` (`82` IQ4_NL, `33` FP32) |
| Hidden Dimension (`dim`) | `1024` |
| Transformer Layers | `8` |
| Attention Query Heads | `16` |
| Multi-Head Latent Attention (MLA) | `q_lora_rank=512`, `kv_lora_rank=256` |
| MoE Routed Experts | `16` |
| MoE Shared Experts | `1` |
| Active Experts / Token | `2` |
| MoE Intermediate Size | `1536` |
| Context Window | `4096` |
| Vocabulary Size | `4311` |
| Training Status | **Trained Checkpoint (Validated)** |

## Quantization Details

This model is quantized in native non-linear 4-bit (`IQ4_NL`) according to ggml specifications:
- 2D linear weight matrices and 3D MoE expert tensors are packed into 18-byte blocks for each 32 weights (2-byte FP16 scale + 16 bytes mapping to non-linear lookup table `kvalues_iq4nl = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]`).
- Normalization weights (`*_norm.weight`) and router bias (`exp_probs_b.bias`) are preserved in full precision `FP32`.

## Usage & Deployment

### llama.cpp

```bash
# Run CLI inference
llama-cli -m mark_engineer_302m_iq4_nl.gguf -p "System architecture analysis:" -n 256

# Start local OpenAI-compatible server
llama-server -m mark_engineer_302m_iq4_nl.gguf --port 8080 -c 4096
```

### Ollama

Create a `Modelfile` containing:

```dockerfile
FROM ./mark_engineer_302m_iq4_nl.gguf

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
ollama create mark-engineer-302m -f Modelfile
ollama run mark-engineer-302m
```

### Python (llama-cpp-python)

```python
from llama_cpp import Llama

llm = Llama(
    model_path="mark_engineer_302m_iq4_nl.gguf",
    n_ctx=4096,
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
