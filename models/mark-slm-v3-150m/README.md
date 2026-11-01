# Mark-SLM-V3-150M

Mark V3 generation micro-model with Hadamard-fused projections and DeepSeek2 MoE routing.

## Forensic Binary Specifications (GGUFReader Verified)

| Property | Value |
| :--- | :--- |
| Architecture | `deepseek2` |
| Quantization Type | `IQ4_NL` (4-bit Non-Linear Weights, FP32 Norms/Bias) |
| Container File | `mark_slm_v3_150m_iq4_nl.gguf` |
| File Size | `88.94 MB` |
| SHA-256 Checksum | `c10b0cb0e1bdb22c9178da7211fc47cbf485f54f29bf06bcd0c0ec23a535b6c6` |
| Total Parameters | `165.35M` (165,349,680 physical) |
| Active Parameters / Token | `72.45M` (72,452,400 active) |
| Total Tensors | `273` (`197` IQ4_NL, `76` FP32) |
| Hidden Dimension (`dim`) | `576` |
| Transformer Layers | `15` |
| Attention Query Heads | `9` |
| Multi-Head Latent Attention (MLA) | `q_lora_rank=256`, `kv_lora_rank=128` |
| MoE Routed Experts | `16` |
| MoE Shared Experts | `1` |
| Active Experts / Token | `2` |
| MoE Intermediate Size | `256` |
| Context Window | `8192` |
| Vocabulary Size | `32032` |
| Training Status | **Architectural Base / Untrained / Weight Initialization via He/Xavier + Hadamard Fusion (Ready for Pre-training or Fine-Tuning)** |

## Quantization Details

This model is quantized in native non-linear 4-bit (`IQ4_NL`) according to ggml specifications:
- 2D linear weight matrices and 3D MoE expert tensors are packed into 18-byte blocks for each 32 weights (2-byte FP16 scale + 16 bytes mapping to non-linear lookup table `kvalues_iq4nl = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]`).
- Normalization weights (`*_norm.weight`) and router bias (`exp_probs_b.bias`) are preserved in full precision `FP32`.

## Usage & Deployment

### llama.cpp

```bash
# Run CLI inference
llama-cli -m mark_slm_v3_150m_iq4_nl.gguf -p "System architecture analysis:" -n 256

# Start local OpenAI-compatible server
llama-server -m mark_slm_v3_150m_iq4_nl.gguf --port 8080 -c 8192
```

### Ollama

Create a `Modelfile` containing:

```dockerfile
FROM ./mark_slm_v3_150m_iq4_nl.gguf

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
ollama create mark-slm-v3-150m -f Modelfile
ollama run mark-slm-v3-150m
```

### Python (llama-cpp-python)

```python
from llama_cpp import Llama

llm = Llama(
    model_path="mark_slm_v3_150m_iq4_nl.gguf",
    n_ctx=8192,
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
