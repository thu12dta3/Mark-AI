# Mark-Experimental-V3-8B

8.17B parameter V3 flagship base featuring 32 layers, 32 routed experts, 2 shared experts, and top-4 routing.

## Forensic Binary Specifications (GGUFReader Verified)

| Property | Value |
| :--- | :--- |
| Architecture | `deepseek2` |
| Quantization Type | `IQ4_NL` (4-bit Non-Linear Weights, FP32 Norms/Bias) |
| Container File | `mark_experimental_v3_8b_iq4_nl.gguf` |
| File Size | `4382.55 MB` |
| SHA-256 Checksum | `792e127a71f62145f87c3775c4a03a0dc7bb68b8ee355acb74602348cf748b4a` |
| Total Parameters | `8.17B` (8,168,001,024 physical) |
| Active Parameters / Token | `2.00B` (2,002,374,144 active) |
| Total Tensors | `579` (`418` IQ4_NL, `161` FP32) |
| Hidden Dimension (`dim`) | `2560` |
| Transformer Layers | `32` |
| Attention Query Heads | `32` |
| Multi-Head Latent Attention (MLA) | `q_lora_rank=1024`, `kv_lora_rank=512` |
| MoE Routed Experts | `32` |
| MoE Shared Experts | `2` |
| Active Experts / Token | `4` |
| MoE Intermediate Size | `896` |
| Context Window | `32768` |
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
llama-cli -m mark_experimental_v3_8b_iq4_nl.gguf -p "System architecture analysis:" -n 256

# Start local OpenAI-compatible server
llama-server -m mark_experimental_v3_8b_iq4_nl.gguf --port 8080 -c 32768
```

### Ollama

Create a `Modelfile` containing:

```dockerfile
FROM ./mark_experimental_v3_8b_iq4_nl.gguf

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
ollama create mark-experimental-v3-8b -f Modelfile
ollama run mark-experimental-v3-8b
```

### Python (llama-cpp-python)

```python
from llama_cpp import Llama

llm = Llama(
    model_path="mark_experimental_v3_8b_iq4_nl.gguf",
    n_ctx=32768,
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
