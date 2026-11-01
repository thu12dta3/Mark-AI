# Mark-SLM-V2-302M

Architectural base container featuring 16 layers of DeepSeek2 MoE and MLA for small language model exploration.

## Forensic Binary Specifications (GGUFReader Verified)

| Property | Value |
| :--- | :--- |
| Architecture | `deepseek2` |
| Quantization Type | `IQ4_NL` (4-bit Non-Linear Weights, FP32 Norms/Bias) |
| Container File | `mark_slm_v2_302m_iq4_nl.gguf` |
| File Size | `176.84 MB` |
| SHA-256 Checksum | `388237d49ea776c955851e575bb15e30a63b3e8218fda3d23cd690188bbb6b3a` |
| Total Parameters | `329.13M` (329,132,000 physical) |
| Active Parameters / Token | `155.72M` (155,723,744 active) |
| Total Tensors | `281` (`202` IQ4_NL, `79` FP32) |
| Hidden Dimension (`dim`) | `768` |
| Transformer Layers | `16` |
| Attention Query Heads | `12` |
| Multi-Head Latent Attention (MLA) | `q_lora_rank=384`, `kv_lora_rank=192` |
| MoE Routed Experts | `16` |
| MoE Shared Experts | `1` |
| Active Experts / Token | `2` |
| MoE Intermediate Size | `384` |
| Context Window | `4096` |
| Vocabulary Size | `49152` |
| Training Status | **Architectural Base / Untrained / Weight Initialization via He/Xavier + Hadamard Fusion (Ready for Pre-training or Fine-Tuning)** |

## Quantization Details

This model is quantized in native non-linear 4-bit (`IQ4_NL`) according to ggml specifications:
- 2D linear weight matrices and 3D MoE expert tensors are packed into 18-byte blocks for each 32 weights (2-byte FP16 scale + 16 bytes mapping to non-linear lookup table `kvalues_iq4nl = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]`).
- Normalization weights (`*_norm.weight`) and router bias (`exp_probs_b.bias`) are preserved in full precision `FP32`.

## Usage & Deployment

### llama.cpp

```bash
# Run CLI inference
llama-cli -m mark_slm_v2_302m_iq4_nl.gguf -p "System architecture analysis:" -n 256

# Start local OpenAI-compatible server
llama-server -m mark_slm_v2_302m_iq4_nl.gguf --port 8080 -c 4096
```

### Ollama

Create a `Modelfile` containing:

```dockerfile
FROM ./mark_slm_v2_302m_iq4_nl.gguf

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
ollama create mark-slm-v2-302m -f Modelfile
ollama run mark-slm-v2-302m
```

### Python (llama-cpp-python)

```python
from llama_cpp import Llama

llm = Llama(
    model_path="mark_slm_v2_302m_iq4_nl.gguf",
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
