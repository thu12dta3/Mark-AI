# Mark-Engineer-V2-8B

8.26B parameter DeepSeek2 MoE architectural base with 32 layers and 24 routed experts.

## Forensic Binary Specifications (GGUFReader Verified)

| Property | Value |
| :--- | :--- |
| Architecture | `deepseek2` |
| Quantization Type | `IQ4_NL` (4-bit Non-Linear Weights, FP32 Norms/Bias) |
| Container File | `mark_engineer_v2_8b_iq4_nl.gguf` |
| File Size | `4431.26 MB` |
| SHA-256 Checksum | `28c014b5b903f7e21ee9777ef743030c7a068111c923baeb33d5f05685e05400` |
| Total Parameters | `8.26B` (8,258,809,040 physical) |
| Active Parameters / Token | `1.77B` (1,770,745,040 active) |
| Total Tensors | `569` (`410` IQ4_NL, `159` FP32) |
| Hidden Dimension (`dim`) | `2560` |
| Transformer Layers | `32` |
| Attention Query Heads | `32` |
| Multi-Head Latent Attention (MLA) | `q_lora_rank=1024`, `kv_lora_rank=512` |
| MoE Routed Experts | `24` |
| MoE Shared Experts | `1` |
| Active Experts / Token | `2` |
| MoE Intermediate Size | `1280` |
| Context Window | `8192` |
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
llama-cli -m mark_engineer_v2_8b_iq4_nl.gguf -p "System architecture analysis:" -n 256

# Start local OpenAI-compatible server
llama-server -m mark_engineer_v2_8b_iq4_nl.gguf --port 8080 -c 8192
```

### Ollama

Create a `Modelfile` containing:

```dockerfile
FROM ./mark_engineer_v2_8b_iq4_nl.gguf

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
ollama create mark-engineer-v2-8b -f Modelfile
ollama run mark-engineer-v2-8b
```

### Python (llama-cpp-python)

```python
from llama_cpp import Llama

llm = Llama(
    model_path="mark_engineer_v2_8b_iq4_nl.gguf",
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
