# Mark 0.1 -- Core Neural Architecture

<div align="center">

![License: AGPLv3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C.svg)
![Hardware](https://img.shields.io/badge/Compute-NVIDIA_A100-76B900.svg)
![Target](https://img.shields.io/badge/Inference-Local_GPU_8GB_VRAM-0052CC.svg)
![Params](https://img.shields.io/badge/Parameters-~3.0B_Dense-green.svg)

**The central cognitive engine of the MARK-PUBLIC distributed ecosystem.**  
*A foundational system assistant engineered from scratch for systems architecture, systems programming, and autonomous reasoning.*

</div>

---

## Executive Overview

Mark 0.1 is not a conversational chatbot. It is a foundational causal language model designed specifically to act as the primary intelligence layer within the MARK-PUBLIC distributed infrastructure:
- Python Cognitive Core (`src/mark/`)
- High-Performance Rust Transport and Security Layer (`src-rust/mark_core/`)
- Android / WearOS Clients

### Hardware Profile
- Accelerator Target: NVIDIA A100 Tensor Core GPUs utilizing bfloat16 mixed precision, FlashAttention-2, fused CUDA operators, and gradient checkpointing.
- Deterministic 8GB Local Inference: Engineered to run on consumer hardware with strict <= 7.5 GB VRAM ceilings using 4-bit NormalFloat (nf4) double quantization and grouped-query KV caching.

---

## Architectural Specification

```
[ Input Tokens ]
       |
       v
[ Token Embeddings (V=49,152, D=2560) ] --- (Tied Weights with LM Head)
       |
       v
 +--------------------------------------------------------+
 | Transformer Layer (x32)                                |
 |                                                        |
 |   Input  ---> [ RMSNorm ]                              |
 |                |                                       |
 |                v                                       |
 |              [ Grouped-Query Attention (GQA 4:1) ]     |
 |              (32 Q Heads, 8 KV Heads, FlashAttn-2)     |
 |              (NTK-Aware Dynamic RoPE: 8k ---> 32k)     |
 |                |                                       |
 |                v                                       |
 |              ( + Residual Connection )                 |
 |                |                                       |
 |                v                                       |
 |              [ RMSNorm ]                               |
 |                |                                       |
 |                v                                       |
 |              [ SwiGLU FFN ] (Intermediate = 6912)      |
 |                |                                       |
 |                v                                       |
 |              ( + Residual Connection )                 |
 +--------------------------------------------------------+
       |
       v
 [ Final RMSNorm ]
       |
       v
 [ Linear LM Head (V=49,152) ] ---> Next-Token Probabilities
```

### Parameter Count Breakdown (~2.95B)
- Hidden Size (d_model): 2560
- Layers: 32
- Query Attention Heads (H_q): 32 (head_dim = 80)
- KV Heads (H_kv): 8 (Grouped-Query Attention ratio 4:1, reducing runtime KV cache memory by 75%)
- SwiGLU Dimension: 6912 (~8/3 * d_model)
- Rotary Position Embeddings (RoPE): Base frequency theta = 500,000 with dynamic NTK extrapolation for context extension up to 32,768 tokens.
- Normalization: Root Mean Square Layer Normalization (RMSNorm, eps = 1e-5) evaluated in float32.

### Deterministic 8GB VRAM Budget (Local Production)
Total VRAM Allocation = 1.68 GB (4-bit NF4 Weights) + 0.34 GB (KV Cache @ 8k FP16) + 0.65 GB (FlashAttn-2 Buffers) = 2.67 GB <= 7.5 GB.

Even scaling to 32,768 tokens, the KV Cache requires only ~1.34 GB, capping dynamic VRAM below 3.7 GB and preserving over 4 GB of headroom for host OS rendering and runtime buffers.

---

## Cognitive Mechanism: Latent Thinking and Native Self-RAG

Mark 0.1 is conditioned to formulate explicit logical reasoning inside `<think>...</think>` tags and actively retrieve project context.

### The Self-RAG State Machine
1. Introspection: The model formulates latent reflection:
   ```
   <think>
   WearOS link bridge is dropping packets. Inspecting stream lifecycle in relay.rs before modifying the lock.
   </think>
   ```
2. Autonomous Retrieval: The model halts generation and emits:
   ```
   [RETRIEVE] src-rust/mark_link/src/relay.rs [/RETRIEVE]
   ```
3. External Injection: The runtime interceptor queries the project index and injects context between boundary tokens:
   ```
   <|rag_context|>
   pub async fn handle_wear_stream(...) { ... }
   <|/rag_context|>
   [EVALUATE]
   ```
4. Self-Judgment: The model validates relevance by emitting `[ACCEPT]` or `[REJECT]` before generating code.

---

## Repository Structure

```
Mark-AI/
├── data/
│   ├── raw/
│   └── synthetic_dpo/
├── export/
│   └── quantize_4bit.py
├── src/
│   ├── model/
│   │   ├── mark_arch.py
│   │   └── self_rag.py
│   ├── tokenizer/
│   │   └── build_bpe.py
│   └── training/
│       ├── reasoning_loss.py
│       └── train_cloud.py
├── inference_8gb.py
├── model.py
├── requirements.txt
├── tokenizer_setup.py
└── train_reasoning.py
```

---

## Local Inference on Consumer 8GB GPU

```python
from tokenizers import Tokenizer
from inference_8gb import load_mark_for_8gb_vram, MarkInferenceEngine

# Loads weights into 4-bit NormalFloat (NF4) under strict 7.5 GB VRAM ceiling
tokenizer = Tokenizer.from_file("data/tokenizer/mark_bpe_tokenizer.json")
model, kv_cache, stats = load_mark_for_8gb_vram(
    checkpoint_path="checkpoints/mark_0_1_step_001000.pt",
    max_context_tokens=8192
)

engine = MarkInferenceEngine(model, tokenizer, kv_cache)
response = engine.generate(
    prompt="Refactor the QUIC relay channel in daemon.rs to avoid deadlock on disconnect.",
    max_new_tokens=512,
    temperature=0.1
)
print(response)
```

---

## Governance

This repository contains foundational neural architectures and model weights for the ecosystem MARK-PUBLIC.

> Notice on Pull Requests:
> This repository is maintained as an authoritative upstream research project. External Pull Requests are disabled and will not be accepted. For bug reports or architectural feedback, please open an Issue.

---

## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0) -- identical to the core MARK-PUBLIC ecosystem. See the [LICENSE](LICENSE) file for complete terms.
