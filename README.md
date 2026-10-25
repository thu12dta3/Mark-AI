# Mark Primal -- Foundational Neural Architecture

<div align="center">

![License: AGPLv3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C.svg)
![Hardware](https://img.shields.io/badge/Compute-NVIDIA_A100-76B900.svg)
![Target](https://img.shields.io/badge/Inference-Local_GPU_8GB_VRAM-0052CC.svg)
![Architecture](https://img.shields.io/badge/Architecture-MLA_%2B_MoE_%2B_Vision-green.svg)

**The central cognitive, polyglot engineering, and multimodal engine of the MARK-PUBLIC ecosystem.**

</div>

---

## Model Hierarchy

| Variant | Total Parameters | Active Params / Token | Target & Purpose |
| :--- | :---: | :---: | :--- |
| **Mark Primal** *(formerly Mark 0.1)* | ~2.98B | ~2.35B | Foundation Multimodal & Polyglot Base (~2.95B dense/MoE). |
| **Mark Engineer 1.5B** | **1.4995B (~1.50B)** | **~315M - 440M** | High-capacity systems engineering, AppSec P0, and literature reasoning. |
| **Mark Engineer 300M** | **302.02M** | **~110M - 165M** | Preserved ultra-lightweight edge production checkpoint (163 MB IQ4_NL). |

---

## Executive Overview

Mark Primal is a high-density causal language and vision-language model architected from the ground up to serve as the autonomous intelligence layer of the MARK-PUBLIC distributed ecosystem:
- Systems Architecture & Polyglot Engineering: Native mastery across Rust, Python, C, C++, Assembly (x86_64/ARM64), Go, Kotlin, Java, C#, TypeScript/JavaScript, CSS, Ruby, PHP, SQL, and Shell, including Fill-in-the-Middle (FIM) code synthesis.
- Global Multilingual Fluency: High-precision comprehension and synthesis across English, Portuguese, Spanish, French, German, Italian, Russian, Japanese, Korean, Arabic, Hindi, and Simplified Chinese (explicitly excluding Traditional Chinese).
- Multimodal High-Resolution Vision: Native AnyRes vision encoder with 2x2 PixelShuffle spatial compression, achieving 4x lower visual KV cache footprint than standard VLMs while preserving fine-grained OCR, schematics, and UI inspection.
- Real-Time Voice Prosody (ElevenLabs & Fish Speech Ready): Dedicated prosody head and acoustic stream formatter that strips internal reasoning tags and normalizes technical symbols for zero-latency voice synthesis.
- Deterministic 8GB Local Inference: Runs with full context expansion (8k to 32k tokens) strictly under a 7.5 GB VRAM ceiling.

---

## Architectural Innovations

```
[ Text & Polyglot Code ]       [ High-Res Images / Camera ]
           |                                |
           v                                v
   [ BPE Embeddings ]          [ AnyRes Vision + 2x2 PixelShuffle ]
           \                                /
            +--------------+---------------+
                           |
                           v
 +------------------------------------------------------------+
 | Transformer Layer (x28)                                    |
 |                                                            |
 |   Input ---> [ RMSNorm (FP32) ]                            |
 |                   |                                        |
 |                   v                                        |
 |         [ DeepSeek Multi-Head Latent Attention (MLA) ]     |
 |         (Low-Rank KV Compression c_KV=512 + Decoupled RoPE)|
 |                   |                                        |
 |                   v                                        |
 |         ( + Residual Connection )                          |
 |                   |                                        |
 |                   v                                        |
 |              [ RMSNorm (FP32) ]                            |
 |                   |                                        |
 |                   v                                        |
 |         [ Hybrid SwiGLU / DeepSeekMoE FFN ]                |
 |         (1 Shared Expert + 16 Fine-Grained Routed Experts) |
 |         (Aux-Loss-Free Dynamic Bias Load Balancing)        |
 |                   |                                        |
 |                   v                                        |
 |         ( + Residual Connection )                          |
 +------------------------------------------------------------+
           |                                |
           v                                v
 [ Main LM Head (t+1) ]          [ Speculative MTP Head (t+2) ]
           |
           v
 [ Prosody & Voice Formatter (ElevenLabs / Fish Speech) ]
```

### Key Subsystems
1. Multi-Head Latent Attention (MLA): Compresses Key and Value states into a joint low-rank latent vector (`kv_lora_rank = 512`) combined with decoupled Rotary Position Embeddings (`qk_rope_head_dim = 32`). This cuts runtime KV Cache memory by 64.4% compared to Grouped-Query Attention (GQA).
2. Hybrid DeepSeekMoE with Aux-Loss-Free Balancing: Combines dense SwiGLU layers with fine-grained Mixture-of-Experts layers (1 Shared Expert + Top-2 of 16 Routed Experts) regulated by dynamic bias updates without auxiliary loss interference.
3. Multi-Token Prediction (MTP): Simultaneously predicts tokens at $t+1$ and $t+2$, strengthening multi-step algorithmic planning and enabling native speculative decoding at inference time.
4. AnyRes Vision Encoder with 2x2 PixelShuffle: Merges $2 \times 2$ visual patch neighborhoods into dense visual embeddings prior to cross-modal attention.
5. Acoustic Prosody Streamer: Automatically separates `<think>` traces and code blocks from spoken responses, generating clean phonetic streams tailored for ElevenLabs Turbo/Flash and Fish Speech 1.5.

---

## Strict 8GB VRAM Budget (Local Production)

- 4-Bit NF4 Weights (Double Quantization): ~1.62 GB
- DeepSeek MLA Latent KV Cache @ 8,192 tokens (FP16): ~0.23 GB (238 MB)
- DeepSeek MLA Latent KV Cache @ 32,768 tokens (FP16): ~0.93 GB (952 MB)
- FlashAttention-2 Buffers & CUDA Context: ~0.60 GB
- Total Allocated VRAM: ~2.45 GB (at 8k context) to ~3.15 GB (at 32k context) <= 7.50 GB ceiling.

---

## Cognitive Mechanism: Latent Thinking and Native Self-RAG

Mark 0.1 formulates explicit architectural analysis inside `<think>...</think>` tags and autonomously queries the MARK-PUBLIC knowledge index before emitting production code:

1. Introspection: `<think> Inspecting QUIC relay framing and lock boundaries... </think>`
2. Autonomous Retrieval: `[RETRIEVE] src-rust/mark_core/src/mcp/protocol.rs [/RETRIEVE]`
3. External Context Injection: `<|rag_context|> ... <|/rag_context|> [EVALUATE]`
4. Self-Judgment: Emits `[ACCEPT]` or `[REJECT]` prior to synthesizing the final solution.

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

## Local Inference & Voice/Vision Usage (8GB GPU)

```python
from tokenizers import Tokenizer
from inference_8gb import load_mark_for_8gb_vram, MarkInferenceEngine

tokenizer = Tokenizer.from_file("data/tokenizer/mark_bpe_tokenizer.json")
model, kv_cache, stats = load_mark_for_8gb_vram(
    checkpoint_path="checkpoints/mark_0_1_step_001000.pt",
    max_context_tokens=8192
)

engine = MarkInferenceEngine(model, tokenizer, kv_cache)
result = engine.generate_with_voice_payload(
    prompt="Analyze the lock contention in the Rust QUIC relay and provide a lock-free C++/Rust bridge.",
    voice_engine="elevenlabs",
    language="en",
    max_new_tokens=512
)

print(result["raw_response"])
print(result["voice_payload"]["spoken_text"])
```

---

## Governance

This repository contains foundational neural architectures for the MARK-PUBLIC ecosystem.

> Notice on Pull Requests:
> External Pull Requests are disabled and will not be accepted. For bug reports or architectural feedback, please open an Issue.

---

## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0) -- identical to the core MARK-PUBLIC ecosystem. See the [LICENSE](LICENSE) file for complete terms.
