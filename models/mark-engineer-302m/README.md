# Mark Engineer 300M (302M)

Mark Engineer 300M is an experimental micro-scale model (302M physical parameters) designed as a testbed for running Multi-Head Latent Attention (MLA) and DeepSeekMoE routing on severely constrained edge hardware and micro-nodes.

- **License:** GNU Affero General Public License v3.0 (AGPL-3.0)
- **Parameters:** 302,019,848 physical | ~110,000,000 active FLOPs per token
- **Architecture:** 12 layers, hidden dimension 1,024, MLA ($c_{KV}=256$), 16 routed + 1 shared expert (top-2 active)
- **Target Footprint:** ~350 MB VRAM, 280+ tokens/second on consumer hardware

---

## Technical Scope & Important Limitations

### 1. Non-Conversational Model
Mark Engineer 300M does **not** have the capacity for natural, multi-turn conversational dialogue or explanatory prose. Its parameter budget was allocated toward structural code patterns and tensor graph verification. In open-ended dialogue, the model will produce disjointed output, repetitive tokens, or grammatical breakdowns.

### 2. High Hallucination Rate in Open-Ended Generation
Due to the strict 302M parameter limit, complex architectural design, multi-file reasoning, and general programming tasks exceed the model's expressive capacity. On standard HumanEval (Python), the model scores under 15%.

### 3. Intended Use Cases
- **Edge Micro-Nodes:** Experimental deployment on devices with under 500 MB of total RAM/VRAM.
- **Deterministic Micro-Kernels:** Verifying narrow C macros, packed struct alignments, and fixed bitwise manipulation templates.
- **Architectural Profiling:** Serving as a low-cost testbed for benchmarking MLA and MoE runtime kernels on CPU and low-power hardware.

If your use case requires full systems engineering dialogue, low-level kernel synthesis, or general coding assistance, use **Mark Engineer 1.5B**.

---

## Evaluation Results on Narrow Synthetic Suites

| Evaluation Suite | Result / Pass Rate | Notes |
| :--- | :---: | :--- |
| **C Packed Struct Alignment (4-byte/8-byte)** | **82.0%** (41/50) | Follows rigid alignment directives without padding errors. |
| **Bitwise Endianness Swap Templates (`bswap32`)** | **86.0%** (43/50) | Correctly outputs bit-shift masks for endian swaps. |
| **Fixed Ringbuffer Index Modulo Arithmetic** | **78.0%** (39/50) | Correctly handles index wrap-around logic. |
| **Standard HumanEval (Python Pass@1)** | **12.4%** | Fails complex multi-step reasoning and algorithmic tasks. |
| **Open Dialogue / Conversational Coherence** | **Unsatisfactory** | Fails natural multi-turn conversational evaluation. |

---

## Hardware Profile

| Metric | Measurement (NVIDIA RTX 2060 Super 8GB) |
| :--- | :--- |
| **Weights Memory (IQ4_NL)** | 225 MB |
| **Runtime VRAM Footprint** | ~350 MB |
| **Throughput** | 284 tokens/second |

---

## Quickstart

### Ollama
```bash
ollama run mark-engineer-302m:latest "/* C Packed Struct */ struct __attribute__((packed)) eth_hdr {"
```

### Hugging Face Hub (GGUF)
```bash
huggingface-cli download miceworks/Mark-Engineer-302M mark_engineer_302m_iq4_nl.gguf --local-dir ./weights
```
