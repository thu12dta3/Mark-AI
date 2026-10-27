# Mark Engineer 300M (302M) - Experimental Edge & Synthetic Testbed
==============================================================================
Model Name: Mark Engineer 300M (302M Parameters)
Architecture: Multi-Head Latent Attention (MLA) + DeepSeekMoE (16 Routed + 1 Shared Expert)
Parameter Count: 302,019,848 Physical | ~110M Active per Token
Status: Experimental Edge Checkpoint / Synthetic Micro-Testbed
Target Hardware: Edge Devices, Embedded Linux, Low-Power Micro-Nodes (< 500 MB VRAM)

---

## Technical Disclaimer & Operational Limitations

### Non-Conversational Architecture
Mark Engineer 300M does NOT possess conversational fluency or natural dialogue capabilities. It was not trained as an assistant and cannot maintain coherent multi-turn chit-chat or explain concepts in prose ("não sabe falar").

### High Hallucination Probability in Free-Form Generation
Due to its ultralight parameter budget (302M physical parameters), open-ended generation, reasoning prompts, or conversational queries will produce disjointed output, repetitive loops, or severe factual hallucinations.

### Purpose and Valid Use Cases
This checkpoint serves exclusively as:
1. An experimental micro-testbed to validate the forward tensor graph of Multi-Head Latent Attention (MLA) and DeepSeekMoE routing on severely constrained hardware.
2. A synthetic syntax validator for micro-kernel snippets, endian transforms, and rigid template completion.
3. Edge deployment benchmarking on low-power devices with under 500 MB RAM/VRAM.

If your task requires conversational tech lead dialogue, system architecture reasoning, multi-file code synthesis, or zero-hallucination analysis, use **Mark Engineer 1.5B** or **MarkVL-2.2B**.

---

## 1. Technical Specifications

| Property | Value |
| :--- | :--- |
| **Physical Parameter Count** | 302,019,848 (302M) |
| **Active FLOPs per Token** | ~110,000,000 (Top-2 Routed + 1 Shared Expert) |
| **Hidden Dimension ($d$)** | 1,024 |
| **Transformer Depth** | 12 Layers |
| **Attention Architecture** | Multi-Head Latent Attention (MLA, $c_{KV} = 256$) |
| **MoE Topology** | 16 Routed Experts + 1 Universal Shared Expert |
| **Quantization Format** | IQ4_NL (4-bit non-linear lookup) / Universal GGUF |
| **On-Disk Checkpoint Size** | 162.99 MB (GGUF: 225.21 MB) |
| **Runtime VRAM Footprint** | ~350 MB on NVIDIA GeForce RTX 2060 Super |
| **Inference Throughput** | 284 tokens/second on RTX 2060S |
| **Context Length** | 4,096 Tokens |

---

## 2. Benchmark Results on Synthetic Test Suites

Evaluated across rigid synthetic kernels vs conversational probes:

| Test Suite / Target Capability | Result | Pass Rate | Evaluation Behavior |
| :--- | :---: | :---: | :--- |
| **Rigid C Packed Structs & Bitmasks** | **PASS** | **96.0%** (48/50) | Accurate deterministic bit alignment. |
| **Endianness Swap (`ntohl`/`bswap32`)** | **PASS** | **98.0%** (49/50) | Exact register mapping and bit shifts. |
| **Fixed-Size Ringbuffer Boundary Logic** | **PASS** | **92.0%** (46/50) | Bounded queue index wrapping correctly. |
| **CRC32 Lookup Table Kernel Synthesis** | **PASS** | **94.0%** (47/50) | Synthesizes unrolled polynomial tables. |
| **Natural Language Conversational Dialogue** | **FAIL** | **12.0%** (6/50) | Disjointed syntax; loops or halts abruptly. |
| **Open-Domain Technical Explanations** | **FAIL** | **16.0%** (8/50) | High hallucination rate on non-kernel concepts. |

---

## 3. Plug-and-Play Quickstart (For Edge & Synthetic Testing)

### 3.1 Ollama
```bash
ollama run mark-engineer-302m:latest "/* C Ringbuffer */"
```

### 3.2 Hugging Face / GGUF
```bash
huggingface-cli download miceworks/Mark-Engineer-302M mark_engineer_302m_iq4_nl.gguf --local-dir ./weights
```

### 3.3 llama.cpp CLI
```bash
./llama-cli \
  -m ./weights/mark_engineer_302m_iq4_nl.gguf \
  -p "/* C Packed Struct */\nstruct __attribute__((packed)) ethernet_hdr {" \
  -n 128
```
