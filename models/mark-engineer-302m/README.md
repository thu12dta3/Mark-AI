# Mark Engineer 300M (302M) - Experimental Edge & Synthetic Testbed
==============================================================================
Model Name: Mark Engineer 300M (302M Parameters)
Architecture: Multi-Head Latent Attention (MLA) + DeepSeekMoE (16 Routed + 1 Shared Expert)
Parameter Count: 302,019,848 Physical | ~110M Active per Token
Status: Experimental Edge Checkpoint / Synthetic Micro-Testbed
Target Hardware: Edge Devices, Embedded Linux, Low-Power Micro-Nodes (< 500 MB VRAM)

Strict Constraint: Zero emojis across all documentation and logs.

---

## Technical Disclaimer & Operational Limitations

### Non-Conversational Architecture
Mark Engineer 300M does NOT possess conversational fluency or natural dialogue capabilities. It was not trained as an assistant and cannot maintain coherent multi-turn chit-chat or explain concepts in prose.

### High Hallucination Probability in Free-Form Generation
Due to its ultralight parameter budget (302M physical parameters), open-ended generation, reasoning prompts, or conversational queries will produce disjointed output, repetitive loops, or severe factual hallucinations.

### Purpose and Valid Use Cases
This checkpoint serves exclusively as:
1. An experimental micro-testbed to validate the forward tensor graph of Multi-Head Latent Attention (MLA) and DeepSeekMoE routing on severely constrained hardware.
2. A synthetic syntax validator for micro-kernel snippets and rigid template completion.
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
| **Context Length** | 4,096 Tokens |

---

## 2. Benchmark Results on Synthetic Test Suites

Mark Engineer 300M passes narrow, deterministic synthetic unit tests while failing free-form conversational evaluations:

- **Rigid C Macro & Ringbuffer Syntax Check:** PASS (10/10 template matches)
- **Assembly Register Allocation Synthetic Scan:** PASS (8/8 deterministic register constraints)
- **Natural Language Dialogue & Explanations:** FAIL (Severe lack of syntactic cohesion)
- **Multi-File Architectural Reasoning:** FAIL (High hallucination rate)

---

## 3. Plug-and-Play Quickstart (For Edge & Synthetic Testing)

### 3.1 Ollama
```bash
cd models/mark-engineer-302m
ollama create mark-engineer-302m -f Modelfile
ollama run mark-engineer-302m "/* C Ringbuffer */"
```

### 3.2 LM Studio
1. Download `mark_engineer_302m_iq4_nl.gguf` from the official release:
   ```bash
   curl -L -O https://github.com/thu12dta3/Mark-AI/releases/download/v1.0.0-302m/mark_engineer_302m_iq4_nl.gguf
   ```
2. Move to `~/.lmstudio/models/Mark-AI/mark-engineer-302m/mark_engineer_302m.gguf`.
3. Load the model. Allocate ~350 MB VRAM.

### 3.3 Python CLI Downloader
```bash
python scripts/download_model.py --model 302m
```
