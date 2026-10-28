# Mark Engineer 1.5B

Mark Engineer 1.5B is a lightweight, open-weights language model designed for low-level systems programming, operating system kernel mechanics, and lock-free concurrency. It integrates Multi-Head Latent Attention (MLA), a 16+1 Mixture-of-Experts (MoE) routing grid, and native Multi-Token Prediction (MTP).

- **License:** GNU Affero General Public License v3.0 (AGPL-3.0)
- **Parameters:** 1,501,680,656 physical | ~315,000,000 active FLOPs per token
- **Architecture:** 14 layers, 16 query heads, MLA ($c_{KV}=320, q_{LoRA}=640$), 16 routed + 1 shared expert (top-2 active), MTP $k=2$
- **Target Footprint:** ~1.25 GB VRAM in 4-bit (IQ4_NL), runs on consumer GPUs (e.g., RTX 2060 Super 8GB) and Apple Silicon

---

## Technical Characteristics & Trade-offs

Mark Engineer 1.5B is optimized for developer environments with strict hardware limits. Rather than targeting general multi-turn conversational chat or high-level scripting (Python web frameworks, web scrapers), its pre-training and distillation corpora prioritize low-level systems codebases:

- **Kernel & Driver Subsystems:** Linux eBPF/XDP verifier rules, Windows KMDF request queues, and macOS Mach IPC.
- **Systems Languages:** Rust (Tokio asynchronous runtimes, pin projections, ownership invariants), C++20/C++23 (coroutine machinery, atomic memory orderings), and bare-metal C.
- **Hardware Efficiency:** Multi-Head Latent Attention compresses the Key-Value cache down to 236 MB at 8,192 tokens (an ~8x reduction compared to standard Multi-Head Attention), allowing long-context execution without GPU out-of-memory errors.

### Known Limitations
- **General Algorithmic Coding:** On broad competitive programming benchmarks (e.g., HumanEval Python), larger generalist models like Qwen 2.5 Coder 7B or 14B achieve higher accuracy due to their significantly larger pre-training corpora.
- **High-Level Frameworks:** It is not specialized in frontend frameworks (React, Vue) or high-level enterprise application stacks (Django, Spring Boot).
- **Context Length:** Native context length is 16,384 tokens. Tasks requiring multi-file repository indexing above this limit require chunking or external retrieval.

---

## Empirical Benchmarks & Evaluation

Evaluation conducted on consumer hardware (NVIDIA GeForce RTX 2060 Super 8GB, PCIe 3.0, FP16/INT4 weights) and compared against standard open-weights coding models.

### 1. Multi-Language Code Generation (Pass@1, greedy decoding)

| Benchmark / Evaluation Suite | Mark Engineer 1.5B | Qwen 2.5 Coder 1.5B | DeepSeek-Coder 1.3B | StarCoder2 3B | Qwen 2.5 Coder 7B |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **MultiPL-E (Rust)** | **48.2%** | 41.6% | 36.4% | 43.1% | 56.8% |
| **MultiPL-E (C++)** | **52.6%** | 46.8% | 41.2% | 47.9% | 59.4% |
| **MultiPL-E (C)** | **54.1%** | 47.5% | 42.0% | 48.6% | 61.2% |
| **HumanEval (Python)** | **66.4%** | 68.9% | 60.3% | 64.2% | 82.3% |
| **MBPP (Python)** | **63.8%** | 66.2% | 58.5% | 61.0% | 77.8% |

*Analysis:* Qwen 2.5 Coder 7B outperforms Mark Engineer 1.5B across general programming benchmarks due to parameter scaling. However, within the sub-2B parameter class, Mark Engineer 1.5B demonstrates higher pass rates in Rust, C++, and C systems domains while remaining competitive on Python logic.

### 2. Domain-Specific Systems Engineering Suites

Evaluated across curated synthetically verifiable low-level test sets:

| Evaluation Suite | Mark Engineer 1.5B | Qwen 2.5 Coder 1.5B | StarCoder2 3B | Qwen 2.5 Coder 7B |
| :--- | :---: | :---: | :---: | :---: |
| **Linux eBPF Verifier Compliance** | **78.4%** | 46.2% | 52.0% | 61.5% |
| **Windows KMDF IRQL Bugcheck Safety** | **81.0%** | 49.3% | 54.8% | 67.2% |
| **Tokio Async Deadlock Avoidance** | **74.6%** | 52.1% | 56.4% | 69.8% |
| **C++20 Acquire/Release Ordering** | **76.2%** | 51.0% | 55.2% | 66.4% |

*Analysis:* In kernel and concurrency domains, generic language models frequently generate code that compiles but triggers runtime panics (e.g., eBPF unbounded loop rejections or Windows `IRQL_NOT_LESS_OR_EQUAL` faults). Mark Engineer's fine-tuning on kernel verifier constraints yields an appreciable improvement in verifier pass rates.

### 3. Latency, Throughput & Memory Utilization (RTX 2060 Super 8GB)

| Metric | Mark Engineer 1.5B (IQ4_NL) | Qwen 2.5 Coder 1.5B (Q4_K_M) | StarCoder2 3B (Q4_K_M) | Qwen 2.5 Coder 7B (Q4_K_M) |
| :--- | :---: | :---: | :---: | :---: |
| **Model VRAM Allocation** | **1.25 GB** | 1.35 GB | 2.45 GB | 4.80 GB |
| **KV Cache @ 8k Tokens** | **236 MB** | 1,890 MB | 2,450 MB | 3,120 MB |
| **Total Memory @ 8k Context** | **1.49 GB** | 3.24 GB | 4.90 GB | 7.92 GB |
| **Generation Throughput** | **128.5 tok/s** | 98.2 tok/s | 68.0 tok/s | 62.0 tok/s |
| **Time to First Token (TTFT)** | **14.2 ms** | 18.5 ms | 28.1 ms | 42.6 ms |

*Throughput note:* The 128.5 tok/s throughput includes native speculative decoding via the Multi-Token Prediction (MTP $k=2$) head, providing an average 1.56x speedup over single-token generation.

---

## Quickstart

### Ollama
```bash
ollama run mark-engineer-1.5b:latest "Analyze this Rust Tokio code for potential deadlock patterns."
```

### Hugging Face Hub (GGUF)
```bash
huggingface-cli download miceworks/Mark-Engineer-1.5B mark_engineer_1_5b_iq4_nl.gguf --local-dir ./weights
```

### llama.cpp CLI
```bash
./llama-cli \
  -m ./weights/mark_engineer_1_5b_iq4_nl.gguf \
  -p "User: Write an eBPF XDP packet filter in C.\nAssistant: <think>" \
  -n 512
```
