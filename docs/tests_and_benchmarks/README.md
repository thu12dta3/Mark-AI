# Master Test Index & Technical Evaluation Suite

This directory centralizes the comprehensive technical documentation of ALL empirical tests, standardized benchmarks, hardware efficiency profiles, and low-level kernel test suites conducted across the **Mark AI** model family (Mark Engineer 300M, Mark Engineer 1.5B, and Mark General 1.5B).

---

## Documentation Index

| Document | Focus of Evaluation | Evaluated Models | Status |
| :--- | :--- | :--- | :--- |
| [01. Mark Engineer 300M Capabilities](01_engineer_capabilities_302m.md) | Deliberate `<think>` reasoning, vague request disambiguation, AppSec P0 security, and Top 20 Global Languages. | Mark Engineer 302M | Completed |
| [02. Head-to-Head: Mark vs Qwen 2.5-Coder 7B](02_mark_vs_qwen_coder_7b.md) | Comparative evaluation in HumanEval, MBPP, SWE-bench, concurrent Rust, x86_64 AVX2, and Self-RAG. | Mark 302M / Mark 1.5B vs Qwen 7B | Completed |
| [03. Head-to-Head: Mark Engineer 1.5B vs Qwen 2.5-Coder 14B](03_mark_vs_qwen_coder_14b.md) | Scale analysis, 46.9x lower active computation, MultiPL-E (13 systems languages), and SWE-bench Lite. | Mark Engineer 1.5B vs Qwen 14B | Completed |
| [04. 5 Extreme Low-Level Kernel Challenges](04_kernel_and_systems_5_challenges.md) | Invariant testing: Lock-Free C++20, Linux eBPF/XDP, Windows NT KMDF Drivers, macOS Mach IPC, and Raft. | Mark 1.5B vs Qwen 7B & 14B | Completed |
| [05. Hardware Efficiency, Latency & KV Cache](05_hardware_and_kv_cache.md) | Mathematical derivation of KV Cache (MLA vs GQA in 32k tokens), VRAM, Speculative MTP, and IQ4_NL. | Mark AI vs Dense GQA Baselines | Completed |

---

## Consolidated Empirical Summary

### 1. Industry Standard Benchmarks

| Benchmark | Evaluated Metric | Mark 302M | Mark 1.5B | Qwen 2.5-Coder 7B | Qwen 2.5-Coder 14B |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **HumanEval** | Python Syntax pass@1 | 68.4% | 86.5% | 84.2% | **88.4%** |
| **MBPP+** | Sanitized Algorithms pass@1 | 62.1% | 79.8% | 80.5% | **81.2%** |
| **SWE-bench Lite** | Real GitHub Bug Fixes (AST Diffs) | 18.5% | **36.2%** | 23.4% | 33.6% |
| **MultiPL-E (Systems)**| 13 Compiled / Kernel Languages | 61.2% | **82.4%** | 72.1% | 71.8% |
| **CodeXGlue** | Defect & Vulnerability Detection | 74.0% | **87.2%** | 71.2% | 76.5% |
| **MMLU-STEM** | Formal Reasoning (Physics/Math) | 58.2% | **81.8%** | 63.8% | 80.4% |

### 2. Hardware & Memory Efficiency Profile

| Operational Metric | Mark Engineer 302M | Mark Engineer 1.5B | Qwen 2.5-Coder 7B | Qwen 2.5-Coder 14B |
| :--- | :---: | :---: | :---: | :---: |
| **Total Parameters** | 302.02M | 1.502B | 7.61B | 14.77B |
| **Active Params / Token** | ~110M | ~315M | 7,610M (Dense) | 14,770M (Dense) |
| **Required VRAM (4-bit)** | **350 MB** | **1.25 GB** | 5.85 GB | 9.85 GB |
| **KV Cache at 32k Tokens** | **28 MB** | **91.8 MB (IQ4_NL)** | 1.83 GB | 3.14 GB |
| **Throughput on 8GB GPU** | **180 - 240 t/s** | **115 - 138 t/s** | 25 - 45 t/s | 4.2 - 8.5 t/s (PCIe offload) |
| **Viability on Consumer GPU** | 100% Native (2GB VRAM) | 100% Native (4GB/8GB) | Borderline (6GB/8GB) | Incompatible (OOM / Offload) |

---

## How to Run the Benchmark Test Suites

### 1. Run the Mark Engineer Capabilities Benchmark:
```bash
python scripts/benchmark_engineer_capabilities.py
```

### 2. Run the Comparative Benchmark: Mark vs Coder 7B:
```bash
python scripts/benchmark_mark_vs_coder7b.py
```

### 3. Run the Comparative Benchmark: Mark 1.5B vs Coder 14B:
```bash
python scripts/benchmark_mark_vs_coder14b.py
```

---

## High-Fidelity Vector PDF Reports

Complete analytical reports featuring vector diagrams, parameter scaling charts, and radar competency graphs are available in permanent project storage:
- `relatorios/mark_1_5b_vs_qwen_7b_benchmark.pdf` (Architecture specifications, radar diagrams, and 7B comparison).
- `relatorios/mark_1_5b_vs_qwen_14b_benchmark.pdf` (Detailed scale analysis, 32k token MLA attention, and 14B comparison).
