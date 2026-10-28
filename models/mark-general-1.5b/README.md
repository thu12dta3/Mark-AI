# Mark General 1.5B

Mark General 1.5B is a foundational generalist checkpoint trained on academic STEM, mathematical logic, and scientific curricula. It incorporates Multi-Head Latent Attention (MLA), a 16+1 Mixture-of-Experts (MoE) architecture, and native epistemic Self-RAG routing tokens (`[UNKNOWN]`, `[WEB_SEARCH]`).

- **License:** GNU Affero General Public License v3.0 (AGPL-3.0)
- **Parameters:** 1,501,680,656 physical | ~315,000,000 active FLOPs per token
- **Architecture:** 14 layers, 16 query heads, MLA ($c_{KV}=320, q_{LoRA}=640$), 16 routed + 1 shared expert (top-2 active), MTP $k=2$
- **Target Footprint:** ~1.25 GB VRAM in 4-bit (IQ4_NL), ~128 tokens/second on consumer hardware

---

## Technical Overview & Focus Areas

Mark General 1.5B was trained across a structured academic curriculum covering ten core STEM and technical reasoning fields:
1. **Quantum Mechanics & Condensed Matter Physics:** Hamiltonian formulations, boundary-value problems, bandgap calculations.
2. **Fluid Dynamics & Aerodynamics:** Compressible/incompressible Navier-Stokes approximations, drag coefficient analysis.
3. **Physical & Organic Chemistry:** Transition-state kinetics, reaction mechanisms, molecular orbital analysis.
4. **Clinical Pharmacology & Pharmacokinetics:** Two-compartment clearance, ADME kinetics, receptor-ligand dynamics.
5. **Pure Mathematics & Formal Logic:** Abstract algebra, manifold geometry, predicate calculus proofs.
6. **Distributed Systems & Consensus:** Formal invariants for Paxos, Raft, and Byzantine Fault Tolerance.
7. **Signal Processing & RF Engineering:** Continuous and discrete Fourier transforms, filter topologies, impedance matching.
8. **Quantitative Modeling & Economics:** Dynamic stochastic modeling, option pricing formulations.
9. **Legal & Licensing Engineering:** Intellectual property licensing semantics (AGPL, GPL, Apache) and contract invariants.
10. **Epistemic Calibration:** Recognition of unanswerable or contradictory premises.

### Epistemic Self-RAG Tokens
- `[UNKNOWN]` (Token ID 14): Triggered when prompt queries are unanswerable, refer to non-existent entities, or contain internal factual contradictions.
- `[WEB_SEARCH]` (Token ID 15): Triggered when queries require external temporal information or post-training cut-off data.

---

## Benchmark Results & Empirical Evaluation

Evaluated against competitive models in the 1B to 2B parameter regime across standard reasoning and academic benchmarks.

### 1. Standard Reasoning & Knowledge Benchmarks

| Benchmark (Metric) | Mark General 1.5B | Qwen 2.5 1.5B | Llama 3.2 1B | SmolLM2 1.7B |
| :--- | :---: | :---: | :---: | :---: |
| **MMLU (5-shot)** | **51.8%** | 56.2% | 44.6% | 42.5% |
| **GSM8K (8-shot)** | **56.2%** | 59.8% | 44.4% | 46.1% |
| **ARC-Challenge (25-shot)** | **52.4%** | 54.1% | 46.2% | 44.8% |
| **SciQ (Direct)** | **88.6%** | 89.2% | 82.5% | 83.1% |
| **STEM Subset (Physics/Math/Chem)** | **58.4%** | 53.1% | 41.0% | 43.8% |

*Analysis:* Qwen 2.5 1.5B maintains a higher overall score on general knowledge (MMLU) and multi-step arithmetic (GSM8K). However, on specific undergraduate STEM subsets, Mark General 1.5B demonstrates solid domain competence, exceeding generalist baselines in formal sciences while remaining within a compact 1.5B footprint.

### 2. Epistemic Calibration & Hallucination Metrics

Evaluated on an unanswerable questions test set (queries with false premises, non-existent references, or conflicting data):

| Metric | Mark General 1.5B | Standard 1.5B Baseline (No Epistemic Tokens) |
| :--- | :---: | :---: |
| **Unanswerable Detection (`[UNKNOWN]`) Precision** | **84.6%** | N/A |
| **Unanswerable Detection (`[UNKNOWN]`) Recall** | **79.2%** | N/A |
| **False-Premise Hallucination Rate** | **9.4%** | 26.8% |
| **Brier Calibration Score (Lower is better)** | **0.114** | 0.285 |

*Analysis:* While the model cannot eliminate hallucinations entirely (9.4% error rate remains), the native `[UNKNOWN]` token provides a measurable reduction in fabricated citations and false confident completions compared to standard baseline decoding.

---

## Hardware Efficiency

| Metric | Measurement (NVIDIA RTX 2060 Super 8GB) |
| :--- | :--- |
| **Weights Memory (IQ4_NL)** | 1.25 GB |
| **KV Cache @ 8,192 Context** | 236 MB (MLA compression) |
| **Total Memory @ 8k Context** | 1.49 GB |
| **Throughput** | 128.5 tokens/second |

---

## Quickstart

### Ollama
```bash
ollama run mark-general-1.5b:latest "Explain the Navier-Stokes continuity equation in cylindrical coordinates."
```

### Hugging Face Hub (GGUF)
```bash
huggingface-cli download miceworks/Mark-General-1.5B mark_general_1_5b_iq4_nl.gguf --local-dir ./weights
```
