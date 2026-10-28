---
language:
- en
- pt
license: agpl-3.0
task_categories:
- text-generation
- question-answering
tags:
- mark-ai
- code
- systems-engineering
- kernel
- rust
- cpp
- ebpf
- robotics
- mechatronics
- synthetic-dpo
- reasoning
size_categories:
- 1K<n<10K
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.parquet
---

# Mark Ultra-Curriculum: Systems, Kernel, Robotics & Epistemic Reasoning Dataset

The **Mark Ultra-Curriculum** is an open-source training dataset containing **4,821** high-density instruction-completion pairs designed for systems programming, low-level concurrency, robotics, STEM reasoning, and epistemic calibration.

- **License:** GNU Affero General Public License v3.0 (AGPL-3.0)
- **Primary Formats:** Apache Parquet (`data/train.parquet`) and JSON Lines (`data/train.jsonl`)
- **Origin:** The foundational training corpus of the Mark AI model ecosystem (Mark Engineer 1.5B/8B, Mark General 1.5B, MarkVL 2.2B).

---

## Dataset Taxonomy & Domain Breakdown

| Domain Identifier | Sample Count |
| :--- | :---: |
| `general_systems_engineering` | 3736 |
| `systems_rust_concurrency` | 722 |
| `academic_stem_reasoning` | 137 |
| `senior_tech_lead_dialogue` | 70 |
| `systems_cpp20_atomics` | 45 |
| `multimodal_dense_vision` | 42 |
| `cybersecurity_and_reversing` | 28 |
| `systems_linux_ebpf` | 21 |
| `robotics_and_mechatronics` | 14 |
| `epistemic_self_rag` | 5 |
| `systems_windows_kmdf` | 1 |
| **Total Unique Samples** | **4,821** |

### Key Domain Descriptions:
1. **`systems_linux_ebpf`:** BPF/XDP packet filters, kprobes, verifier bounded-loop invariants, zero-copy socket handling.
2. **`systems_windows_kmdf`:** Kernel-Mode Driver Framework, IRQL separation (`PASSIVE` vs `DISPATCH`), IRP queuing, pool allocation safety.
3. **`systems_rust_concurrency`:** Asynchronous Tokio runtimes, pin projection, cross-await deadlock avoidance, `Send`/`Sync` formal properties.
4. **`systems_cpp20_atomics`:** C++20/C++23 coroutines, atomic memory orderings (`acquire`/`release`/`seq_cst`), lock-free ringbuffers.
5. **`robotics_and_mechatronics`:** PID motor control, forward/inverse kinematics, CAN bus communication, PCB differential routing in KiCad.
6. **`academic_stem_reasoning`:** Quantum mechanics (Hamiltonian systems), fluid dynamics (Navier-Stokes), chemical kinetics, clinical pharmacokinetics (ADME).
7. **`epistemic_self_rag`:** Built-in `[UNKNOWN]` and `[WEB_SEARCH]` calibration pairs for identifying false premises and unanswerable queries.
8. **`cybersecurity_and_reversing`:** Memory corruption mitigation (ASLR, DEP, CFI), ROP chain mechanics, ELF/PE parsing, disassembly analysis.
9. **`senior_tech_lead_dialogue`:** Pragmatic engineering trade-offs, legacy system refactoring, empathetic architectural leadership.

---

## Data Structure

Each record contains the following fields:

```json
{
  "id": "mark_sample_000123",
  "domain": "systems_linux_ebpf",
  "source_curriculum": "v2_8b_extended_master",
  "prompt": "<think>\n...\n</think>\nUser query...",
  "completion": "<think>\n1. Technical step breakdown...\n</think>\nImplementation..."
}
```

---

## How to Load in Python

```python
from datasets import load_dataset

# Load directly from Hugging Face Hub
dataset = load_dataset("miceworks/Mark-Ultra-Curriculum", split="train")

print(dataset[0])
```

---

## License & Distribution

Released under the **GNU Affero General Public License v3.0 (AGPL-3.0)** to guarantee open accessibility, reproducibility, and copyleft sharing across the open-source machine learning research community.
