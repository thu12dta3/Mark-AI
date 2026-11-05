---
language:
- en
- pt
license: apache-2.0
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
size_categories:
- 1K<n<10K
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.parquet
---

# Mark Ultra-Curriculum: Systems, Concurrency, and STEM Dataset

The **Mark Ultra-Curriculum** is an open-source dataset containing **1,131** deduplicated instruction-completion pairs focused on systems programming, Rust/C++ concurrency, Linux eBPF, robotics kinematics, and STEM reasoning.

- **License:** Apache License 2.0 (`apache-2.0`)
- **Primary Formats:** Apache Parquet (`data/train.parquet`) and JSON Lines (`data/train.jsonl`)
- **Total Unique Samples:** 1,131 (sanitized and deduplicated from 4,821 raw records)

## Dataset Taxonomy and Domain Breakdown

| Domain Identifier | Sample Count |
| :--- | :---: |
| `general_systems_engineering` | 586 |
| `systems_rust_concurrency` | 508 |
| `systems_cpp20_atomics` | 13 |
| `systems_linux_ebpf` | 9 |
| `academic_stem_reasoning` | 4 |
| `cybersecurity_and_reversing` | 4 |
| `epistemic_self_rag` | 3 |
| `robotics_and_mechatronics` | 2 |
| `senior_tech_lead_dialogue` | 1 |
| `systems_windows_kmdf` | 1 |
| **Total Unique Samples** | **1,131** |

## Data Structure

Each record contains the following fields:

```json
{
  "id": "mark_ultra_00000",
  "domain": "systems_linux_ebpf",
  "source_curriculum": "v2_8b_extended_master",
  "prompt": "User query...",
  "completion": "<think>\nTechnical step breakdown...\n</think>\nImplementation..."
}
```

## How to Load in Python

```python
from datasets import load_dataset

# Load directly from Hugging Face Hub
dataset = load_dataset("miceworks/Mark-Ultra-Curriculum", split="train")
print(dataset[0])
```

## License and Attribution

Released under the **Apache License 2.0** (`apache-2.0`). Open-source architectural discussions referencing Redis, Nginx, Linux kernel, and Tokio components retain their respective upstream attributions.
