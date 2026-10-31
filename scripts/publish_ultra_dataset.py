#!/usr/bin/env python3
"""
Consolidates and publishes the Mark AI Ultra-Curriculum Dataset to Hugging Face Hub.
Ingests all specialized curricula (systems, kernel, robotics, STEM, vision, epistemic self-RAG),
deduplicates, categorizes into domains, and exports both Parquet and JSONL with an AGPL-3.0 Dataset Card.
"""

import os
import sys
import json
import hashlib
from pathlib import Path
import pandas as pd
import huggingface_hub
from huggingface_hub import HfApi

SOURCES = [
    ("/home/thu12/hd_1tb/mark_training_hub/04_master_curriculum/mark_engineer_v2_8b_extended_master_train.jsonl", "v2_8b_extended_master"),
    ("/home/thu12/hd_1tb/mark_training_hub/04_master_curriculum/mark_engineer_v2_8b_master_train.jsonl", "v2_8b_master"),
    ("/home/thu12/hd_1tb/mark_training_hub/04_master_curriculum/mark_engineer_train.jsonl", "v1_engineer_train"),
    ("/home/thu12/hd_1tb/mark_training_hub/01_code_systems_corpus/systems_real_code_curriculum.jsonl", "real_code_corpus"),
    ("/home/thu12/hd_1tb/mark_training_hub/01_code_systems_corpus/robotics_systems_curriculum.jsonl", "robotics_mechatronics"),
    ("/home/thu12/hd_1tb/mark_training_hub/04_master_curriculum/mark_general_train.jsonl", "general_stem_corpus"),
    ("/home/thu12/hd_1tb/mark_training_hub/04_master_curriculum/mark_vl_dense_captioning_train.jsonl", "vl_dense_captioning"),
    ("/home/thu12/hd_1tb/mark_training_hub/01_code_systems_corpus/systems_code_curriculum.jsonl", "systems_code_curriculum"),
    ("/home/thu12/hd_1tb/mark_training_hub/02_technical_literature/cs_literature_curriculum.jsonl", "cs_literature"),
    ("/home/thu12/hd_1tb/mark_training_hub/03_engineering_interaction/senior_engineering_interaction.jsonl", "senior_interaction"),
    ("/mnt/ssd_1/mark ai/sensitive_archive/data/synthetic_dpo/mark_general_train.jsonl", "general_stem_ssd")
]

def classify_domain(prompt: str, completion: str) -> str:
    text = (prompt + " " + completion).lower()
    if any(k in text for k in ["ebpf", "xdp", "kprobe", "tracepoint", "sk_buff", "bpf_"]):
        return "systems_linux_ebpf"
    elif any(k in text for k in ["kmdf", "irql", "wdm", "bugcheck", "driverentry", "nonpagedpool"]):
        return "systems_windows_kmdf"
    elif any(k in text for k in ["tokio", "async", "pin<", "deadlock", "unsafe", "send + sync"]):
        return "systems_rust_concurrency"
    elif any(k in text for k in ["std::atomic", "memory_order", "co_await", "coroutine", "c++20", "c++23"]):
        return "systems_cpp20_atomics"
    elif any(k in text for k in ["robotics", "kinematics", "mechatronics", "pid controller", "can bus", "kicad", "pcb", "servo"]):
        return "robotics_and_mechatronics"
    elif any(k in text for k in ["navier-stokes", "hamiltonian", "schrödinger", "pharmacokinetics", "adme", "sglt2", "quantum"]):
        return "academic_stem_reasoning"
    elif any(k in text for k in ["[unknown]", "[web_search]", "[fact_check]", "epistemic"]):
        return "epistemic_self_rag"
    elif any(k in text for k in ["rop chain", "aslr", "use-after-free", "shellcode", "heap spray", "disassembly", "pwn"]):
        return "cybersecurity_and_reversing"
    elif any(k in text for k in ["ocr", "bounding box", "pixel-shuffle", "facs", "visual tokens"]):
        return "multimodal_dense_vision"
    elif any(k in text for k in ["gerente", "legado", "refatoração", "tech lead", "liderança", "second-system"]):
        return "senior_tech_lead_dialogue"
    else:
        return "general_systems_engineering"

def build_dataset():
    records = []
    seen = set()
    
    print("Ingesting and deduplicating curriculum sources...")
    for path_str, source_tag in SOURCES:
        p = Path(path_str)
        if not p.exists():
            continue
        count_src = 0
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except Exception:
                    continue
                prompt = data.get("prompt") or data.get("input") or data.get("instruction") or ""
                completion = data.get("completion") or data.get("output") or data.get("response") or ""
                if not prompt.strip() or not completion.strip():
                    continue
                
                h = hashlib.sha256((prompt.strip() + "\n---\n" + completion.strip()).encode("utf-8")).hexdigest()
                if h in seen:
                    continue
                seen.add(h)
                
                domain = classify_domain(prompt, completion)
                records.append({
                    "id": f"mark_sample_{len(records):06d}",
                    "domain": domain,
                    "source_curriculum": source_tag,
                    "prompt": prompt.strip(),
                    "completion": completion.strip()
                })
                count_src += 1
        print(f"  [{source_tag}] Loaded {count_src} unique samples from {p.name}")

    print(f"\nTotal consolidated samples: {len(records)}")
    df = pd.DataFrame(records)
    print("\nDomain distribution:")
    print(df["domain"].value_counts().to_string())
    return df

def generate_dataset_card(df: pd.DataFrame) -> str:
    domain_counts = df["domain"].value_counts().to_dict()
    table_rows = "\n".join(f"| `{domain}` | {count} |" for domain, count in domain_counts.items())

    return f"""---
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

The **Mark Ultra-Curriculum** is an open-source training dataset containing **{len(df):,}** high-density instruction-completion pairs designed for systems programming, low-level concurrency, robotics, STEM reasoning, and epistemic calibration.

- **License:** GNU Affero General Public License v3.0 (AGPL-3.0)
- **Primary Formats:** Apache Parquet (`data/train.parquet`) and JSON Lines (`data/train.jsonl`)
- **Origin:** The foundational training corpus of the Mark AI model ecosystem (Mark Engineer 1.5B/8B, Mark General 1.5B, MarkVL 2.2B).

---

## Dataset Taxonomy & Domain Breakdown

| Domain Identifier | Sample Count |
| :--- | :---: |
{table_rows}
| **Total Unique Samples** | **{len(df):,}** |

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
{{
  "id": "mark_sample_000123",
  "domain": "systems_linux_ebpf",
  "source_curriculum": "v2_8b_extended_master",
  "prompt": "<think>\\n...\\n</think>\\nUser query...",
  "completion": "<think>\\n1. Technical step breakdown...\\n</think>\\nImplementation..."
}}
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
"""

def main():
    repo_id = "miceworks/Mark-Ultra-Curriculum"
    output_dir = Path(__file__).resolve().parent.parent / "datasets" / "ultra_curriculum"
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = output_dir / "data"
    data_dir.mkdir(exist_ok=True)

    df = build_dataset()
    
    parquet_path = data_dir / "train.parquet"
    jsonl_path = data_dir / "train.jsonl"
    readme_path = output_dir / "README.md"

    print(f"\nExporting dataset to Parquet ({parquet_path})...")
    df.to_parquet(parquet_path, index=False, compression="snappy")
    print(f"Parquet file size: {parquet_path.stat().st_size / (1024*1024):.2f} MB")

    print(f"Exporting dataset to JSONL ({jsonl_path})...")
    df.to_json(jsonl_path, orient="records", lines=True, force_ascii=False)
    print(f"JSONL file size: {jsonl_path.stat().st_size / (1024*1024):.2f} MB")

    print(f"Generating Dataset Card ({readme_path})...")
    readme_content = generate_dataset_card(df)
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)

    # Hugging Face upload
    print(f"\nPublishing to Hugging Face Hub: {repo_id}...")
    api = HfApi()
    api.create_repo(repo_id=repo_id, repo_type="dataset", exist_ok=True, private=False)

    print("Uploading README.md (Dataset Card)...")
    api.upload_file(
        path_or_fileobj=readme_path,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="docs: publish Mark Ultra-Curriculum dataset card"
    )

    print("Uploading train.parquet...")
    api.upload_file(
        path_or_fileobj=parquet_path,
        path_in_repo="data/train.parquet",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="feat: add consolidated train.parquet"
    )

    print("Uploading train.jsonl...")
    api.upload_file(
        path_or_fileobj=jsonl_path,
        path_in_repo="data/train.jsonl",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="feat: add raw train.jsonl"
    )

    print(f"\nSUCCESS: Dataset live at https://huggingface.co/datasets/{repo_id}")

if __name__ == "__main__":
    main()
