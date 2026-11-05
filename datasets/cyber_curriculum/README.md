---
license: apache-2.0
task_categories:
- text-generation
- question-answering
language:
- en
tags:
- cybersecurity
- secops
- vulnerability-research
- dpo
- sft
size_categories:
- 1K<n<10K
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-00000-of-00001.parquet
---

# Mark-Cyber-Curriculum: Defensive and Offensive Cybersecurity Dataset

`Mark-Cyber-Curriculum` is a curated, strictly deduplicated cybersecurity training dataset containing **4,613 unique samples** across vulnerability research (CWE/CVE), Red Team penetration testing, Blue Team SIEM/DFIR, binary exploitation, Linux kernel eBPF SecOps, and AI prompt-injection sentinel defense.

## Dataset Summary and Provenance

- **Total Unique Samples:** 4,613 (strictly deduplicated across normalized prompt, first-150-character response prefix, and `compressed_think` from 6,524 raw records)
- **License:** `apache-2.0`
- **Upstream Open-Source Attributions:**
  * [`CyberNative/Code_Vulnerability_Security_DPO`](https://huggingface.co/datasets/CyberNative/Code_Vulnerability_Security_DPO) (Apache-2.0) — paired secure (`chosen`) and vulnerable (`rejected`) code implementations across CWE categories.
  * [`Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset`](https://huggingface.co/datasets/Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset) (Apache-2.0) — defensive engineering, SOC operations, and incident response instructions.
  * [`neuralchemy/Cybersecurity-Dataset`](https://huggingface.co/datasets/neuralchemy/Cybersecurity-Dataset) — offensive/defensive MITRE ATT&CK analysis and red/blue team workflows.

## Domain Breakdown

| Category | Unique Samples |
| :--- | :---: |
| `vulnerability_research_cve` | 1,465 |
| `red_team_pentest` | 1,077 |
| `blue_team_siem_dfir` | 1,051 |
| `binary_exploitation_re` | 896 |
| `kernel_ebpf_secops` | 116 |
| `ai_red_teaming_sentinel` | 8 |
| **Total** | **4,613** |

## Schema

Each record contains:
- `id`: Unique sample identifier.
- `category`: Domain category (`vulnerability_research_cve`, `red_team_pentest`, `blue_team_siem_dfir`, `binary_exploitation_re`, `kernel_ebpf_secops`, `ai_red_teaming_sentinel`).
- `cwe_or_mitre`: Associated CWE or MITRE ATT&CK technique identifier.
- `instruction`: Technical security prompt or code task.
- `compressed_think`: Structured threat vector and remediation reasoning summary.
- `response`: Verified defensive/remediation implementation or analysis.
- `chosen`: Formatted reasoning + response target.
- `rejected`: Upstream vulnerable code baseline (populated for `CyberNative/Code_Vulnerability_Security_DPO` pairs; empty string `""` for SFT-only records).

## Quick Start

```python
from datasets import load_dataset

dataset = load_dataset("miceworks/Mark-Cyber-Curriculum", split="train")
print(f"Loaded {len(dataset)} unique samples")
print(dataset[0])
```
