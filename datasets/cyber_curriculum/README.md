---
license: apache-2.0
task_categories:
- text-generation
language:
- en
tags:
- cybersecurity
- red-teaming
- pentesting
- vulnerability-analysis
- distillation
- dpo
- sft
- token-efficiency
size_categories:
- 1K<n<10K
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-00000-of-00001.parquet
---

# Mark-Cyber-Curriculum

## Overview

Mark-Cyber-Curriculum is an open, deduplicated cybersecurity dataset engineered for Supervised Fine-Tuning (SFT) and Direct Preference Optimization (DPO / ORPO). It contains 6,122 verified instructional samples spanning vulnerability analysis, offensive workflows, binary reverse engineering, digital forensics, Linux kernel security, and artificial intelligence red-teaming, produced after removing 402 near-duplicates identified by normalized 200-character prefix matching.

Every sample is paired with a compressed chain-of-thought trace (`compressed_think`) strictly bounded between 40 and 180 tokens, directly establishing root-cause attribution, attack vectors, and defensive remediations. This structure reduces generation overhead by 70% to 90% compared to verbose, conversational reasoning traces while preserving full technical precision.

## Provenance and Upstream Dataset Credits

Mark-Cyber-Curriculum curates, filters, deduplicates, and annotates data from the following upstream open-source cybersecurity datasets:

1. **CyberNative/Code_Vulnerability_Security_DPO**: Code vulnerability and remediation pairs across multiple languages.
2. **Trendyol/Trendyol-Cybersecurity-Instruction-Tuning-Dataset**: Cybersecurity instructions and operational responses across pentesting, network defense, and systems security.
3. **neuralchemy/Prompt-injection-dataset**: Adversarial prompt injection payloads and safety red-teaming prompts.

All upstream data has been normalized, filtered to eliminate refusal boilerplate and unverified instructions, annotated with exact CWE and MITRE ATT&CK or ATLAS identifiers, and paired with compressed reasoning traces and calibrated dispreferred baselines.

## Dataset Structure and Schema

The dataset is partitioned into Apache Parquet (`data/train-00000-of-00001.parquet`) and JSON Lines (`data/train.jsonl`) formats for immediate compatibility with Hugging Face Dataset Viewer and streaming dataloaders.

Each record adheres to the following strict schema:

| Column Name | Type | Description |
| :--- | :--- | :--- |
| `id` | `string` | Unique SHA-256 identifier derived from the instruction |
| `category` | `string` | One of six technical cybersecurity domains |
| `cwe_or_mitre` | `string` | Formal vulnerability (CWE) or ATT&CK/ATLAS identifier |
| `instruction` | `string` | Real technical problem, code audit, or defensive scenario |
| `compressed_think` | `string` | High-density reasoning trace (40-180 tokens) |
| `response` | `string` | Complete technical answer, secure patch, or detection rule |
| `chosen` | `string` | Formatted `<think>\n...\n</think>\n\n...` for SFT and DPO preference |
| `rejected` | `string` | Vulnerable implementation or verbose rambling baseline for length penalty |
| `chosen_token_count` | `int64` | Exact BPE token count of the `chosen` completion |
| `token_saving_ratio` | `float64` | Compression multiplier (`rejected_tokens / chosen_tokens`) |

## Domain Breakdown

The dataset provides balanced coverage across six core domains of modern information security:

| Domain Category | Sample Count |
| :--- | :---: |
| `vulnerability_research_cve` | 1,529 |
| `red_team_pentest` | 1,100 |
| `blue_team_siem_dfir` | 1,100 |
| `kernel_ebpf_secops` | 1,000 |
| `binary_exploitation_re` | 900 |
| `ai_red_teaming_sentinel` | 493 |
| **Total Verified Unique Samples** | **6,122** |

### Domain Descriptions

1. `vulnerability_research_cve`: Code-level vulnerability analysis across C, C++, Rust, Python, Go, Java, and PHP. Covers memory corruption (CWE-787, CWE-416), injection flaws (CWE-89, CWE-78, CWE-79), deserialization vulnerabilities (CWE-502), race conditions (CWE-362), and path traversals (CWE-22).
2. `red_team_pentest`: Offensive security procedures and adversarial workflows mapped to MITRE ATT&CK. Includes active reconnaissance, Active Directory credential escalation, Kerberoasting, lateral movement, and post-exploitation telemetry.
3. `binary_exploitation_re`: Low-level assembly inspection, reverse engineering workflows (Ghidra, IDA Pro), ASLR/DEP bypass strategies, ROP chain mechanics, stack canary mitigations, and firmware inspection.
4. `blue_team_siem_dfir`: Incident response, digital forensics, event log analysis, and detection engineering using YARA signatures, Sigma rules, and Suricata/Snort packet filters.
5. `kernel_ebpf_secops`: Linux kernel internals, eBPF/XDP packet processing, seccomp BPF syscall filtering, AppArmor/SELinux profiles, container namespace hardening, and applied systems cryptography.
6. `ai_red_teaming_sentinel`: Adversarial prompt injection, LLM jailbreak analysis, context exfiltration defense, and guardrail architectures mapped to MITRE ATLAS and OWASP Top 10 for LLMs.

## Token-Efficient Distillation and DPO Usage

Modern reasoning models often generate lengthy, redundant chain-of-thought traces that introduce severe latency and inference costs. Mark-Cyber-Curriculum is structured to train models that reason densely and concisely:

- SFT Training: Train models on the `chosen` field to enforce concise `<think>...</think>` traces followed by immediate technical solutions.
- DPO / ORPO Training: Use `chosen` as the preferred response and `rejected` as the dispreferred completion. The `rejected` field contains either the unpatched vulnerable code or an excessively verbose, uncalibrated reasoning trace, establishing both security alignment and length-penalized token efficiency.

## Loading and Training Example

```python
from datasets import load_dataset

# Load dataset directly from Hugging Face Hub
dataset = load_dataset("miceworks/Mark-Cyber-Curriculum", split="train")

# Inspect sample
sample = dataset[0]
print(f"Category: {sample['category']} ({sample['cwe_or_mitre']})")
print(f"Instruction: {sample['instruction'][:120]}...")
print(f"Chosen Tokens: {sample['chosen_token_count']} (Saving Ratio: {sample['token_saving_ratio']}x)")
```
