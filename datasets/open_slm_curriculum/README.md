---
license: apache-2.0
task_categories:
- text-generation
language:
- en
tags:
- open-slm
- slm
- pretraining
- curriculum-learning
- arithmetic
- reasoning
size_categories:
- 10K<n<100K
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train-00000-of-00001.parquet
---

# Mark-Open-SLM-Curriculum

`Mark-Open-SLM-Curriculum` is a public, curated pretraining corpus designed for training Small Language Models (SLMs) from scratch in compliance with Rule 02 of the Open SLM Leaderboard.

Every sample in this dataset is sourced from the official training splits of open academic benchmarks or procedural arithmetic generators, verified against ARC-Easy and ARC-Challenge validation and test splits to eliminate cross-split overlap.

## Dataset Summary

- **Total Samples:** 11,990 (after removing 10 cross-split duplicate questions present in upstream training sets that overlapped with ARC validation/test splits)
- **Total Token Count:** 3,467,662 tokens (using the Mark-AI 32,032 BPE tokenizer)
- **Average Length:** 289 tokens per sample
- **License:** Apache License 2.0 (`apache-2.0`)
- **Primary Model Target:** `Mark-SLM-V3-150M`

## Domain Composition

| Domain | Source Split | Samples | Percentage | Description |
|---|---|---|---|---|
| `physical_commonsense` | `Rowan/hellaswag` (train) & `ybisk/piqa` (train) | 3,000 | 25.0% | Physical commonsense scenario reasoning and continuation |
| `scientific_reasoning` | `allenai/ai2_arc` Easy/Challenge (train) & `allenai/sciq` (train) | 2,990 | 24.9% | Elementary and middle school scientific explanations and facts |
| `procedural_arithmetic` | ArithMark-3 Procedural Generator | 3,500 | 29.2% | Multi-step arithmetic (addition, subtraction, multiplication, rates, percentages) |
| `general_stem` | `HuggingFaceTB/smollm-corpus` `cosmopedia-v2` (train) | 2,500 | 20.9% | Educational STEM textbooks and course modules |

## Schema

Each entry in the Parquet and JSONL files contains:

```json
{
  "id": "slm_e7b29a10f3c8",
  "domain": "scientific_reasoning",
  "source": "arc_easy_train",
  "prompt": "Which factor will most likely cause a person to develop a fever?",
  "completion": "a bacterial population in the bloodstream",
  "text": "Which factor will most likely cause a person to develop a fever? a bacterial population in the bloodstream",
  "token_count": 24
}
```

## Quick Start

Load the dataset directly via Hugging Face Datasets:

```python
from datasets import load_dataset

dataset = load_dataset("miceworks/Mark-Open-SLM-Curriculum", split="train")
print(f"Loaded {len(dataset)} samples")
print(dataset[0])
```
