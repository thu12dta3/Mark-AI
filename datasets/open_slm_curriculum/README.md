---
license: agpl-3.0
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
---

# Mark-Open-SLM-Curriculum

Mark-Open-SLM-Curriculum is a public, curated pretraining corpus designed for training Small Language Models (SLMs) from scratch in strict compliance with Rule 02 of the Open SLM Leaderboard.

Every sample in this dataset is sourced exclusively from the official training splits of open, permissive academic benchmarks or procedural generators, completely excluding validation and evaluation splits to eliminate benchmark contamination.

## Dataset Summary

- Total Samples: 12,000
- Total Token Count: 3,467,902 tokens (using the Mark-AI 32,032 BPE tokenizer)
- Average Length: 288 tokens per sample
- Deduplication Rate: 100% unique prompts (SHA-256 verified)
- Primary Model Target: Mark-SLM-V3-150M and downstream Mark-AI architectures

## Domain Composition

| Domain | Source Split | Samples | Percentage | Description |
|---|---|---|---|---|
| physical_commonsense | Rowan/hellaswag (train) & ybisk/piqa (train) | 3,000 | 25.0% | Physical commonsense scenario reasoning and continuation |
| scientific_reasoning | ai2_arc Easy/Challenge (train) & sciq (train) | 3,000 | 25.0% | Elementary and middle school scientific explanations and facts |
| procedural_arithmetic | ArithMark-3 Procedural Generator | 3,500 | 29.2% | Multi-step arithmetic (addition, subtraction, multiplication, rates, percentages) |
| general_stem | HuggingFaceTB/smollm-corpus cosmopedia-v2 (train) | 2,500 | 20.8% | High-density educational STEM textbooks and course modules |

## Rule 02 Compliance Statement

The Open SLM Leaderboard establishes clear standards regarding model pretraining and contamination avoidance:
1. Pure Pretraining from Scratch: Mark-SLM-V3-150M is initialized from random Gaussian tensor weights without importing or adapting proprietary foundation weights.
2. Zero Validation Contamination: Only the training splits of open corpora are utilized. No test or validation sets from HellaSwag, ARC, PIQA, or GSM8K are included.
3. Open Data Transparency: All pretraining data partitions are fully accessible for reproducibility.

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

Carregar o dataset diretamente via Hugging Face Datasets:

```python
from datasets import load_dataset

dataset = load_dataset("miceworks/Mark-Open-SLM-Curriculum", split="train")
print(f"Loaded {len(dataset)} samples")
print(dataset[0])
```
