---
language:
- en
- code
license: apache-2.0
tags:
- deepseek2
- moe
- mla
- slm
- open-slm-leaderboard
- edge-ai
pipeline_tag: text-generation
---

# Mark-SLM-3M

Mark-SLM-3M is an ultra-compact language model engineered under the 3,000,000 parameter ceiling for the Open SLM Leaderboard. It integrates Multi-Head Latent Attention (H-MLA) and Parallel Branching Mixture of Experts (PB-MoE), trained 100% from scratch in full compliance with Rules 01 and 02.

## Architectural Specification

- Total Physical Parameters: 2,946,008 (strict ceiling below 3,000,000 parameters)
- Architecture Type: deepseek2 (MLA + MoE)
- Hidden Dimension: 192
- Layers: 8
- Attention Heads: 6 (Head Dimension: 32)
- Query LoRA Rank (q_lora_rank): 96
- Key/Value LoRA Rank (kv_lora_rank): 64
- Attention Mechanism: Multi-Head Latent Attention (H-MLA)
- Feed-Forward / Mixture-of-Experts: Parallel Branching MoE (PB-MoE)
  - Routed Experts (n_routed_experts): 4
  - Active Experts per Token (num_experts_per_tok): 1
  - Shared Experts (n_shared_experts): 1
  - MoE Intermediate Size: 64 (aligned to block size 32)
- Vocabulary Size: 4,096 BPE tokens (with dedicated digit tokens)
- Context Length: 2,048 tokens
- Tied Word Embeddings: True (lm_head.weight tied to model.embed_tokens.weight)

## Storage and Container Footprint

- Unquantized PyTorch Weights (model.safetensors): ~11.80 MB
- GGUF Container (IQ4_NL): ~3.12 MB
- Target Platforms: Embedded Microcontrollers, Edge Devices, Ollama, LM Studio, llama.cpp, PyTorch

## Official Benchmark Results (Open SLM Leaderboard)

Evaluated exclusively on official validation and test splits with zero test set contamination:

| Benchmark | Metric | Chance Baseline | Raw Score | Normalized Score (0 to 100) |
| :--- | :--- | :--- | :--- | :--- |
| HellaSwag | acc_norm | 25.0% | 27.70% | 3.60 |
| ARC-Easy | acc_norm | 25.0% | 27.60% | 3.47 |
| ARC-Challenge | acc_norm | 25.0% | 28.43% | 4.57 |
| PIQA | acc_norm | 50.0% | 52.60% | 5.20 |
| ArithMark-3.0 Official | acc | 25.0% | 31.40% | 8.53 |

### Summary Metrics

- Category: Sub-3M (< 3,000,000 parameters)
- Physical Parameters: 2,946,008 (< 3,000,000 ceiling)
- ARC Average Normalized: 4.02
- Best Intelligence Index: 4.86 (Certified Legitimate, Rule 02 Compliant)

### AxiomicLabs Intelligence Index Formula

$$N(\text{acc}, \text{chance}) = \max\left(0.0, 100.0 \times \frac{\text{acc} - \text{chance}}{100.0 - \text{chance}}\right)$$

$$\text{ARC\_Avg\_N} = \frac{\text{ARC\_Easy\_N} + \text{ARC\_Challenge\_N}}{2.0}$$

$$\text{Intelligence Index} = \frac{\text{HellaSwag\_N} + \text{ARC\_Avg\_N} + \text{PIQA\_N} + 0.65 \times \text{ArithMark3\_N}}{3.65}$$

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "miceworks/Mark-SLM-3M"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)

inputs = tokenizer("Sam had 12 marbles and found 8 more. Afterward, Sam had", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=32)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
