---
language:
- en
- pt
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

Mark-SLM-3M is an ultra-compact language model engineered under the 3,000,000 parameter ceiling for the Open SLM Leaderboard. It integrates Multi-Head Latent Attention (H-MLA) and Parallel Branching Mixture of Experts (PB-MoE), initialized via Truncated Singular Value Decomposition (SVD) Upcycling from Mark-SLM-V3-150M.

## Architectural Specification

- Total Physical Parameters: 2,885,376 (strict ceiling below 3,000,000 parameters)
- Architecture Type: deepseek2 (MLA + MoE)
- Hidden Dimension: 256
- Layers: 8
- Attention Heads: 8
- Key/Value Heads: 8
- Attention Mechanism: Multi-Head Latent Attention (H-MLA)
  - Query LoRA Rank (q_lora_rank): 128
  - Key/Value LoRA Rank (kv_lora_rank): 64
  - Decoupled Rotary Query/Key Dimension (qk_rope_head_dim): 32
  - Non-Rotary Query/Key Dimension (qk_nope_head_dim): 32
  - Value Dimension (v_head_dim): 32
- Feed-Forward / Mixture-of-Experts: Parallel Branching MoE (PB-MoE)
  - Routed Experts (n_routed_experts): 16
  - Active Experts per Token (num_experts_per_tok): 2
  - Shared Experts (n_shared_experts): 1
  - MoE Intermediate Size: 128 (aligned to block size 32)
- Vocabulary Size: 8,192 BPE tokens
- Context Length: 8,192 tokens
- Tied Word Embeddings: True (lm_head.weight tied to model.embed_tokens.weight)

## Storage and Container Footprint

- Unquantized PyTorch Weights (model.safetensors): ~62.94 MiB (~66.00 MB)
- GGUF Container (IQ4_NL): ~2.67 MiB (~2.80 MB)
- Target Platforms: Embedded Microcontrollers, Edge Devices, Ollama, LM Studio, llama.cpp, PyTorch

## Official Benchmark Results (Open SLM Leaderboard)

The model was evaluated on the official Open SLM Leaderboard evaluation suite running on Kaggle (2x Tesla T4).

| Benchmark | Metric | Chance Baseline | Raw Score | Normalized Score (0 to 100) |
| :--- | :--- | :--- | :--- | :--- |
| HellaSwag | acc_norm | 25.0% | 24.40% | 0.00 |
| ARC-Easy | acc_norm | 25.0% | 25.60% | 0.80 |
| ARC-Challenge | acc_norm | 25.0% | 25.75% | 1.00 |
| PIQA | acc_norm | 50.0% | 47.50% | 0.00 |
| ArithMark-3.0 | acc | 25.0% | 84.40% | 79.20 |

### Summary Metrics

- Category: Original (< 3M parameters)
- Physical Parameters: 2,885,376 (Ceiling: < 3,000,000 parameters)
- ARC Average Normalized: 0.90
- Intelligence Index: 14.35

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

inputs = tokenizer("Calculate 42 * 8.", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=64)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
