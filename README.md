# Mark 0.1 — Core Neural Architecture & Training Pipeline

<div align="center">

![License: AGPLv3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C.svg)
![Hardware](https://img.shields.io/badge/Training-NVIDIA_A100_(Colab)-76B900.svg)
![Target](https://img.shields.io/badge/Inference-Local_GPU_8GB_VRAM-0052CC.svg)
![Params](https://img.shields.io/badge/Parameters-~3.0B_Dense-green.svg)

**The central cognitive engine of the MARK-PUBLIC distributed ecosystem.**  
*A foundational system assistant engineered from scratch for systems architecture, systems programming, and autonomous reasoning.*

</div>

---

## 📌 Executive Overview

**Mark 0.1** is not a conversational chatbot. It is a foundational causal language model designed specifically to act as the primary intelligence layer within the **MARK-PUBLIC** distributed infrastructure:
- **Python Cognitive Core** (`src/mark/`)
- **High-Performance Rust Transport & Security Layer** (`src-rust/mark_core/`)
- **Android / WearOS Clients**

### Dual-Constraint Hardware Design
1. **Cloud Supercomputing Training:** Trained on Google Colab using high-end **NVIDIA A100 (80GB/40GB)** clusters leveraging `bfloat16` mixed precision, FlashAttention-2, fused CUDA operators, and gradient checkpointing.
2. **Deterministic 8GB Local Inference:** Engineered to run seamlessly on commodity consumer hardware with **strict $\le 7.5\text{ GB}$ VRAM ceilings** using 4-bit NormalFloat (`nf4`) double quantization and grouped-query KV caching.

---

## 🏛️ Mathematical Foundations & Architecture

```
[ Input Tokens ]
       │
       ▼
[ Token Embeddings (V=49,152, D=2560) ] ── (Tied Weights with LM Head)
       │
       ▼
 ┌────────────────────────────────────────────────────────┐
 │ Transformer Layer (x32)                                │
 │                                                        │
 │   Input  ──► [ RMSNorm ]                               │
 │                │                                       │
 │                ▼                                       │
 │              [ Grouped-Query Attention (GQA 4:1) ]     │
 │              (32 Q Heads, 8 KV Heads, FlashAttn-2)     │
 │              (NTK-Aware Dynamic RoPE: 8k ──► 32k)      │
 │                │                                       │
 │                ▼                                       │
 │              ( + Residual Connection )                 │
 │                │                                       │
 │                ▼                                       │
 │              [ RMSNorm ]                               │
 │                │                                       │
 │                ▼                                       │
 │              [ SwiGLU FFN ] (Intermediate = 6912)      │
 │                │                                       │
 │                ▼                                       │
 │              ( + Residual Connection )                 │
 └────────────────────────────────────────────────────────┘
       │
       ▼
 [ Final RMSNorm ]
       │
       ▼
 [ Linear LM Head (V=49,152) ] ──► Next-Token Probabilities
```

### Parameter Count Breakdown ($\sim 2.95\text{B}$)
* **Hidden Size ($d_{\text{model}}$):** `2560`
* **Layers:** `32`
* **Query Attention Heads ($H_q$):** `32` ($d_{\text{head}} = 80$)
* **KV Heads ($H_{kv}$):** `8` (Grouped-Query Attention ratio $4:1$, saving 75% KV Cache memory)
* **SwiGLU Dimension:** `6912` ($\approx \frac{8}{3} d_{\text{model}}$)
* **Rotary Embeddings (RoPE):** High base frequency ($\theta = 500\,000$) with dynamic NTK extrapolation for context extension up to 32,768 tokens.
* **Normalization:** Root Mean Square Layer Normalization (`RMSNorm`, $\epsilon = 10^{-5}$) computed in `float32`.

### Strict 8GB VRAM Budget (Local Production)
$$\text{VRAM}_{\text{total}} = \underbrace{1.68\text{ GB}}_{\text{4-bit NF4 Weights}} + \underbrace{0.34\text{ GB}}_{\text{KV Cache @ 8k (FP16)}} + \underbrace{0.65\text{ GB}}_{\text{FlashAttn-2 Buffers}} = \mathbf{2.67\text{ GB}} \ll 7.5\text{ GB}$$
Even scaling to **32,768 tokens**, the KV Cache requires only $\approx 1.34\text{ GB}$, capping dynamic VRAM below $3.7\text{ GB}$ and leaving over $4\text{ GB}$ of free memory for the OS and local runtime buffers.

---

## 🧠 Cognitive Mechanism: Latent Thinking & Native Self-RAG

Unlike generic models that output unstructured answers, **Mark 0.1** is conditioned during pre-training and alignment to formulate explicit logical reasoning inside `<think>...</think>` tags and actively retrieve missing context.

### The Self-RAG State Machine
1. **Introspection:** The model reflects:
   ```
   <think>
   WearOS link bridge is dropping packets. Inspecting stream lifecycle in relay.rs before modifying the lock.
   </think>
   ```
2. **Autonomous Retrieval:** The model halts autoregression and emits:
   ```
   [RETRIEVE] src-rust/mark_link/src/relay.rs [/RETRIEVE]
   ```
3. **External Injection:** The inference interceptor queries the project index (`src/mark/cognition/rag.py`) and returns context between special boundary tokens:
   ```
   <|rag_context|>
   pub async fn handle_wear_stream(...) { ... }
   <|/rag_context|>
   [EVALUATE]
   ```
4. **Self-Judgment:** The model validates relevance by emitting `[ACCEPT]` or `[REJECT]` before writing production code.

### Composite Loss Function
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{weighted-CE}} + \lambda_{\text{struct}} \cdot \max\left(0, \tau_{\text{think}} - P_\theta(\texttt{<think>} \mid x)\right) + \lambda_{\text{orpo}} \cdot \mathcal{L}_{\text{ORPO}}$$
- $\alpha_{\text{think}} = 1.35$ (boosts logical steps)
- $\alpha_{\text{rag}} = 1.80$ (heavily rewards correct emission of `[RETRIEVE]` and evaluation tags)
- RAG raw context tokens are masked ($w_t = 0$, `label = -100`) so the model learns to *evaluate* context instead of memorizing it.
- **ORPO** (*Odds-Ratio Preference Optimization*) contrasts structured thought pairs against shallow answers without requiring a secondary reference model in VRAM.

---

## 📂 Repository Topology

```
Mark-AI/
├── colab_notebooks/
│   └── Mark_0_1_A100_Trainer.ipynb     # Interactive Google Colab A100 training notebook
├── data/
│   ├── raw/                            # Source corpus (Rust, Python, Kotlin, MCP specs)
│   └── synthetic_dpo/                  # High-quality distilled reasoning traces
├── export/
│   └── quantize_4bit.py                # BitsAndBytes NF4 4-bit quantization & PagedKVCache
├── src/
│   ├── model/
│   │   ├── mark_arch.py                # Core PyTorch model (GQA, RoPE, SwiGLU, RMSNorm)
│   │   └── self_rag.py                 # Self-RAG control tokens and runtime interceptor
│   ├── tokenizer/
│   │   └── build_bpe.py                # Byte-Level BPE tokenizer & dataset distillation
│   └── training/
│       ├── reasoning_loss.py           # Weighted CE + Structure Penalty + ORPO Loss
│       └── train_cloud.py              # A100 BF16 training loop & Drive auto-checkpointing
├── inference_8gb.py                    # Root entrypoint for local 8GB execution
├── model.py                            # Root architecture entrypoint
├── requirements.txt                    # Production dependencies
├── tokenizer_setup.py                  # Root tokenizer and dataset preparation script
└── train_reasoning.py                  # Root cloud training entrypoint
```

---

## ⚡ Quickstart

### 1. Training on Google Colab (A100 SXM4 80GB)
Open a new notebook with the A100 runtime enabled and run:

```python
# 1. Mount Google Drive for persistent checkpoints
from google.colab import drive
drive.mount('/content/drive')

# 2. Clone Mark-AI
!rm -rf /content/Mark-AI
!git clone https://github.com/thu12dta3/Mark-AI.git /content/Mark-AI
%cd /content/Mark-AI
!pip install -q -r requirements.txt

# 3. Build Tokenizer and Synthetic Reasoning Corpus
from tokenizer_setup import generate_synthetic_reasoning_dataset, build_mark_bpe_tokenizer
generate_synthetic_reasoning_dataset()
build_mark_bpe_tokenizer()

# 4. Launch A100 Training (BF16 + Fused AdamW + FlashAttention-2)
from train_reasoning import train_mark_a100
train_mark_a100(
    checkpoint_dir="/content/drive/MyDrive/Mark-AI/checkpoints",
    max_steps=5000,
    save_every_n_steps=250,
    batch_size=8,
    grad_accum_steps=4,
    use_full_3b_config=True
)
```

### 2. Local Inference on Consumer 8GB GPU
```python
from tokenizers import Tokenizer
from inference_8gb import load_mark_for_8gb_vram, MarkInferenceEngine

# Loads weights into 4-bit NormalFloat (NF4) under strict 7.5 GB VRAM ceiling
tokenizer = Tokenizer.from_file("data/tokenizer/mark_bpe_tokenizer.json")
model, kv_cache, stats = load_mark_for_8gb_vram(
    checkpoint_path="checkpoints/mark_0_1_step_001000.pt",
    max_context_tokens=8192
)

engine = MarkInferenceEngine(model, tokenizer, kv_cache)
response = engine.generate(
    prompt="Refactor the QUIC relay channel in daemon.rs to avoid deadlock on disconnect.",
    max_new_tokens=512,
    temperature=0.1
)
print(response)
```

---

## 🔒 Governance & Contributions

This repository contains foundational neural architectures and model weights for the private ecosystem **MARK-PUBLIC**.

> **Notice on Pull Requests:**
> This repository is maintained as an authoritative upstream research project. **External Pull Requests are disabled and will not be accepted.** For bug reports or architectural feedback, please open an Issue.

---

## 📜 License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)** — identical to the core `MARK-PUBLIC` ecosystem. See the [LICENSE](LICENSE) file for complete terms.
