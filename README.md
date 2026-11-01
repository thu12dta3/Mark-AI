# Mark AI -- Foundational Sparse Neural Architectures & Systems Engine

The central cognitive, polyglot systems engineering, and multimodal engine of the MARK ecosystem. Architected from first principles on native DeepSeek2 sparse topologies, Multi-Head Latent Attention (MLA), and 3D Mixture-of-Experts (MoE) in non-linear 4-bit quantization (IQ4_NL), engineered specifically to achieve industrial-grade inference within an 8 GB consumer GPU memory boundary.

---

## 1. Ecosystem Ground-Truth Matrix (11 Models - GGUF Binary Truth)

The matrix below documents all 11 models across the Mark ecosystem. All metrics are derived from physical binaries via `gguf.GGUFReader`:

| Model Name | Weight State | Binary Size | Total Parameters | Active Parameters | Vocab Size | SHA-256 | Hugging Face Repository |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- | :--- |
| **Mark-Engineer-302M** | Trained Checkpoint (Validated) | 135.12 MiB (141.69 MB / 141,686,432 bytes) | 251,452,416 | 119,331,840 | 4311 | `5e9d7d6feb...` | [miceworks/Mark-Engineer-302M](https://huggingface.co/miceworks/Mark-Engineer-302M) |
| **Mark-SLM-V2-302M** | Trained Checkpoint (Validated) | 177.76 MiB (186.39 MB / 186,394,944 bytes) | 329,132,000 | 155,723,744 | 49152 | `34a3e0e39c...` | [miceworks/Mark-SLM-V2-302M](https://huggingface.co/miceworks/Mark-SLM-V2-302M) |
| **Mark-Engineer-1.5B** | Trained Checkpoint (Validated) | 802.94 MiB (841.94 MB / 841,941,664 bytes) | 1,496,121,824 | 364,151,264 | 4311 | `fa7d3061f3...` | [miceworks/Mark-Engineer-1.5B](https://huggingface.co/miceworks/Mark-Engineer-1.5B) |
| **Mark-General-1.5B** | Trained Checkpoint (Validated) | 802.97 MiB (841.98 MB / 841,977,664 bytes) | 1,496,185,824 | 364,215,264 | 4377 | `4f2be0d7c3...` | [miceworks/Mark-General-1.5B](https://huggingface.co/miceworks/Mark-General-1.5B) |
| **Mark-VL-2.2B** | Trained Checkpoint (Validated) | 1166.72 MiB (1223.40 MB / 1,223,397,152 bytes) | 2,174,310,096 | 395,499,216 | 32000 | `073d3c00ea...` | [miceworks/Mark-VL-2.2B](https://huggingface.co/miceworks/Mark-VL-2.2B) |
| **Mark-Engineer-V2-8B** | Untrained Architectural Base | 4431.26 MiB (4646.52 MB / 4,646,516,576 bytes) | 8,258,809,040 | 1,770,745,040 | 49152 | `28c014b5b9...` | [miceworks/Mark-Engineer-V2-8B](https://huggingface.co/miceworks/Mark-Engineer-V2-8B) |
| **Mark-SLM-V3-150M** | Untrained Architectural Base | 79.97 MiB (83.85 MB / 83,849,984 bytes) | 146,899,248 | 54,001,968 | 32032 | `6308c8c152...` | [miceworks/Mark-SLM-V3-150M](https://huggingface.co/miceworks/Mark-SLM-V3-150M) |
| **Mark-V3-1.1B** | Untrained Architectural Base | 656.13 MiB (688.00 MB / 688,002,048 bytes) | 1,222,464,960 | 330,651,072 | 32032 | `e12c358e81...` | [miceworks/Mark-V3-1.1B](https://huggingface.co/miceworks/Mark-V3-1.1B) |
| **Mark-General-V3-1.5B** | Untrained Architectural Base | 816.03 MiB (855.67 MB / 855,669,248 bytes) | 1,520,496,960 | 405,729,600 | 32032 | `7677ebd444...` | [miceworks/Mark-General-V3-1.5B](https://huggingface.co/miceworks/Mark-General-V3-1.5B) |
| **Mark-Engineer-V3-1.5B** | Untrained Architectural Base | 816.03 MiB (855.67 MB / 855,669,248 bytes) | 1,520,496,960 | 405,729,600 | 32032 | `62b8bd003a...` | [miceworks/Mark-Engineer-V3-1.5B](https://huggingface.co/miceworks/Mark-Engineer-V3-1.5B) |
| **Mark-Experimental-V3-8B** | Untrained Architectural Base | 4382.55 MiB (4595.44 MB / 4,595,438,848 bytes) | 8,168,001,024 | 2,002,374,144 | 32032 | `792e127a71...` | [miceworks/Mark-Experimental-V3-8B](https://huggingface.co/miceworks/Mark-Experimental-V3-8B) |

---

## 2. Mathematical Architecture & Core Innovations

### 2.1 Multi-Head Latent Attention (MLA) & Hadamard-MLA (H-MLA)
Standard Multi-Head Attention (MHA) creates an unsustainable memory bottleneck during long-context autoregressive generation due to linear expansion of stored Key-Value caches. Mark AI implements Multi-Head Latent Attention (MLA), projecting keys and values into a shared low-rank latent representation:

$$\mathbf{c}_t^{KV} = W^{DKV} \mathbf{h}_t \quad (\mathbf{c}_t^{KV} \in \mathbb{R}^{d_{cKV}})$$

$$\mathbf{c}_t^Q = W^{DQ} \mathbf{h}_t \quad (\mathbf{c}_t^Q \in \mathbb{R}^{d_{cQ}})$$

$$\mathbf{q}_t^C = \text{RMSNorm}(W^{UQ} \mathbf{c}_t^Q), \quad \mathbf{k}_t^C = \text{RMSNorm}(W^{UK} \mathbf{c}_t^{KV})$$

To maintain positional awareness without expanding the cached latent vector, decoupled rotary position embeddings are concatenated along a separate dimension:

$$\mathbf{k}_t^R = \text{RoPE}(W^{KR} \mathbf{h}_t), \quad \mathbf{q}_{t, i} = [\mathbf{q}_{t, i}^C; \mathbf{q}_{t, i}^R], \quad \mathbf{k}_{t, i} = [\mathbf{k}_{t, i}^C; \mathbf{k}_t^R]$$

In the V3 series, **Hadamard-MLA (H-MLA)** suppresses latent activation outliers by absorbing the normalized Walsh-Hadamard transformation:

$$\mathbf{H}_{256} = \frac{1}{16} \mathbf{H}_2^{\otimes 8}$$

directly into the offline projection weights ($W_{wkv_a}$ and $W_{wkv_b}$), eliminating activation kurtosis without adding runtime execution overhead.

### 2.2 DeepSeekMoE & Partitioned-Balancing MoE (PB-MoE)
To decouple parameter scale from inference compute costs, Mark models partition feed-forward layers into fine-grained specialists:
- **Shared Experts**: Fixed shared pathways capture cross-domain foundational invariants.
- **Routed Experts**: Activated dynamically via top-$K$ gating with learned Aux-Loss-Free router biases ($s_{t, e} = \text{Softmax}(\mathbf{x}_t W_{\text{router}} + \mathbf{b}_e)$).
- **Domain Clan Partitioning (V3)**: Experts are partitioned into 4 distinct functional clans (Systems/Code, STEM/Logic, Security/Sentinel, Multimodal) with soft-pinning constraints ensuring balanced domain representation per token.

### 2.3 Epistemic Self-RAG & Shannon Entropy Routing
The ecosystem incorporates an adaptive epistemic evaluation framework (`src/model/self_rag.py`). By monitoring token-level prediction entropy:

$$H(X) = -\sum_{i=1}^{V} P(x_i) \log_2 P(x_i)$$

the model autonomously detects uncertainty spikes, triggering structured `<think>` reasoning paths or external knowledge retrieval phases before finalizing output tokens.

---

## 3. Binary Engineering & Non-Linear 4-Bit Quantization (IQ4_NL)

### Native DeepSeek2 GGUF Graph
Mark AI preserves native 3D tensor layouts (`blk.{i}.ffn_gate_exps.weight`, `blk.{i}.ffn_up_exps.weight`, `blk.{i}.ffn_down_exps.weight`) within the `deepseek2` GGUF container, bypassing the memory fragmentation associated with 2D dense flattening.

### Super-Block Structure (GGML_TYPE_IQ4_NL)
- Linear weight matrices are compressed into 18-byte blocks per 32 weights:
  - 2 bytes: Half-precision scale factor $d$.
  - 16 bytes: 32 four-bit nibbles mapped onto the non-linear 16-level quantization grid:
    `[-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]`
- 1D RMSNorm tensors and router bias vectors remain in full precision `FP32`, ensuring numerical stability at layer boundaries.

---

## 4. Repository Structure

```
Mark-AI/
├── artifacts/              # Forensic binary audits and verification reports
│   └── gguf_forensic_audit.json
├── export/                 # DeepSeek2 GGUF export and quantization pipelines
│   └── convert_to_deepseek2_gguf.py
├── models/                 # Model specifications, configs, tokenizers and Modelfiles
│   ├── mark-engineer-1.5b/
│   ├── mark-engineer-302m/
│   ├── mark-engineer-v2-8b/
│   ├── mark-engineer-v3-1.5b/
│   ├── mark-experimental-v3-8b/
│   ├── mark-general-1.5b/
│   ├── mark-general-v3-1.5b/
│   ├── mark-slm-v2-302m/
│   ├── mark-slm-v3-150m/
│   ├── mark-v3-1.1b/
│   └── mark-vl-2.2b/
├── scripts/                # Ecosystem synchronization and verification utilities
│   ├── generate_verification_manifests.py
│   ├── gguf_forensic_xray.py
│   ├── sanitize_all_models_metadata.py
│   ├── sync_to_huggingface.py
│   └── update_all_readmes.py
├── spaces/                 # Interactive inference spaces and deployment code
│   └── mark-engineer-1.5b/
│       ├── app.py
│       ├── requirements.txt
│       └── README.md
├── src/                    # PyTorch reference neural architectures
│   └── model/
└── tests/                  # Automated metadata and architectural test suite (70 tests)
```

---

## 5. Automated Verification & Test Suite

The entire Mark AI repository is continuously validated against physical GGUF binaries via an auditable test suite:

```bash
# 1. Validate Python syntax across all modules
python3 -m compileall -q src/ export/ scripts/ models/ tests/ spaces/

# 2. Run the 70-test comprehensive validation suite
pytest tests/ -v
```

All 70 tests must pass with 100% success, verifying:
- Cross-consistency between `config.json`, `tokenizer_config.json`, `tokenizer.json`, `Modelfile`, and GGUF headers.
- Integer parity for total and active parameter counts.
- Dual unit size verification (`MiB` and `MB`).
- Strict absence of unregistered special tokens.
