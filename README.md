# Mark AI -- Foundational Neural Architecture & Systems Engine

<div align="center">

![License: AGPLv3](https://img.shields.io/badge/License-AGPL_v3-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C.svg)
![Hardware](https://img.shields.io/badge/Compute-NVIDIA_A100-76B900.svg)
![Target](https://img.shields.io/badge/Inference-Local_GPU_8GB_VRAM-0052CC.svg)
![Architecture](https://img.shields.io/badge/Architecture-MLA_%2B_MoE_%2B_MTP-green.svg)

**The central cognitive, polyglot systems engineering, and multimodal engine of the MARK ecosystem.**

</div>

---

## Model Hierarchy

| Variant | Total Parameters | Active Params / Token | Target & Purpose | Disk (IQ4_NL) | Local VRAM |
| :--- | :---: | :---: | :--- | :---: | :---: |
| **Mark Engineer 300M (302M)** | **302.02M** | **~110M - 165M** | Preserved ultra-lightweight edge production checkpoint. | **163 MB** | **~350 MB** |
| **Mark Engineer 1.5B** | **1.502B** | **~315M - 440M** | High-capacity systems engineering, AppSec P0, low-level OS & concurrency. | **820 MB** | **~1.25 GB** |
| **Mark General 1.5B** | **1.502B** | **~315M - 440M** | Omniscient reasoning across 10 official cognition domains, STEM & Self-RAG. | **820 MB** | **~1.25 GB** |
| **Mark Primal Base** | **~2.98B** | **~2.35B** | Multimodal foundation base with vision encoder and acoustic streaming. | **1.62 GB** | **~2.45 GB** |

---

## Quickstart: Mark Engineer 300M (302M)

The 302M trained model is ready for 1-click download as a single file (163 MB IQ4_NL), just like downloading from Hugging Face:

### 1. LM Studio & Ollama (Plug-and-Play GGUF)

```bash
# Download direto do arquivo GGUF para o LM Studio (576 MB)
mkdir -p ~/.lmstudio/models/Mark-AI/mark-engineer-302m
curl -L -o ~/.lmstudio/models/Mark-AI/mark-engineer-302m/mark_engineer_302m.gguf \
  https://github.com/thu12dta3/Mark-AI/releases/download/v1.0.0-302m/mark_engineer_302m_iq4_nl.gguf
```

### 2. PyTorch Native Single-File Download (163 MB IQ4_NL)

```bash
# Opcao A: Download direto via curl (1 unico arquivo)
curl -L -O https://github.com/thu12dta3/Mark-AI/releases/download/v1.0.0-302m/mark_engineer_302m_iq4_nl.pt

# Opcao B: Download com verificacao de integridade SHA256 via Python
python scripts/download_302m.py
```

### 3. Run Local Inference

```bash
# Execucao com prompt customizado
python scripts/inference.py --model 302m --prompt "Escreva um lock-free ring buffer concorrente em C++20."
```

### 3. Python API Quickstart

```python
from src.mark import load_model, enforce_vram_ceiling

# Trava de seguranca automatica para manter VRAM < 7.5 GB
enforce_vram_ceiling(max_vram_gb=7.5)

# Carregamento em 1 linha
model, config = load_model("302m")
print(f"Modelo 302M carregado com sucesso: {config.count_parameters():,} parametros.")
```

---

## Architectural Innovations

```
[ Input Tokens: Text / Code / RFC ]
               |
               v
     [ BPE Embeddings (4,377 voc) ]
               |
               v
 +------------------------------------------------------------+
 | Transformer Layer Block (x8 ou x16)                        |
 |                                                            |
 |   Input ---> [ RMSNorm (FP32) ]                            |
 |                   |                                        |
 |                   v                                        |
 |         [ DeepSeek Multi-Head Latent Attention (MLA) ]     |
 |         (Low-Rank KV Compression c_KV + Decoupled RoPE)    |
 |                   |                                        |
 |                   v                                        |
 |         ( + Residual Connection )                          |
 |                   |                                        |
 |                   v                                        |
 |              [ RMSNorm (FP32) ]                            |
 |                   |                                        |
 |                   v                                        |
 |         [ Hybrid SwiGLU / DeepSeekMoE FFN ]                |
 |         (1 Shared Expert + 16 Fine-Grained Routed Experts) |
 |         (Aux-Loss-Free Dynamic Bias Load Balancing)        |
 |                   |                                        |
 |                   v                                        |
 |         [ Recursive Depth LoRA (r=64) Refinement Loop ]    |
 |                   |                                        |
 |                   v                                        |
 |         ( + Residual Connection )                          |
 +------------------------------------------------------------+
               |                                |
               v                                v
     [ Main LM Head (t+1) ]          [ Speculative MTP Head (t+2) ]
```

### Key Subsystems
1. **Multi-Head Latent Attention (MLA):** Compresses Key and Value states into a joint low-rank latent vector (`kv_lora_rank = 256/320`) combined with decoupled Rotary Position Embeddings (`qk_rope_head_dim = 32`). Reduces runtime KV Cache footprint by over 64% compared to standard GQA.
2. **Hybrid DeepSeekMoE with Aux-Loss-Free Balancing:** Combines dense layers with 16 fine-grained routed experts and 1 shared expert, regulated by dynamic router bias updates without loss interference.
3. **Multi-Token Prediction (MTP):** Simultaneously predicts tokens at $t+1$ and $t+2$, accelerating multi-step algorithmic planning and enabling native speculative decoding at inference time.
4. **Native IQ4_NL Non-Linear Quantization:** Uses a non-linear 16-level quantization table tuned to Gaussian weight distributions, enabling 4-bit precision with near-FP16 fidelity on pure PyTorch without external dependencies.

---

## Repository Structure

```
Mark-AI/
├── download_302m.py            # 1-click script para download e preparacao do modelo 302M
├── inference.py                # Motor de inferencia unificado local (< 7.5GB VRAM)
├── pyproject.toml              # Definicao de pacote padrao Python (pip install -e .)
├── requirements.txt            # Dependencias minimas (PyTorch, Tokenizers, Accelerate)
├── export/
│   └── quantize_4bit.py        # Quantizador e carregador nativo IQ4_NL / NF4
├── scripts/
│   ├── download_model.py       # Utilitario de download por variante
│   └── benchmark_*.py          # Protocolos de benchmark de engenharia
├── src/
│   ├── mark/                   # Pacote principal Mark AI (API publica)
│   │   └── __init__.py         # Exportacoes publicas e helper load_model()
│   ├── model/
│   │   ├── mark_arch.py        # Arquitetura neural (MLA, MoE, MTP, IQ4_NL)
│   │   └── self_rag.py         # Controlador de raciocinio Self-RAG e Perplexity
│   └── tokenizer/
│       └── build_bpe.py        # Construtor do vocabulario BPE (4.377 tokens)
├── LICENSE                     # Licenca AGPL-3.0
└── README.md                   # Documentacao oficial e instrucoes
```

---

## Security & Architecture Protection

To safeguard core intellectual property and research integrity:
- Internal training loss formulations, synthetic instruction generation pipelines, and raw corpus preparation recipes are isolated from public repository tracking and maintained in private training storage.
- All production checkpoint weights and training pipelines operate in hardened cloud compute infrastructure (NVIDIA A100-SXM4-80GB), with export artifacts verified via cryptographic SHA256 checksums.

---

## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0). See the [LICENSE](LICENSE) file for complete terms.
