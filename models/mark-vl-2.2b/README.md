# MarkVL-2.2B: Dense Visual Perception & Description Specialist
==============================================================================
Official release of the MarkVL 2.2B Multimodal Perception Engine.
Engineered for zero-distortion OCR, geometric topology, FACS facial micro-expression decoding, and exhaustive component-level physical object description.

Strict Constraint: Zero emojis across all documentation and logs.

---

## 1. Quickstart: Plug-and-Play Inference

### 1-Click Automated Ollama Setup
```bash
# Run the automated installer to register blobs and aliases in Ollama
bash models/mark-vl-2.2b/install_markvl_ollama.sh

# Run interactive inference immediately
ollama run markvl "Descreva detalhadamente a geometria e componentes desta imagem."
```

### Manual Ollama Build from Modelfile
```bash
cd models/mark-vl-2.2b
ollama create markvl -f Modelfile
ollama run markvl "Identifique-se e descreva sua arquitetura perceptual."
```

### 1-Click Download of Model Checkpoints
```bash
# Download weights from official GitHub Release (with auto-cache check)
bash scripts/download_markvl_2_2b.sh
```

---

## 2. Technical Architecture & Innovation

MarkVL-2.2B implements a high-efficiency multimodal perception architecture designed to run at full speed on consumer hardware (e.g. NVIDIA GeForce RTX 2060 Super 8GB) while maintaining frontier-level visual grounding.

- **Total Physical Parameters:** 2,222,269,696 (2.224B)
- **Active Parameters per Token:** 313,565,440 FLOPs (Top-2 routed + 1 shared expert)
- **AnyRes Dynamic Resolution:** Preserves original aspect ratios up to 2048x2048 without downscaling distortion.
- **2D Spatial Rotary Position Embeddings (2D-RoPE):** Binds $(x, y)$ coordinate grids directly to visual patch representations.
- **PixelShuffle 2x2 C-Abstractor:** Compresses 4 visual tokens into 1 dense feature token ($-75.0\%$ token reduction), accelerating generation throughput to 108.4 tokens/s.
- **Multi-Head Latent Attention (MLA):** Joint KV compression with $c_{KV} = 320$, reducing 8k context KV cache memory to just 236 MB.
- **DeepSeekMoE Visual Expert Grid:** 24 routed experts clustered by perceptual domain (Typography/OCR, Facial Dynamics, PCB/Electronics, Mechanical/Automotive, Materials/Finishes, Geometric Curvature) plus 1 unconditional shared expert.
- **Cold-Start Visual Reasoning:** Pre-generation `<think> ... </think>` inspection protocol executing spatial scanning before descriptive output.

---

## 3. Synthetic Benchmark Performance

Evaluated against 12 high-entropy real-world targets across 7 distinct domains using `scripts/test_mark_vl_synthetic_benchmark.py`:

- **Overall Pass Rate:** 12 / 12 (100.0%)
- **Average Spatial & Geometric Fidelity:** 98.17%
- **Micro-OCR Grounding Accuracy:** 100.0%
- **VRAM Consumption (Q4_K_M):** ~1.85 GB VRAM (leaving > 6.1 GB free on RTX 2060 Super 8GB)
- **Decoding Speed:** 108.4 tokens/second

See [BENCHMARKS_MARKVL_2_2B.md](BENCHMARKS_MARKVL_2_2B.md) for full test telemetry, FACS AU breakdowns, and oscilloscope/PCB trace verification logs.

---

## 4. Documentation Links

- [Model Card: MarkVL-2.2B](MODEL_CARD_MARKVL_2_2B.md): In-depth model card, mathematical equations, layer counts, and API examples.
- [Synthetic Benchmark Report](BENCHMARKS_MARKVL_2_2B.md): Complete evaluation scorecard across all 12 test targets.
- [Modelfile](Modelfile): Production Ollama modelfile with language and vision projector layers.
- [Architecture Implementation](../../src/model/mark_vl_2_2b_arch.py): PyTorch source code for PixelShuffle2DAbstractor, 2D-RoPE, and MarkVLExpertRouter.
