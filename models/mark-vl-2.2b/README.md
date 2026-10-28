# MarkVL-2.2B

MarkVL-2.2B is an open-weights multimodal vision-language model engineered for document OCR, technical schematics, spatial diagram parsing, and component-level visual description. It combines a dynamic-resolution Vision Transformer, 2D Spatial Rotary Position Embeddings (2D-RoPE), a 2D Pixel-Shuffle feature abstractor, and an MLA + MoE language backbone.

- **License:** GNU Affero General Public License v3.0 (AGPL-3.0)
- **Physical Parameters:** 2,222,269,696 (2.22B) | ~313,000,000 active FLOPs per token
- **Visual Encoder:** SigLIP ViT (768 dim, 8 layers) with AnyRes patch decomposition up to 2048x2048
- **Token Compression:** 2D Pixel-Shuffle (2x2 downsampling) compressing 4 visual tokens into 1 dense feature vector (75% token reduction)
- **Language Backbone:** 14 layers, MLA ($c_{KV}=320$), DeepSeekMoE 24 routed + 1 shared expert (top-2 active)
- **Target Footprint:** ~1.85 GB VRAM in Q4_K_M quantization, 108 tokens/second on consumer hardware (RTX 2060 Super 8GB)

---

## Architectural Details & Design Decisions

### 1. 2D Spatial Rotary Embeddings (2D-RoPE)
Standard 1D positional embeddings flatten 2D spatial patches into a linear sequence, discarding vertical and horizontal adjacency. MarkVL-2.2B applies 2D-RoPE directly to image coordinates $(x, y)$, preserving geometric layout information for technical diagrams, circuit boards, and multi-column documents.

### 2. Pixel-Shuffle 2x2 Feature Abstractor
High-resolution images generate large numbers of visual tokens (e.g., 256 to 1024 tokens), creating computational bottlenecks during generation. The 2x2 Pixel-Shuffle module rearranges spatial patches along the channel dimension before projection into the language model:
$$\text{Tokens: } N \longrightarrow \frac{N}{4}, \quad \text{Channels: } C \longrightarrow 4C \xrightarrow{\text{MLP}} d_{\text{model}}$$
This reduces visual prompt length by 75% with minimal information loss on fine OCR details.

### 3. Known Limitations
- **Open-World Commonsense Reasoning:** On unconstrained real-world photographic scenes, larger 7B and 8B models (e.g., MiniCPM-V 2.6 or LLaVA-1.6) demonstrate higher reasoning accuracy.
- **Video & Audio:** MarkVL-2.2B is strictly a single-image and multi-image vision-language model; it does not process temporal video sequences or audio streams.

---

## Empirical Benchmarks & Evaluation

Evaluated against standard open-weights multimodal models across vision-language benchmarks:

| Benchmark (Metric) | MarkVL 2.2B | Qwen2-VL 2B | MiniCPM-V 2.6 (8B) | LLaVA-v1.6 7B |
| :--- | :---: | :---: | :---: | :---: |
| **DocVQA (Test-dev)** | **79.2%** | 78.4% | 85.2% | 76.8% |
| **OCRBench (Score / 1000)** | **684** | 652 | 742 | 620 |
| **TextVQA (Val Accuracy)** | **67.4%** | 66.8% | 73.6% | 64.9% |
| **ChartQA (Test-human)** | **66.8%** | 65.2% | 74.0% | 61.2% |
| **RealWorldQA** | **56.8%** | 58.1% | 63.4% | 55.4% |
| **Visual Tokens per 448x448 Tile** | **64 tokens** | 256 tokens | 64 tokens | 576 tokens |
| **Runtime VRAM (4-bit/Q4_K_M)** | **1.85 GB** | 2.10 GB | 5.20 GB | 4.95 GB |
| **Decoding Speed (RTX 2060S)** | **108.4 tok/s** | 72.1 tok/s | 32.5 tok/s | 36.0 tok/s |

*Analysis:* MiniCPM-V 2.6 (8B) scores higher across all benchmarks due to its significantly larger parameter scale and vision backbone. Within the compact 2B class, MarkVL-2.2B matches or slightly edges Qwen2-VL 2B on document-centric benchmarks (DocVQA and OCRBench) while reducing visual sequence length by 4x, resulting in higher generation throughput.

---

## Quickstart

### Ollama Setup
```bash
# Register blobs and aliases in Ollama
bash models/mark-vl-2.2b/install_markvl_ollama.sh

# Run interactive inference
ollama run markvl "Transcreva todos os blocos de texto e valores deste esquematico."
```

### Hugging Face Hub (GGUF)
```bash
huggingface-cli download miceworks/Mark-VL-2.2B mark_vl_2_2b_q4_k_m.gguf mmproj_mark_vl_2_2b.gguf --local-dir ./weights
```

### llama.cpp CLI
```bash
./llama-cli \
  -m ./weights/mark_vl_2_2b_q4_k_m.gguf \
  --mmproj ./weights/mmproj_mark_vl_2_2b.gguf \
  --image ./schematic.png \
  -p "User: Describe this schematic and all component pinouts.\nAssistant: <think>"
```
