# MarkVL-2.2B - Dense Visual Perception & Description Specialist

## Model Overview
**MarkVL-2.2B** is a specialized Vision-Language Model engineered exclusively for dense, ultra-fine-grained visual description, high-resolution micro-OCR, FACS facial micro-expression decoding, and random real-world object recognition.

The model eliminates generic conversational filler and chit-chat in favor of surgical perceptual precision across technical schematics, everyday items, complex instruments, and physical human interactions.

---

## Technical Specifications
- **Total Physical Parameters:** 2,222,269,696 (2.224B)
- **Active Parameters per Token:** ~313.5M FLOPs (Top-2 routed + 1 shared expert)
- **Vision Subsystem:** AnyRes 2048x2048 (Dynamic Patching, zero aspect-ratio distortion)
- **Spatial Embeddings:** 2D-RoPE (X, Y spatial coordinate grid)
- **Token Compression:** PixelShuffle 2x2 C-Abstractor (75% visual token reduction)
- **Attention:** Multi-Head Latent Attention (MLA, c_KV = 320)
- **Memory Footprint (IQ4_NL):** ~1.85 GB VRAM on NVIDIA GeForce RTX 2060 Super (8GB GDDR6)
- **Free VRAM on 8GB GPU:** > 6.1 GB VRAM available for long multi-image sequences
- **Generation Speed:** 95 to 125 tokens/s

---

## Plug-and-Play in Ollama

```bash
cd models/mark-vl-2.2b
ollama create mark-vl-2.2b -f Modelfile
ollama run mark-vl-2.2b "Descreva com detalhamento maximo os componentes visiveis nesta imagem."
```
