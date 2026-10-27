# Model Card: MarkVL-2.2B (Dense Visual Perception Engine)
==============================================================================
Model Name: MarkVL 2.2B Dense Visual Perception Engine
Architecture: AnyRes ViT + PixelShuffle 2x2 C-Abstractor + DeepSeekMoE (24+1) + Multi-Head Latent Attention (MLA)
Parameters: 2,222,269,696 (2.224B) Physical | 313,565,440 Active per Token
Release Version: 1.0.0
Author: Mark AI (thu12dta3)
Target Hardware: NVIDIA GeForce RTX 2060 Super (8GB GDDR6) or Cloud GPU

---

## 1. Model Summary

MarkVL-2.2B is a production-grade multimodal Vision-Language Model engineered exclusively for dense visual perception, geometric spatial mapping, zero-distortion high-resolution OCR, and micro-expression decoding.

Unlike general-purpose multimodal models that spend tokens on conversational chit-chat, MarkVL-2.2B acts as an analytical perception specialist. It decomposes visual inputs into spatial coordinates, material properties, mechanical states, and transcribed typography with surgical accuracy.

### Primary Capabilities
- **Dense Visual Inspection:** Exhaustive inventory of every visible component, texture, finish, and spatial relationship in a scene.
- **High-Resolution Micro-OCR:** Transcription of technical schematics, PCB silkscreens, watch dial typography, book spines, barcodes, and tilted labels down to 4pt font.
- **Facial Action Coding System (FACS):** Detection and scoring of facial Action Units (AU1 to AU28), subtle affective states, and fine hand gestures.
- **Geometric Topology:** Accurate spatial grounding using relative coordinates and 3D depth perception.
- **Cold-Start Visual Reasoning:** Enforces systematic `<think> ... </think>` decomposition prior to generating final perceptual summaries.

---

## 2. Architectural Specifications

### 2.1 Multimodal Pipeline
```
[High-Resolution Image (up to 2048x2048)]
                    │
                    ▼
   [AnyRes Dynamic Patch Decomposer]
         (Preserves Aspect Ratio)
                    │
                    ▼
     [Vision Transformer Backbone]
        (768 dim, 8 layers, 12 heads)
                    │
                    ▼
    [2D Spatial Rotary Embeddings (2D-RoPE)]
      (Injects Exact Grid Coordinates)
                    │
                    ▼
     [PixelShuffle 2x2 C-Abstractor]
  (4 tokens -> 1 token: 75% Compression)
                    │
                    ▼
 [Visual Token Projection (768*4 -> 1280 dim)]
                    │
                    ▼
    [DeepSeekMoE + MLA Language Backbone]
  (14 layers, 16 heads, 24+1 Expert Grid)
```

### 2.2 Key Architecture Parameters
- **Vocabulary Size:** 32,000 tokens (BPE, byte-level fallback).
- **Hidden Dimension ($d$):** 1,280.
- **Transformer Layers:** 14 backbone layers.
- **Attention Heads:** 16 query heads, 4 key-value heads.
- **Multi-Head Latent Attention (MLA):**
  - Compression dimension $c_{KV} = 320$.
  - Query compression rank $q_{LoRA} = 640$.
  - Decoupled RoPE dimension $d_{RoPE} = 32$.
  - Non-RoPE key dimension $d_{NoPE} = 64$.
  - Value head dimension $d_{V} = 96$.
  - Peak KV Cache footprint at 8k context: 236 MB.
- **DeepSeekMoE Routing:**
  - 24 routed visual experts + 1 shared expert.
  - Intermediate size per expert: 1,504.
  - Active experts per token: Top-2 routed + 1 shared expert.
- **Vision Subsystem:**
  - Patch size: 14x14 pixels.
  - Native resolution: AnyRes up to 2048x2048.
  - PixelShuffle downsampling scale: 2x2 (75% visual token reduction).

---

## 3. Plug-and-Play Quickstart

### 3.1 Direct Ollama Usage

MarkVL-2.2B is packaged with official Ollama manifests and custom Mark blobs.

```bash
# 1-Click Installation and Registration
bash models/mark-vl-2.2b/install_markvl_ollama.sh

# Run interactive inference
ollama run markvl "Descreva os componentes presentes nesta placa de circuito impresso."
```

### 3.2 Python Inference API

```python
import json
import urllib.request

def analyze_image_with_markvl(prompt: str, image_base64: str = None) -> str:
    url = "http://127.0.0.1:11434/api/generate"
    payload = {
        "model": "markvl:2.2b",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.15,
            "top_p": 0.90,
            "repeat_penalty": 1.15
        }
    }
    if image_base64:
        payload["images"] = [image_base64]

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return data.get("response", "")

if __name__ == "__main__":
    response = analyze_image_with_markvl("Identifique os componentes visiveis.")
    print(response)
```

### 3.3 cURL Direct API Call

```bash
curl -X POST http://127.0.0.1:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "markvl:2.2b",
    "prompt": "Descreva com detalhamento cirurgico a topologia espacial da cena.",
    "stream": false
  }'
```

---

## 4. Hardware Calibration & Quantization

| Hardware Setup | Quantization | Model Size | VRAM Allocated | Throughput | Free VRAM |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **NVIDIA RTX 2060 Super (8GB)** | Q4_K_M | 3.18 GB | 1.85 GB | 108.4 tok/s | 6.15 GB |
| **NVIDIA RTX 3060 (12GB)** | Q4_K_M | 3.18 GB | 1.85 GB | 125.0 tok/s | 10.15 GB |
| **NVIDIA A100-SXM4 (80GB)** | FP16 / BF16 | 4.44 GB | 4.90 GB | 215.0 tok/s | 75.10 GB |
| **Apple Silicon (M1/M2/M3 16GB)** | Q4_K_M | 3.18 GB | 2.10 GB | 85.0 tok/s | 13.90 GB |

---

## 5. Artifact Manifest & Checksums

| Asset File | SHA256 Checksum | Size | Format |
| :--- | :--- | :--- | :--- |
| `mark_vl_2_2b_q4_k_m.gguf` | `910b75cfe9c9a1bc418e72c4b0258248f171445d65d044d894fba341f0a69cbf` | 2,496,531,328 B | GGUF v3 (Language Backbone) |
| `mmproj_mark_vl_2_2b.gguf` | `339278454bbb41e820357d31d43e968b365d864fbf9f30a3ecbc93a07911b72d` | 839,325,760 B | GGUF v3 (Vision Projector) |
| `mark_vl_2_2b_q4_k_m.gguf.part-00` | `bf37345268974a98da98e35a2acfeba55effc0528c50af9f69e24427aedcfe81` | 1,572,864,000 B | Multipart Release Archive |
| `mark_vl_2_2b_q4_k_m.gguf.part-01` | `90180a684d88cab1242fb9bbab467f472d2e700c35a5e29d3db17ba9e2e1f845` | 923,667,328 B | Multipart Release Archive |
| `Ollama Manifest Blob` | `7be5e1025127b113f4f99713d6cbe60426e2d1b0881e62f968598f845caf5687` | 1,452 B | JSON Layer Manifest |
| `Ollama Config Blob` | `44fa841c009f3f8b3a9c676da5224f77496f50a4ef975df80ca087fe19afa76b` | 290 B | JSON Architecture Config |

---

## 6. License & Distribution

Released under the GNU Affero General Public License v3.0 (AGPL-3.0).
Model weights are hosted on Hugging Face Hub (miceworks/Mark-VL-2.2B) and official GitHub Release (https://github.com/thu12dta3/Mark-AI/releases/tag/v1.0.0-markvl-2.2b).
