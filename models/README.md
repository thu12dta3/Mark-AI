# Mark AI - Models Hub (LM Studio, Ollama & llama.cpp Plug-and-Play)

This directory centralizes architectural definitions, configuration files, tokenizers, and deployment templates for running Mark AI models directly in **LM Studio**, **Ollama**, **llama.cpp**, and **vLLM**.

---

## Available Models

| Model | Parameters | Format | Local VRAM (RTX 2060S) | Specialization & Operational Status | Directory |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **Mark Engineer 1.5B** | **1.50B** | **GGUF / IQ4_NL** | **~1.25 GB** | **Production Flagship:** Systems programming, kernel internals (eBPF, KMDF), Tokio, C++20 & Tech Lead dialogue. | [`mark-engineer-1.5b/`](./mark-engineer-1.5b/) |
| **Mark General 1.5B** | **1.50B** | **GGUF / IQ4_NL** | **~1.25 GB** | **Preserved Baseline:** 10 STEM & Scientific Cognition Domains, Epistemic Self-RAG (`[UNKNOWN]`). | [`mark-general-1.5b/`](./mark-general-1.5b/) |
| **MarkVL-2.2B** | **2.22B** | **GGUF / Q4_K_M** | **~1.85 GB** | **Perception Specialist:** AnyRes 2048x2048, PixelShuffle 2x2 C-Abstractor, Micro-OCR, FACS. | [`mark-vl-2.2b/`](./mark-vl-2.2b/) |
| **Mark Engineer 300M (302M)** | **302M** | **GGUF / IQ4_NL** | **~350 MB** | **Experimental Edge Testbed:** Non-conversational, high hallucination risk in open chat, passes rigid synthetic micro-tests. | [`mark-engineer-302m/`](./mark-engineer-302m/) |

---

## 1. How to Use in LM Studio (Plug-and-Play)

### Option A: Mark Engineer 1.5B (Senior Systems & Conversational)
```bash
# 1. Create destination directory
mkdir -p ~/.lmstudio/models/Mark-AI/mark-engineer-1.5b

# 2. Download the compiled GGUF
curl -L -o ~/.lmstudio/models/Mark-AI/mark-engineer-1.5b/mark_engineer_1_5b.gguf \
  https://github.com/thu12dta3/Mark-AI/releases/download/v1.0.0-engineer-1.5b/mark_engineer_1_5b_iq4_nl.gguf
```

### Option B: Mark General 1.5B (Preserved Baseline - General Knowledge & STEM)
```bash
# 1. Create destination directory
mkdir -p ~/.lmstudio/models/Mark-AI/mark-general-1.5b

# 2. Download the compiled GGUF
curl -L -o ~/.lmstudio/models/Mark-AI/mark-general-1.5b/mark_general_1_5b.gguf \
  https://github.com/thu12dta3/Mark-AI/releases/download/v1.0.0-general-1.5b/mark_general_1_5b_iq4_nl.gguf
```

### Option C: Mark Engineer 300M (Ultralight Embedded)
```bash
mkdir -p ~/.lmstudio/models/Mark-AI/mark-engineer-302m
curl -L -o ~/.lmstudio/models/Mark-AI/mark-engineer-302m/mark_engineer_302m.gguf \
  https://github.com/thu12dta3/Mark-AI/releases/download/v1.0.0-302m/mark_engineer_302m_iq4_nl.gguf
```

### Option D: MarkVL-2.2B (Multimodal Dense Visual Perception & Micro-OCR)
```bash
# 1. Download MarkVL weights and projector
bash scripts/download_markvl_2_2b.sh

# 2. Or download via Python utility
python scripts/download_model.py --model markvl-2.2b
```

---

## 2. How to Use in Ollama

Navigate to the target model directory and build the model:

```bash
# For MarkVL-2.2B (Dense Visual Perception Engine)
bash models/mark-vl-2.2b/install_markvl_ollama.sh
# Or create manually:
cd models/mark-vl-2.2b && ollama create markvl -f Modelfile
ollama run markvl "Descreva detalhadamente os componentes visiveis nesta imagem."

# For Mark Engineer 1.5B
cd models/mark-engineer-1.5b
ollama create mark-engineer-1.5b -f Modelfile
ollama run mark-engineer-1.5b "Analyze this Rust Tokio code for cross-await deadlocks."

# For Mark General 1.5B
cd models/mark-general-1.5b
ollama create mark-general-1.5b -f Modelfile
ollama run mark-general-1.5b "Explain SGLT2 inhibitor renal hemodynamics."

# For Mark Engineer 302M
cd models/mark-engineer-302m
ollama create mark-engineer -f Modelfile
ollama run mark-engineer "Optimize this C ringbuffer."
```

---

## 3. How to Use in llama.cpp

```bash
# Running Mark Engineer 1.5B
./llama-cli \
  -m ~/.lmstudio/models/Mark-AI/mark-engineer-1.5b/mark_engineer_1_5b.gguf \
  -p "User: Write an eBPF filter in C.\nAssistant: <think>" \
  -n 512

# Running MarkVL-2.2B with Multimodal Projector
./llama-minicpmv-cli \
  -m models/mark-vl-2.2b/mark_vl_2_2b.gguf \
  --mmproj models/mark-vl-2.2b/mmproj_mark_vl_2_2b.gguf \
  --image ./schematic.png \
  -p "Describe the circuit topology and pinouts."
```

---

## 4. Automated Python Downloader

The repository includes a standalone Python downloader script with SHA256 checksum verification:
```bash
# Download MarkVL-2.2B
python scripts/download_model.py --model markvl-2.2b

# Download Mark Engineer 1.5B
python scripts/download_model.py --model engineer-1.5b

# Download Mark General 1.5B
python scripts/download_model.py --model general-1.5b

# Download Mark Engineer 302M
python scripts/download_model.py --model 302m
```
