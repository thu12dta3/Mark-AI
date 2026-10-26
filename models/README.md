# Mark AI - Models Hub (LM Studio, Ollama & llama.cpp Plug-and-Play)

This directory centralizes architectural definitions, configuration files, and deployment templates for running Mark AI models directly in **LM Studio**, **Ollama**, **llama.cpp**, and **vLLM**.

---

## Available Models

| Model | Parameters | Format | Local VRAM | Status | Directory |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Mark Engineer 300M (302M)** | **302M** | **GGUF / IQ4_NL** | **~350 MB** | **Available (Production)** | [`mark-engineer-302m/`](./mark-engineer-302m/) |
| **Mark Engineer 1.5B** | **1.50B** | **GGUF / IQ4_NL** | **~1.25 GB** | **Training on A100** | [`mark-engineer-1.5b/`](./mark-engineer-1.5b/) |
| **Mark General 1.5B** | **1.50B** | **GGUF / IQ4_NL** | **~1.25 GB** | **Training on A100** | [`mark-general-1.5b/`](./mark-general-1.5b/) |

---

## 1. How to Use in LM Studio (Plug-and-Play)

### Method 1: Direct Download of Compiled GGUF
Download the 302M model directly into your local LM Studio models folder:

```bash
# Create the destination directory for LM Studio
mkdir -p ~/.lmstudio/models/Mark-AI/mark-engineer-302m

# Download the compiled GGUF directly from the official release
curl -L -o ~/.lmstudio/models/Mark-AI/mark-engineer-302m/mark_engineer_302m.gguf \
  https://github.com/thu12dta3/Mark-AI/releases/download/v1.0.0-302m/mark_engineer_302m_iq4_nl.gguf
```

### Method 2: Loading in UI
1. Open **LM Studio**.
2. Select **Mark-AI / mark-engineer-302m** from the local models dropdown.
3. The model loads instantly with only **~350 MB of VRAM** consumption.

---

## 2. How to Use in Ollama

Navigate to the model directory and build the container:

```bash
cd models/mark-engineer-302m
ollama create mark-engineer -f Modelfile
ollama run mark-engineer "Optimize this concurrent algorithm in Rust."
```

---

## 3. How to Use in llama.cpp

```bash
./llama-cli \
  -m ~/.lmstudio/models/Mark-AI/mark-engineer-302m/mark_engineer_302m.gguf \
  -p "User: Analyze this C++ code for race conditions.\nAssistant: <think>" \
  -n 256
```
