# Mark Engineer 300M (302M) - LM Studio & Ollama Plug-and-Play

This model is ready for local execution across platforms such as **LM Studio**, **Ollama**, **llama.cpp**, and **vLLM**.

---

## 1. Plug-and-Play in LM Studio

### Step 1: Download the GGUF Asset
Download the single-file GGUF asset via curl:
```bash
curl -L -O https://github.com/thu12dta3/Mark-AI/releases/download/v1.0.0-302m/mark_engineer_302m_iq4_nl.gguf
```

### Step 2: Copy to LM Studio Models Directory
Copy or symlink the downloaded file into your LM Studio models folder:
```bash
mkdir -p ~/.lmstudio/models/Mark-AI/mark-engineer-302m
cp mark_engineer_302m_iq4_nl.gguf ~/.lmstudio/models/Mark-AI/mark-engineer-302m/mark_engineer_302m.gguf
```

### Step 3: Launch in LM Studio
1. Open **LM Studio**.
2. Select **Mark-AI / mark-engineer-302m** from the model picker.
3. The model loads with only **~350 MB VRAM** and supports a 4,096 token context window.

---

## 2. Plug-and-Play in Ollama

In this directory, run:
```bash
ollama create mark-engineer -f Modelfile
ollama run mark-engineer "Optimize this concurrent algorithm in Rust with a lock-free ringbuffer."
```

---

## 3. Technical Specifications

- **Total Parameters:** 302,019,848 (302M)
- **Active Parameters:** ~110M per token
- **Attention:** Multi-Head Latent Attention (MLA)
- **MoE:** 16 Routed Experts + 1 Shared Expert
- **VRAM Footprint:** ~350 MB
- **Format:** GGUF (universal float16 / IQ4_NL runtime support)
