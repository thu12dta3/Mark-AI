# Mark Engineer 1.5B - LM Studio & Ollama Plug-and-Play

This model is ready for local execution across platforms such as **LM Studio**, **Ollama**, **llama.cpp**, and **vLLM**.

---

## 1. Plug-and-Play in LM Studio

### Step 1: Download the GGUF Asset
Download the single-file GGUF asset via curl:
```bash
curl -L -O https://github.com/thu12dta3/Mark-AI/releases/download/v1.0.0-engineer-1.5b/mark_engineer_1_5b_iq4_nl.gguf
```

### Step 2: Copy to LM Studio Models Directory
Copy or symlink the downloaded file into your LM Studio models folder:
```bash
mkdir -p ~/.lmstudio/models/Mark-AI/mark-engineer-1.5b
cp mark_engineer_1_5b_iq4_nl.gguf ~/.lmstudio/models/Mark-AI/mark-engineer-1.5b/mark_engineer_1_5b.gguf
```

### Step 3: Launch in LM Studio
1. Open **LM Studio**.
2. Select **Mark-AI / mark-engineer-1.5b** from the model dropdown.
3. The model loads with only **~1.25 GB VRAM** (leaving over 6.7 GB free on an 8 GB GPU like the RTX 2060 Super) and delivers over **125 tokens/s**.

---

## 2. Plug-and-Play in Ollama

In this directory, run:
```bash
ollama create mark-engineer-1.5b -f Modelfile
ollama run mark-engineer-1.5b "Write a Linux eBPF XDP filter in C that drops UDP packets without verifier rejection."
```

---

## 3. Plug-and-Play in llama.cpp

```bash
./llama-cli \
  -m ~/.lmstudio/models/Mark-AI/mark-engineer-1.5b/mark_engineer_1_5b.gguf \
  -p "User: Analyze this Windows KMDF driver DPC routine for IRQL violations.\nAssistant: <think>" \
  -n 512
```

---

## 4. Automatic Python Download Utility

You can also use the bundled repository script to download and verify the SHA256 checksum automatically:
```bash
python scripts/download_model.py --model engineer-1.5b
```

---

## 5. Technical Specifications

- **Total Parameters:** 1,501,680,656 (1.502B)
- **Active Parameters:** ~315M FLOPs per token (Top-2 routed + 1 shared)
- **Attention:** Multi-Head Latent Attention (MLA, c_KV = 320)
- **MoE:** 16 Routed Experts + 1 Shared Expert (moe_dim = 1504)
- **Speculative Generation:** Multi-Token Prediction (MTP k=2)
- **VRAM Footprint (RTX 2060 Super):** ~1.25 GB VRAM (823.84 MB file size)
- **Generation Speed:** 115 to 138 tokens/s (average 128.5 t/s)
- **Specialization:** Linux Kernels (eBPF, VFS), Windows (KMDF, IRQLs), macOS (Mach, IOKit), Rust Async (Tokio), C++20, and Senior Tech Lead Conversational Alignment.
