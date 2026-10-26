# Mark General 1.5B - LM Studio & Ollama Plug-and-Play

This directory provides the preserved baseline configuration and single-file download instructions for **Mark General 1.5B**, trained to convergence loss `2.8743` on NVIDIA A100-SXM4-80GB.

---

## 1. Plug-and-Play in LM Studio

### Step 1: Download the GGUF Asset
Download the single-file GGUF asset via curl:
```bash
curl -L -O https://github.com/thu12dta3/Mark-AI/releases/download/v1.0.0-general-1.5b/mark_general_1_5b_iq4_nl.gguf
```

### Step 2: Copy to LM Studio Models Directory
Copy or symlink the downloaded file into your LM Studio models folder:
```bash
mkdir -p ~/.lmstudio/models/Mark-AI/mark-general-1.5b
cp mark_general_1_5b_iq4_nl.gguf ~/.lmstudio/models/Mark-AI/mark-general-1.5b/mark_general_1_5b.gguf
```

### Step 3: Launch in LM Studio
1. Open **LM Studio**.
2. Select **Mark-AI / mark-general-1.5b** from the model dropdown.
3. The model loads with only **~1.25 GB VRAM** (leaving over 6.7 GB free on an 8 GB GPU like the RTX 2060 Super) and generates at over **120 tokens/s**.

---

## 2. Plug-and-Play in Ollama

In this directory, run:
```bash
ollama create mark-general-1.5b -f Modelfile
ollama run mark-general-1.5b "Explain the aerodynamic drag difference between G1 and G7 ballistic standards."
```

---

## 3. Plug-and-Play in llama.cpp

```bash
./llama-cli \
  -m ~/.lmstudio/models/Mark-AI/mark-general-1.5b/mark_general_1_5b.gguf \
  -p "User: Deduce the Eyring-Polanyi rate constant equation.\nAssistant: <think>" \
  -n 512
```

---

## 4. Automatic Python Download Utility

You can also use the bundled repository script to download and verify the SHA256 checksum automatically:
```bash
python scripts/download_model.py --model general-1.5b
```

---

## 5. Technical Specifications & Training Convergence

- **Total Parameters:** 1,501,680,656 (1.502B)
- **Active Parameters:** ~315M FLOPs per token (Top-2 routed + 1 shared)
- **Attention:** Multi-Head Latent Attention (MLA, c_KV = 320)
- **DeepSeekMoE:** 16 Routed Experts + 1 Shared Expert (moe_dim = 1504)
- **Speculative Generation:** Multi-Token Prediction (MTP k=2)
- **VRAM Footprint (RTX 2060 Super):** ~1.25 GB VRAM (823.84 MB file size)
- **Generation Speed:** 115 to 138 tokens/s
- **Epistemic Honesty:** Built-in Self-RAG with `[UNKNOWN]` and `[WEB_SEARCH]` tokens.
- **Training Cluster:** NVIDIA A100-SXM4-80GB (600 steps, loss converged from `11.05` to `2.87`).
- **Preservation Status:** Permanently archived. Never deleted or modified in future versions.
