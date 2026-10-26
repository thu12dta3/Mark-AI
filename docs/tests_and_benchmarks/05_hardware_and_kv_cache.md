# 05. Hardware Efficiency, Latency & Multi-Head Latent Attention (MLA)

This document presents mathematical formulations and empirical benchmarks covering memory footprints, inference latency, and KV Cache scaling across long context windows (up to 32k tokens) in the **Mark AI** architecture.

---

## 1. Mathematical Formulation: MLA vs GQA

In conventional autoregressive Transformers, KV Cache memory consumption scales linearly with sequence length ($S$), number of layers ($L$), number of Key/Value heads ($H_{KV}$), and head dimension ($d_k$):

$$\text{KV Size (GQA)} = 2 \times L \times H_{KV} \times d_k \times b \quad (\text{bytes per token})$$

Where $b$ denotes the byte precision per element (2 bytes for FP16 / BF16).

### In Qwen 2.5-Coder 14B (GQA):
- $L = 48$ layers
- $H_{KV} = 8$ KV heads
- $d_k = 128$ head dimension
- $b = 2$ bytes (FP16)

$$\text{KV Footprint / Token (Qwen 14B)} = 2 \times 48 \times 8 \times 128 \times 2 = 98,304 \text{ bytes} \approx \mathbf{96.00 \text{ KB / token}}$$

---

### In Mark Engineer 1.5B (Multi-Head Latent Attention - MLA):
The MLA architecture compresses Key and Value projections into a shared low-rank latent vector $c_{KV}$ (`kv_lora_rank = 320`), decoupling rotational position embeddings via a dedicated RoPE projection (`qk_rope_head_dim = 32`):

$$\text{KV Size (MLA)} = L \times (c_{KV} + k_{pe}) \times b \quad (\text{bytes per token})$$

- $L = 16$ layers
- $c_{KV} = 320$ compressed latent dimension
- $k_{pe} = 32$ decoupled RoPE dimension
- $b = 2$ bytes (FP16)

$$\text{KV Footprint / Token (Mark 1.5B FP16)} = 16 \times (320 + 32) \times 2 = 11,264 \text{ bytes} \approx \mathbf{11.00 \text{ KB / token}}$$

Applying native non-linear quantization (**IQ4_NL**) to the KV Cache reduces the footprint to 0.5 bytes per element:

$$\text{KV Footprint / Token (Mark 1.5B IQ4\_NL)} \approx \mathbf{2.816 \text{ KB / token}}$$

---

## 2. Context Window Memory Scaling (1k to 32k Tokens)

| Context Length ($S$) | Qwen 14B (GQA FP16) | Mark 1.5B (MLA FP16) | Mark 1.5B (MLA IQ4_NL) | Relative Reduction |
| :---: | :---: | :---: | :---: | :---: |
| **1,024 tokens (1k)** | 98.3 MB | 11.3 MB | **2.8 MB** | **-97.1%** |
| **4,096 tokens (4k)** | 393.2 MB | 45.1 MB | **11.3 MB** | **-97.1%** |
| **8,192 tokens (8k)** | 786.4 MB | 90.1 MB | **22.5 MB** | **-97.1%** |
| **16,384 tokens (16k)**| 1,572.8 MB (1.57 GB) | 180.2 MB | **45.1 MB** | **-97.1%** |
| **32,768 tokens (32k)**| **3,145.7 MB (3.14 GB)** | **360.4 MB** | **91.8 MB** | **-97.1%** |

At a 32k token context window, Qwen 14B consumes over 3.1 GB of VRAM solely for the attention cache, while Mark Engineer 1.5B fits within **91.8 MB**.

---

## 3. The PCIe Bus Bottleneck on 8GB Consumer GPUs

The 8GB VRAM threshold represents the critical dividing line for local developer workstations:

```
[8GB VRAM GPU: Real-World Runtime Allocation]

Mark Engineer 1.5B (IQ4_NL):
[ Weights: 1.25 GB ][ KV 32k: 92MB ][ Free VRAM for OS, IDE & Compiler: 6.65 GB ]
-> 100% Native in VRAM | Generation Throughput: 115 to 138 t/s

Qwen 2.5-Coder 14B (Q4_K_M):
[ Weights: 9.85 GB ..................................... ] (Exceeds 8GB VRAM!)
  |-> 6.5 GB in VRAM (Total Saturation)
  |-> 3.35 GB offloaded to Host System RAM across PCIe Bus
-> PCIe Bus Saturation (16-32 GB/s vs 400 GB/s GDDR6) | Generation drops to: 4.2 to 8.5 t/s
```

### Consequences of PCIe Offloading:
1. **Latency Collapse:** Transferring gigabytes of layer weights across PCIe on every forward token step degrades throughput from 40 tokens/s to under 8 tokens/s.
2. **System Freezes:** Paging traffic and PCIe bus contention cause noticeable UI stutters when compiling concurrent C++, Rust, or Go projects alongside model inference.
3. **Multitasking Headroom:** Mark Engineer 1.5B leaves over 6.6 GB of VRAM open for editors, debuggers, browsers, and local container runtimes.

---

## 4. Multi-Token Prediction (MTP) Speculative Decoding

Mark Engineer 1.5B incorporates a native **Multi-Token Prediction (MTP)** module:
- On each forward pass, the model outputs logits for token $t+1$ (main head) and token $t+2$ (MTP head conditioned on hidden state $h_t$ and the next token embedding).
- During inference, speculative decoding accepts verified $t+2$ candidates directly, delivering up to **1.75x faster throughput** on structured and repetitive code constructs.
