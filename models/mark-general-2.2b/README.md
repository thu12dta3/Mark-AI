# Mark General 2.2B Multimodal Vision (Mark-VL 2.0)

## Overview
**Mark General 2.2B Multimodal Vision** is the second-generation flagship vision-language model of the Mark AI ecosystem. It scales the architecture from 1.50B to **2.208B total parameters** by expanding the Mixture-of-Experts grid to **24 fine-grained routed experts + 1 shared expert**, while incorporating an advanced **AnyRes Dynamic Resolution Vision Transformer with 2D-RoPE and PixelShuffle 2x2 C-Abstractor**.

## Key Architectural Advancements over Mark General 1.5B
1. **Capacity Expansion (1.50B -> 2.208B Total Parameters):**
   - 24 Domain-Routed Experts (up from 16) + 1 Shared Expert.
   - Activates only Top-2 experts per token (~450M FLOPs per token), maintaining ultralight inference speed.
2. **Native Multimodal Vision System (Qwen-VL Competitor):**
   - **AnyRes Dynamic Patching:** Preserves native aspect ratio up to 2048x2048 without distortion or downsampling artifacts (crucial for PCB traces, circuit schematics, architectural blueprints, and high-density OCR).
   - **2D Spatial RoPE:** Visual tokens retain physical `(X, Y)` coordinate awareness.
   - **PixelShuffle 2x2 C-Abstractor:** Compresses 4 visual patches into 1 dense token (`75%` visual token reduction), enabling instant image processing.
3. **Cold-Start Visual Chain-of-Thought (`<think> ... </think>`):**
   - Generates internal visual reasoning traces before synthesizing the final output, dramatically reducing hallucinations in complex diagrams and math proofs.
4. **Hardware Calibration (NVIDIA GeForce RTX 2060 Super 8GB):**
   - Footprint in native `IQ4_NL`: **~1.85 GB VRAM** (leaves > 6 GB VRAM free for operating system, context, and desktop tasks).
   - Generation throughput: **95 to 125 tokens/s** via native Multi-Token Prediction (`k=2`).

## Preservation of Baseline
The original **Mark General 1.5B** checkpoint (trained to loss `2.8743` on NVIDIA A100) remains permanently preserved in [`../mark-general-1.5b/`](../mark-general-1.5b/) as the frozen text-only baseline.
