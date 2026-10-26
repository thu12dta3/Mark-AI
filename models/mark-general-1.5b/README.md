# Mark General 1.5B (Trained & Preserved Baseline)

## Model Overview
This directory preserves the official configuration and specification of **Mark General 1.5B**, trained to convergence on NVIDIA A100-SXM4-80GB and retained as the frozen foundation baseline in the Mark AI ecosystem.

## Training Convergence & Artifacts
- **Training Cluster:** NVIDIA A100-SXM4-80GB (High-RAM Colab Pro Enterprise)
- **Training Steps:** 600 full steps completed
- **Convergence Loss:** Reduced from `11.0529` to `2.8743` (stable asymptotic convergence)
- **Exported Formats:**
  - `mark_general_1_5b_inference.pt` (2,874.07 MB BF16)
  - `mark_general_1_5b_iq4_nl.pt` (823.84 MB native 4-bit non-linear quantization)
- **Preservation Status:** Permanently archived. Never deleted or overwritten by newer versions.

## Technical Specifications
- **Total Parameters:** 1,501,680,656 (1.502B)
- **Active Parameters:** ~315M FLOPs per token (Top-2 routed + 1 shared)
- **Attention:** Multi-Head Latent Attention (MLA, c_KV = 320)
- **DeepSeekMoE:** 16 Routed Experts + 1 Shared Expert (moe_dim = 1504)
- **Memory Footprint (IQ4_NL):** ~823 MB on disk | ~1.25 GB VRAM on NVIDIA RTX 2060 Super
- **Core Domains:** 10 Official MARK Cognition Domains, Advanced STEM, Perplexity-style Self-RAG, Epistemic [UNKNOWN] Verification.

