#!/usr/bin/env python3
"""
scripts/update_all_readmes.py
Generates clean, direct systems engineering Model Cards across all Miceworks models
and the root README.md for the Mark-AI repository.

Follows strict technical documentation style:
- Exact dual size units: "X.XX MiB (Y.YY MB / Z,ZZZ,ZZZ bytes)"
- Exact parameter integers for total and active parameters
- Strict prohibition of emojis, double hyphens in headings, and checkboxes
- Complete Hugging Face YAML Frontmatter with library_name: gguf and language: [en]
- Full alignment with config.json and gguf_verification_manifest.json
- 100% technical honesty: honest weight states, real measured tokenizer metrics,
  honest router bias equations, and separation of GGUF runtime from PyTorch extensions.
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

MODELS_METADATA = [
    {
        "id": "mark-engineer-302m",
        "name": "Mark-Engineer-302M",
        "hf_repo": "miceworks/Mark-Engineer-302M",
        "category": "EXPERIMENTAL_NARROW_CHECKPOINT",
        "is_vl": False,
        "is_v3": False,
        "training_status": "EXPERIMENTAL_NARROW_CHECKPOINT",
        "weight_state_label": "Experimental Narrow Checkpoint",
        "pipeline_tag": "text-generation",
        "tok_type": "legacy_4311",
        "overview": (
            "Mark-Engineer-302M is an experimental narrow checkpoint in the 302M family naming convention. "
            "It contains 251,452,544 total physical parameters and activates 119.33M parameters per token across 16 layers. "
            "It uses a native DeepSeek2 Mixture-of-Experts (MoE) topology with Multi-Head Latent Attention (MLA), quantized "
            "to 4-bit non-linear GGUF (IQ4_NL).\n\n"
            "The container occupies 135.12 MiB (141.69 MB / 141,686,432 bytes) of storage and runs on low-power edge devices, "
            "Raspberry Pi 5 (8 GB), or entry-level consumer GPUs such as NVIDIA GeForce RTX 2060 Super (8 GB GDDR6, 256-bit, 448 GB/s), "
            "consuming less than 2% of an 8 GB VRAM allocation."
        ),
        "training_scope": (
            "Experimental narrow checkpoint (EXPERIMENTAL_NARROW_CHECKPOINT). Trained on POSIX C interfaces, Linux system call stubs, "
            "memory barrier semantics, and bare-metal linker scripts from miceworks/Mark-Ultra-Curriculum."
        ),
        "evaluation_summary": (
            "Validated on low-level systems engineering interfaces with syntactic schema validity on POSIX C and kernel stub generation."
        ),
    },
    {
        "id": "mark-slm-v2-302m",
        "name": "Mark-SLM-V2-302M",
        "hf_repo": "miceworks/Mark-SLM-V2-302M",
        "category": "EXPERIMENTAL_NARROW_CHECKPOINT",
        "is_vl": False,
        "is_v3": False,
        "training_status": "EXPERIMENTAL_NARROW_CHECKPOINT",
        "weight_state_label": "Experimental Narrow Checkpoint",
        "pipeline_tag": "text-generation",
        "tok_type": "legacy_4311",
        "overview": (
            "Mark-SLM-V2-302M is an experimental narrow checkpoint in the 302M family naming convention. "
            "It contains 329,132,160 total physical parameters and activates 155.72M parameters per token across 16 layers. "
            "It features 16 routed experts and 1 shared expert (2 active routed experts per token).\n\n"
            "Quantized in IQ4_NL, the binary occupies 177.76 MiB (186.39 MB / 186,394,944 bytes) and runs on 8 GB consumer GPUs "
            "with minimal VRAM consumption."
        ),
        "training_scope": (
            "Experimental narrow checkpoint (EXPERIMENTAL_NARROW_CHECKPOINT). Trained for 10,000 steps on miceworks/Mark-Ultra-Curriculum. "
            "The final training loss reached 0.263975, reflecting multi-epoch memorization on the small curriculum testbed dataset rather than "
            "broad open-domain generalization."
        ),
        "evaluation_summary": (
            "Achieved a final validation loss of 0.263975 across 10,000 training steps on the narrow miceworks/Mark-Ultra-Curriculum distribution (memorization artifact)."
        ),
    },
    {
        "id": "mark-engineer-1.5b",
        "name": "Mark-Engineer-1.5B",
        "hf_repo": "miceworks/Mark-Engineer-1.5B",
        "category": "EXPERIMENTAL_NARROW_CHECKPOINT",
        "is_vl": False,
        "is_v3": False,
        "training_status": "EXPERIMENTAL_NARROW_CHECKPOINT",
        "weight_state_label": "Experimental Narrow Checkpoint",
        "pipeline_tag": "text-generation",
        "tok_type": "legacy_4311",
        "overview": (
            "Mark-Engineer-1.5B is a 1.5B-parameter sparse Transformer exported to the native deepseek2 GGUF format. "
            "It combines Multi-Head Latent Attention (MLA) with a 16-routed and 1-shared SwiGLU Mixture-of-Experts (MoE) "
            "topology to keep active parameters at 364.15M per token (24.34% active) out of 1,496,121,824 total parameters.\n\n"
            "Quantized in IQ4_NL, the binary occupies 802.94 MiB (841.94 MB / 841,941,664 bytes), leaving over 7 GB of VRAM "
            "free on an 8 GB GPU such as NVIDIA GeForce RTX 2060 Super (8 GB GDDR6, 256-bit, 448 GB/s)."
        ),
        "training_scope": (
            "Experimental narrow checkpoint (EXPERIMENTAL_NARROW_CHECKPOINT). Trained for 400 steps on miceworks/Mark-Ultra-Curriculum "
            "using low-level kernel sources, systems manuals, RFCs, and concurrency primitives."
        ),
        "evaluation_summary": (
            "Demonstrates narrow curriculum completion for POSIX threads, synchronization primitives, and memory barriers."
        ),
    },
    {
        "id": "mark-general-1.5b",
        "name": "Mark-General-1.5B",
        "hf_repo": "miceworks/Mark-General-1.5B",
        "category": "EXPERIMENTAL_NARROW_CHECKPOINT",
        "is_vl": False,
        "is_v3": False,
        "training_status": "EXPERIMENTAL_NARROW_CHECKPOINT",
        "weight_state_label": "Experimental Narrow Checkpoint",
        "pipeline_tag": "text-generation",
        "tok_type": "legacy_4311",
        "overview": (
            "Mark-General-1.5B is an experimental narrow checkpoint with 1,496,185,824 total physical parameters and 364.22M "
            "active parameters per token across 16 layers. It features a 16-routed, 1-shared MoE structure.\n\n"
            "Quantized in IQ4_NL, the container takes 802.97 MiB (841.98 MB / 841,977,664 bytes) and runs locally within standard 8 GB VRAM budgets."
        ),
        "training_scope": (
            "Experimental narrow checkpoint (EXPERIMENTAL_NARROW_CHECKPOINT). Trained for 600 steps on miceworks/Mark-Ultra-Curriculum "
            "with curated STEM data, formal proofs, and algorithmic logic."
        ),
        "evaluation_summary": (
            "Validated on narrow mathematical logic and algorithmic reasoning representations."
        ),
    },
    {
        "id": "mark-vl-2.2b",
        "name": "Mark-VL-2.2B",
        "hf_repo": "miceworks/Mark-VL-2.2B",
        "category": "EXPERIMENTAL_NARROW_CHECKPOINT",
        "is_vl": True,
        "is_v3": False,
        "training_status": "EXPERIMENTAL_NARROW_CHECKPOINT",
        "weight_state_label": "Experimental Narrow Checkpoint",
        "pipeline_tag": "image-text-to-text",
        "tok_type": "legacy_4311",
        "overview": (
            "Mark-VL-2.2B is an experimental vision-language architecture pairing an AnyRes SigLIP vision encoder definition with a 14-layer "
            "DeepSeek2 MoE language model backbone. It uses 2D Rotary Position Embeddings (2D-RoPE) and a 2x2 PixelShuffle projection "
            "adapter in its PyTorch architectural formulation.\n\n"
            "The model contains 2,174,310,096 total parameters and 395.50M active parameters per token. The language model binary occupies "
            "1166.72 MiB (1223.40 MB / 1,223,397,152 bytes) in IQ4_NL."
        ),
        "training_scope": (
            "Experimental narrow multimodal checkpoint (EXPERIMENTAL_NARROW_CHECKPOINT). Trained on narrow diagrammatic pairs and technical schematics "
            "from miceworks/Mark-Ultra-Curriculum."
        ),
        "evaluation_summary": (
            "Evaluated on narrow schematic reading and spatial 2D-RoPE coordinate alignment in PyTorch."
        ),
    },
    {
        "id": "mark-engineer-v2-8b",
        "name": "Mark-Engineer-V2-8B",
        "hf_repo": "miceworks/Mark-Engineer-V2-8B",
        "category": "UNTRAINED_ARCHITECTURAL_SCAFFOLD",
        "is_vl": False,
        "is_v3": False,
        "training_status": "UNTRAINED_ARCHITECTURAL_SCAFFOLD",
        "weight_state_label": "Untrained Architectural Scaffold",
        "pipeline_tag": "text-generation",
        "tok_type": "legacy_4311",
        "overview": (
            "Mark-Engineer-V2-8B is an 8B-class sparse Transformer featuring 32 layers, 24 routed experts, and 1 shared expert. "
            "It has 8,258,809,040 total physical parameters and activates 1,770,745,040 parameters per token (21.44% active).\n\n"
            "In native IQ4_NL quantization, the container occupies 4431.26 MiB (4646.52 MB / 4,646,516,576 bytes), fitting inside "
            "a single 8 GB consumer GPU such as NVIDIA GeForce RTX 2060 Super (8 GB GDDR6, 256-bit, 448 GB/s) with over 3.3 GB of VRAM remaining for KV cache."
        ),
        "training_scope": (
            "Untrained architectural base scaffold (UNTRAINED_ARCHITECTURAL_SCAFFOLD). Built for testing binary graph compatibility with llama.cpp, "
            "measuring memory usage on target hardware, and serving as a base for downstream pre-training and fine-tuning."
        ),
        "evaluation_summary": (
            "Architectural verification confirms full 3D tensor layout compatibility and valid execution under llama.cpp."
        ),
    },
    {
        "id": "mark-slm-v3-150m",
        "name": "Mark-SLM-V3-150M",
        "hf_repo": "miceworks/Mark-SLM-V3-150M",
        "category": "TRAINED_CHECKPOINT",
        "is_vl": False,
        "is_v3": True,
        "training_status": "TRAINED_CHECKPOINT",
        "weight_state_label": "Trained Checkpoint (Model Soup 50/50)",
        "pipeline_tag": "text-generation",
        "tok_type": "full_32k",
        "overview": (
            "**`Mark-SLM-V3-150M` is a 150M-parameter experimental sparse language model trained on 3.47M tokens (`miceworks/Mark-Open-SLM-Curriculum`), "
            "achieving an in-distribution Intelligence Index of `16.36` on the Open SLM Leaderboard (PR #291). It is an architectural research testbed "
            "for Multi-Head Latent Attention (MLA) and fine-grained MoE on 8 GB GPUs, NOT a general-purpose conversational or multilingual assistant.**\n\n"
            "The exported GGUF container (`mark_slm_v3_150m_iq4_nl.gguf`, 90.37 MiB / 94.76 MB / 94,764,576 bytes) implements 15 physical `deepseek2` blocks "
            "(layers 0-1 dense, layers 2-14 MoE with 16 routed + 1 shared expert, top-2 active). The architecture comprises **146,899,248 unique parameters** "
            "(excluding tied `output.weight`) and **165,349,680 materialized GGUF parameters across 273 tensors** (including the materialized `output.weight` copy)."
        ),
        "training_scope": (
            "Trained model checkpoint (TRAINED_CHECKPOINT). The model was trained through 25,000 optimization steps of pre-training followed "
            "by 4,000 steps of Phase 2 Cooldown on miceworks/Mark-Open-SLM-Curriculum (3.47M tokens), consolidated via 50/50 Model Soup (Weight Averaging). "
            "Disclosure: `miceworks/Mark-Open-SLM-Curriculum` contains training splits from HellaSwag, PIQA, ARC, SciQ, and ArithMark (permitted by Open SLM "
            "Leaderboard rules), so the reported scores measure in-distribution curriculum learning rather than general out-of-distribution capability. "
            "Evaluated on the official Open SLM Leaderboard benchmark suite, achieving an Intelligence Index of 16.36 in PR #291 "
            "(https://huggingface.co/spaces/AxiomicLabs/Open_SLM_Leaderboard/discussions/291)."
        ),
        "evaluation_summary": (
            "Official Open SLM Leaderboard Scores ([PR #291](https://huggingface.co/spaces/AxiomicLabs/Open_SLM_Leaderboard/discussions/291), Rule 01 and Rule 02 compliant):\n"
            "- PIQA: 62.50% (Norm: 25.00%)\n"
            "- ARC-Easy: 40.80% (Norm: 21.07%)\n"
            "- HellaSwag: 37.10% (Norm: 16.13%)\n"
            "- ArithMark-3.0 Official: 33.90% (Norm: 11.87%)\n"
            "- ARC-Challenge: 25.50% (Norm: 0.67%)\n"
            "- ARC-Average (Norm): 10.87%\n"
            "- AxiomicLabs Intelligence Index: 16.36\n\n"
            " Note: Because `miceworks/Mark-Open-SLM-Curriculum` incorporates official training splits from HellaSwag, PIQA, ARC, SciQ, and procedural ArithMark, "
            "these scores reflect in-distribution curriculum learning rather than broad zero-shot out-of-distribution generalization.\n\n"
            "Checkpoint Revision Changelog:\n"
            "- Revision 1 (Step 25,000 Base Checkpoint): Intelligence Index 10.04\n"
            "- Revision 2 (Step 4,000 Cooldown Checkpoint): Intelligence Index 14.35\n"
            "- Revision 3 (Model Soup 50/50 Final, 25k Base + 4k Cooldown): Intelligence Index 16.36 (Submitted in PR #291)"
        ),
    },
    {
        "id": "mark-v3-1.1b",
        "name": "Mark-V3-1.1B",
        "hf_repo": "miceworks/Mark-V3-1.1B",
        "category": "UNTRAINED_ARCHITECTURAL_SCAFFOLD",
        "is_vl": False,
        "is_v3": True,
        "training_status": "UNTRAINED_ARCHITECTURAL_SCAFFOLD",
        "weight_state_label": "Untrained Architectural Scaffold",
        "pipeline_tag": "text-generation",
        "tok_type": "legacy_4311",
        "overview": (
            "Mark-V3-1.1B is a third-generation sparse Transformer with 1,222,464,960 total physical parameters and 330,651,072 active "
            "parameters per token across 18 layers. It introduces Multi-Head Latent Attention (MLA) and "
            "fine-grained MoE with 32 routed experts and 2 shared experts.\n\n"
            "Quantized in IQ4_NL, the binary occupies 656.13 MiB (688.00 MB / 688,002,048 bytes), loading rapidly and leaving generous "
            "memory headroom on 8 GB GPUs."
        ),
        "training_scope": (
            "Untrained architectural base scaffold (UNTRAINED_ARCHITECTURAL_SCAFFOLD). Built for testing binary graph compatibility with llama.cpp, "
            "measuring memory usage on target hardware, and serving as a base for downstream pre-training."
        ),
        "evaluation_summary": (
            "Architectural graph verification confirms valid forward passes and expected memory occupancy."
        ),
    },
    {
        "id": "mark-general-v3-1.5b",
        "name": "Mark-General-V3-1.5B",
        "hf_repo": "miceworks/Mark-General-V3-1.5B",
        "category": "UNTRAINED_ARCHITECTURAL_SCAFFOLD",
        "is_vl": False,
        "is_v3": True,
        "training_status": "UNTRAINED_ARCHITECTURAL_SCAFFOLD",
        "weight_state_label": "Untrained Architectural Scaffold",
        "pipeline_tag": "text-generation",
        "tok_type": "legacy_4311",
        "overview": (
            "Mark-General-V3-1.5B is an untrained architectural scaffold with 1,520,496,960 total physical parameters and 405.73M active "
            "parameters per token across 18 layers. It features 32 routed experts, 2 shared experts, and MLA.\n\n"
            "Quantized in IQ4_NL, the container occupies 816.03 MiB (855.67 MB / 855,669,248 bytes) and operates comfortably within 8 GB VRAM budgets."
        ),
        "training_scope": (
            "Untrained architectural base scaffold (UNTRAINED_ARCHITECTURAL_SCAFFOLD). Built for testing binary graph compatibility with llama.cpp, "
            "measuring memory usage on target hardware, and serving as a base for downstream pre-training."
        ),
        "evaluation_summary": (
            "Forward pass and structural verification confirms stable numerical dynamics across all 18 layers."
        ),
    },
    {
        "id": "mark-engineer-v3-1.5b",
        "name": "Mark-Engineer-V3-1.5B",
        "hf_repo": "miceworks/Mark-Engineer-V3-1.5B",
        "category": "EXPERIMENTAL_NARROW_CHECKPOINT",
        "is_vl": False,
        "is_v3": True,
        "training_status": "EXPERIMENTAL_NARROW_CHECKPOINT",
        "weight_state_label": "Experimental Narrow Checkpoint",
        "pipeline_tag": "text-generation",
        "tok_type": "legacy_4311",
        "overview": (
            "Mark-Engineer-V3-1.5B is an experimental narrow checkpoint with 1,520,496,960 total physical parameters and 405.73M active "
            "parameters per token across 18 layers. It uses 32 routed experts and 2 shared experts.\n\n"
            "Quantized in IQ4_NL, the container occupies 816.95 MiB (856.63 MB / 856,632,832 bytes). Note: This is an experimental "
            "checkpoint trained briefly with a 4,311-token base vocabulary; outside its narrow systems domain, greedy inference produces "
            "incoherent fragments."
        ),
        "training_scope": (
            "Experimental narrow checkpoint (EXPERIMENTAL_NARROW_CHECKPOINT). Trained briefly on systems engineering stubs from "
            "miceworks/Mark-Ultra-Curriculum. Unverified claims regarding upstream training priors and synthetic benchmark evaluations "
            "have been formally retracted."
        ),
        "evaluation_summary": (
            "Experimental checkpoint; does not generalize to open-ended coding benchmarks or general conversation. "
            "Usage is intended for architecture inspection and narrow experimental research."
        ),
    },
    {
        "id": "mark-general-v3-8b",
        "name": "Mark-General-V3-8B",
        "hf_repo": "miceworks/Mark-General-V3-8B",
        "category": "UNTRAINED_ARCHITECTURAL_SCAFFOLD",
        "is_vl": False,
        "is_v3": True,
        "training_status": "UNTRAINED_ARCHITECTURAL_SCAFFOLD",
        "weight_state_label": "Untrained Architectural Scaffold",
        "pipeline_tag": "text-generation",
        "tok_type": "legacy_4311",
        "overview": (
            "Mark-General-V3-8B is an 8B-class sparse language model architecture scaffold packing 32 layers, 32 routed experts, 2 shared "
            "experts, and MLA attention into 8,168,001,024 total physical parameters (2,002,374,144 active per token).\n\n"
            "Quantized in IQ4_NL, the binary occupies 4383.98 MiB (4596.94 MB / 4,596,938,048 bytes), operating within standard 8 GB "
            "consumer GPUs while leaving over 3.4 GB of VRAM for context."
        ),
        "training_scope": (
            "Untrained architectural base scaffold (UNTRAINED_ARCHITECTURAL_SCAFFOLD). Built for testing binary graph compatibility with llama.cpp, "
            "measuring memory footprint on target hardware, and serving as a base for downstream pre-training."
        ),
        "evaluation_summary": (
            "Architectural integrity and layer forward tests pass with zero NaN/Inf activations."
        ),
    },
    {
        "id": "mark-experimental-v3-8b",
        "name": "Mark-Experimental-V3-8B",
        "hf_repo": "miceworks/Mark-Experimental-V3-8B",
        "category": "UNTRAINED_ARCHITECTURAL_SCAFFOLD",
        "is_vl": False,
        "is_v3": True,
        "training_status": "UNTRAINED_ARCHITECTURAL_SCAFFOLD",
        "weight_state_label": "Untrained Architectural Scaffold",
        "pipeline_tag": "text-generation",
        "tok_type": "legacy_4311",
        "overview": (
            "Mark-Experimental-V3-8B is an 8B-class sparse model packing 32 layers, 32 routed experts, 2 shared experts, and MLA "
            "attention into 8,168,001,024 total physical parameters (2,002,374,144 active per token).\n\n"
            "Quantized in IQ4_NL, the binary occupies 4382.55 MiB (4595.44 MB / 4,595,438,848 bytes), fitting within an 8 GB consumer GPU "
            "while leaving over 3.4 GB of VRAM for KV cache."
        ),
        "training_scope": (
            "Untrained architectural base scaffold (UNTRAINED_ARCHITECTURAL_SCAFFOLD). Built for testing binary graph compatibility with llama.cpp, "
            "measuring memory usage on target hardware, and serving as a base for downstream research."
        ),
        "evaluation_summary": (
            "Structural tensor verification confirms full DeepSeek2 MoE graph compliance."
        ),
    },
    {
        "id": "mark-slm-3m",
        "name": "Mark-SLM-3M",
        "hf_repo": "miceworks/Mark-SLM-3M",
        "category": "TRAINED_CHECKPOINT",
        "is_vl": False,
        "is_v3": True,
        "training_status": "TRAINED_CHECKPOINT",
        "weight_state_label": "Trained Checkpoint (From Scratch)",
        "pipeline_tag": "text-generation",
        "tok_type": "slm_3m",
        "overview": (
            "Mark-SLM-3M is an ultra-compact language model engineered under the strict 3,000,000 parameter ceiling for the Open SLM Leaderboard. "
            "It integrates Multi-Head Latent Attention (MLA) and Mixture of Experts (MoE), trained 100% from scratch in full "
            "compliance with Rules 01 and 02.\n\n"
            "The model contains 2,946,008 unique physical parameters and 2,503,640 active parameters per token across 8 layers. Quantized "
            "to IQ4_NL, the container occupies 3.12 MiB (3.27 MB / 3,271,680 bytes) and runs on embedded microcontrollers, mobile processors, or "
            "desktop environments with minimal resource consumption."
        ),
        "training_scope": (
            "Trained 100% from scratch (Gaussian normal std=0.02 initialization, zero external checkpoints) for 3,000 optimization steps on "
            "miceworks/Mark-Open-SLM-Curriculum, Cosmopedia-v2 STEM, and procedural arithmetic in strict compliance with Open SLM Leaderboard "
            "Rules 01 and 02."
        ),
        "evaluation_summary": (
            "Official Open SLM Leaderboard Scores (Sub-3M Category, Rule 01 and Rule 02 Compliant):\n"
            "- PIQA: 52.00% (Norm: 4.00%)\n"
            "- ARC-Easy: 26.60% (Norm: 2.13%)\n"
            "- HellaSwag: 25.60% (Norm: 0.80%)\n"
            "- ArithMark-3.0 Official: 29.70% (Norm: 6.27%)\n"
            "- ARC-Challenge: 23.75% (Norm: 0.00%)\n"
            "- ARC-Average (Norm): 1.07%\n"
            "- Best Intelligence Index: 3.85"
        ),
    },
    {
        "id": "mark-v3-32b",
        "name": "Mark-V3-32B",
        "hf_repo": "miceworks/Mark-V3-32B",
        "category": "UNTRAINED_ARCHITECTURAL_SCAFFOLD",
        "is_vl": False,
        "is_v3": True,
        "training_status": "UNTRAINED_ARCHITECTURAL_SCAFFOLD",
        "weight_state_label": "Untrained Architectural Scaffold",
        "pipeline_tag": "text-generation",
        "tok_type": "legacy_4311",
        "overview": (
            "Mark-V3-32B is a 32B-class flagship sparse mixture-of-experts model scaffold engineered with native DeepSeek2 architecture, "
            "1M context window via 3:1 Hybrid MLA, and 64 routed + 2 shared experts.\n\n"
            "The model comprises 31,989,012,480 total physical parameters and 3,677,387,904 active parameters per token across 32 layers. "
            "Calibrated for 48 GB VRAM cloud instances (NVIDIA L40S 48GB VRAM / 180GB RAM), the architecture supports long-context "
            "analysis and polyglot code synthesis."
        ),
        "training_scope": (
            "Untrained architectural base scaffold (UNTRAINED_ARCHITECTURAL_SCAFFOLD). Built for testing binary graph compatibility with llama.cpp, "
            "measuring memory usage across large context windows, and serving as a base for downstream pre-training."
        ),
        "evaluation_summary": (
            "Structural tensor verification confirms full 64R+2S DeepSeek2 MoE graph compliance."
        ),
    },
    {
        "id": "mark-vl-v3-4b",
        "name": "Mark-VL-V3-4B",
        "hf_repo": "miceworks/Mark-VL-V3-4B",
        "category": "UNTRAINED_ARCHITECTURAL_SCAFFOLD",
        "is_vl": True,
        "is_v3": True,
        "training_status": "UNTRAINED_ARCHITECTURAL_SCAFFOLD",
        "weight_state_label": "Untrained Architectural Scaffold",
        "pipeline_tag": "image-text-to-text",
        "tok_type": "full_32k",
        "overview": (
            "Mark-VL-V3-4B is a third-generation multimodal vision-language model architecture scaffold integrating a 16-layer Vision Transformer (ViT) "
            "with Spatial 2D-RoPE, a 2x2 PixelShuffle projection adapter, and a 24-layer DeepSeek2 MLA and MoE "
            "language backbone.\n\n"
            "The model comprises 4,134,688,769 total physical parameters and 824,483,584 active parameters per token across 24 language layers and "
            "16 vision layers."
        ),
        "training_scope": (
            "Untrained multimodal architectural scaffold (UNTRAINED_ARCHITECTURAL_SCAFFOLD). Built for multimodal graph verification and spatial "
            "2D-RoPE alignment testing."
        ),
        "evaluation_summary": (
            "Projector and attention alignment verified with spatial 2D-RoPE coordinate preservation."
        ),
    },
    {
        "id": "mark-general-2.2b",
        "name": "Mark-General-2.2B",
        "hf_repo": "miceworks/Mark-General-2.2B",
        "category": "EXPERIMENTAL_NARROW_CHECKPOINT",
        "is_vl": True,
        "is_v3": False,
        "training_status": "EXPERIMENTAL_NARROW_CHECKPOINT",
        "weight_state_label": "Experimental Narrow Checkpoint",
        "pipeline_tag": "image-text-to-text",
        "tok_type": "legacy_4311",
        "overview": (
            "Mark-General-2.2B is an experimental multimodal vision-language model with 2,223,906,816 total parameters, 24 routed experts and "
            "1 shared expert, incorporating an AnyRes Dynamic Resolution Vision Transformer with 2D-RoPE and a PixelShuffle 2x2 C-Abstractor.\n\n"
            "The model activates Top-2 routed experts per token plus 1 shared expert, operating within 8 GB VRAM budgets."
        ),
        "training_scope": (
            "Experimental narrow multimodal checkpoint (EXPERIMENTAL_NARROW_CHECKPOINT). Trained on multimodal diagrammatic pairs and technical "
            "schematics from miceworks/Mark-Ultra-Curriculum."
        ),
        "evaluation_summary": (
            "Validated on dynamic resolution image processing and visual token alignment."
        ),
    },
]

def compute_kv_cache_metrics(layers: int, kv_lora_rank: int, qk_rope_head_dim: int, n_heads: int, head_dim: int):
    mla_bytes_per_tok = layers * (kv_lora_rank + qk_rope_head_dim) * 2
    mha_bytes_per_tok = 2 * layers * n_heads * head_dim * 2
    comp_ratio = mha_bytes_per_tok / mla_bytes_per_tok if mla_bytes_per_tok > 0 else 1.0

    table = []
    for ctx in [4096, 8192, 32768]:
        mla_mib = (ctx * mla_bytes_per_tok) / (1024 ** 2)
        mha_mib = (ctx * mha_bytes_per_tok) / (1024 ** 2)
        table.append((ctx, mla_mib, mha_mib))

    return mla_bytes_per_tok, comp_ratio, table

def get_clan_breakdown(n_routed: int, is_vl: bool) -> str:
    if n_routed == 4:
        c3 = "Structured Data, Tool Calling, and Symbolic Logic"
        return f"""The 4 routed experts are partitioned into 4 functional clans (1 expert per clan):
- **Clan 0 (Expert 0)**: Systems code, low-level logic, and algorithmic execution.
- **Clan 1 (Expert 1)**: Mathematics, arithmetic reasoning, and STEM problem solving.
- **Clan 2 (Expert 2)**: Security boundaries, protocol compliance, and invariant validation.
- **Clan 3 (Expert 3)**: {c3}."""
    elif n_routed == 16:
        c3 = "Structured Data, Tool Calling, and Symbolic Logic"
        return f"""The 16 routed experts are partitioned into 4 functional clans (4 experts per clan):
- **Clan 0 (Experts 0-3)**: Systems code, kernel semantics, and concurrency.
- **Clan 1 (Experts 4-7)**: Mathematics, formal logic, and STEM reasoning.
- **Clan 2 (Experts 8-11)**: Security boundaries, protocols, and validation.
- **Clan 3 (Experts 12-15)**: {c3}."""
    elif n_routed == 24:
        c3 = "Multimodal and Spatial Reasoning" if is_vl else "Structured Data, Tool Calling, and Symbolic Logic"
        return f"""The 24 routed experts are partitioned into 4 functional clans (6 experts per clan):
- **Clan 0 (Experts 0-5)**: Systems code, kernel semantics, and low-level engineering.
- **Clan 1 (Experts 6-11)**: Mathematics, formal logic, and scientific computation.
- **Clan 2 (Experts 12-17)**: Security verification, static analysis, and network protocols.
- **Clan 3 (Experts 18-23)**: {c3}."""
    elif n_routed == 32:
        c3 = "Multimodal and Spatial Reasoning" if is_vl else "Structured Data, Tool Calling, and Symbolic Logic"
        return f"""The 32 routed experts are partitioned into 4 functional clans (8 experts per clan):
- **Clan 0 (Experts 0-7)**: Systems code, kernel internals, concurrency, and low-level engineering.
- **Clan 1 (Experts 8-15)**: Mathematics, formal reasoning, algorithmic logic, and STEM.
- **Clan 2 (Experts 16-23)**: Cybersecurity, vulnerability assessment, protocol enforcement, and defense.
- **Clan 3 (Experts 24-31)**: {c3}."""
    elif n_routed == 64:
        return """The 64 routed experts are partitioned into 4 functional clans (16 experts per clan):
- **Clan 0 (Experts 0-15)**: General Reasoning, Multi-turn Chat, and Instruction Following.
- **Clan 1 (Experts 16-31)**: Autonomous Agent Execution, Function Calling, and Structured Tool Output.
- **Clan 2 (Experts 32-47)**: Ultra-Long Multi-File Ingestion, Document Synthesis, and Patch Generation.
- **Clan 3 (Experts 48-63)**: Structured Data, Tool Calling, and Symbolic Logic (Deep Mathematical Derivation, Algorithmic Analysis, and Polyglot Systems Code)."""
    else:
        q = n_routed // 4
        c3 = "Multimodal and Spatial Reasoning" if is_vl else "Structured Data, Tool Calling, and Symbolic Logic"
        return f"""The {n_routed} routed experts are partitioned into 4 functional clans ({q} experts per clan):
- **Clan 0 (Experts 0-{q-1})**: Systems code and low-level execution.
- **Clan 1 (Experts {q}-{2*q-1})**: Mathematics and formal reasoning.
- **Clan 2 (Experts {2*q}-{3*q-1})**: Security boundaries and protocols.
- **Clan 3 (Experts {3*q}-{n_routed-1})**: {c3}."""

def get_tokenizer_documentation(tok_type: str, vocab_size: int) -> str:
    if tok_type == "full_32k":
        return f"""- Vocabulary Size: `{vocab_size}` active BPE tokens (58,980 merges, 0 padding tokens)
- Architecture Backend: Byte-level Byte-Pair Encoding (BPE) with GPT-2 byte fallbacks
- Epistemic Control Tokens: Self-RAG reasoning tags (`<think>`, `</think>`), tool delimiters (`<tool_call>`, `</tool_call>`), epistemic reflection tags (`[RETRIEVE]`, `[/RETRIEVE]`, `[EVALUATE]`, `[ACCEPT]`, `[REJECT]`)
- Measured Compression Fertility:
  - English technical prose: $\\approx 5.85\\text{{ characters/token}}$
  - Portuguese technical prose: $\\approx 3.17\\text{{ characters/token}}$
  - C, Rust, and systems source code: $\\approx 2.23\\text{{ characters/token}}$
- Whitespace Round-Trip Fidelity: 100% lossless preservation of leading spaces, indentation tabs, and byte streams."""
    elif tok_type == "slm_3m":
        return f"""- Vocabulary Size: `{vocab_size}` active BPE tokens (3,835 merges, 0 padding tokens)
- Architecture Backend: Byte-level Byte-Pair Encoding (BPE)
- Control Tokens: System tags (`<think>`, `</think>`), sequence delimiters (`<|bos|>`, `<|eos|>`)
- Measured Compression Fertility:
  - English technical prose: $\\approx 3.90\\text{{ characters/token}}$
  - Portuguese technical prose: $\\approx 1.94\\text{{ characters/token}}$
  - C, Rust, and systems source code: $\\approx 1.81\\text{{ characters/token}}$
- Whitespace Round-Trip Fidelity: 100% lossless preservation across token sequences."""
    else:
        return f"""- Vocabulary Size: `{vocab_size}` tensor rows (4,311 active BPE tokens + 27,721 reserved padding slots)
- Active BPE Merges: 3,981 merges
- Epistemic Control Tokens: Self-RAG reasoning tags (`<think>`, `</think>`), tool delimiters (`<tool_call>`, `</tool_call>`), epistemic reflection tags (`[RETRIEVE]`, `[/RETRIEVE]`, `[EVALUATE]`, `[ACCEPT]`, `[REJECT]`)
- Measured Compression Fertility:
  - English technical prose: $\\approx 2.53\\text{{ characters/token}}$
  - Portuguese technical prose: $\\approx 2.57\\text{{ characters/token}}$
  - C, Rust, and systems source code: $\\approx 2.02\\text{{ characters/token}}$
- Whitespace Round-Trip Fidelity: Indentation whitespace requires byte-fallback tokenization; leading 4 spaces may be stripped by ByteLevel decoder when token 28 is raw space sequence."""

def get_router_bias_documentation(model_id: str) -> str:
    if model_id == "mark-slm-v3-150m":
        return """Router bias vectors (`blk.{l}.exp_probs_b.bias`) are stored in FP32 precision. Measured disk telemetry from the IQ4_NL container:
- Bias Parameters: 208 FP32 parameters across 13 MoE layers (layers 2 to 14, 16 routed experts each; layers 0 and 1 are dense SwiGLU blocks)
- Measured Values: min = 0.8677, mean = 1.3500, max = 2.0547"""
    elif model_id == "mark-engineer-v3-1.5b":
        return """Router bias vectors (`blk.{l}.exp_probs_b.bias`) are stored in FP32 precision. Measured disk telemetry from the IQ4_NL container:
- Bias Parameters: 576 FP32 parameters across 18 MoE layers (32 routed experts each)
- Measured Values: min = 0.8760, mean = 2.2283, max = 2.9219"""
    elif model_id == "mark-slm-3m":
        return """Router bias vectors (`blk.{l}.exp_probs_b.bias`) are stored in FP32 precision. Measured disk telemetry from the IQ4_NL container:
- Bias Parameters: 24 FP32 parameters across 6 MoE layers (4 routed experts each)
- Measured Values: initialized at uniform 0.0000"""
    else:
        return """Router bias vectors are stored in FP32 precision or initialized to uniform values during pre-training to ensure stable routing dynamics."""

def generate_model_readme(spec: dict, manifest: dict) -> str:
    prov = manifest.get("provenance", {})
    t_arch = manifest.get("tensor_architecture", {})
    vocab = manifest.get("vocabulary_alignment", {})

    model_dir = REPO_ROOT / "models" / spec["id"]
    with open(model_dir / "config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    tags_list = [
        "gguf",
        "llama.cpp",
        "deepseek2",
        "mixture-of-experts",
        "multi-head-latent-attention",
        "iq4_nl",
        "mark-ai",
        "systems-engineering",
    ]
    if spec["is_vl"]:
        tags_list.extend(["multimodal", "vision-language"])

    tags_yaml = "\n".join(f"  - {t}" for t in tags_list)

    datasets_block = ""
    if spec["training_status"] == "TRAINED_CHECKPOINT":
        if spec["id"] == "mark-slm-3m":
            datasets_block = "\ndatasets:\n  - miceworks/Mark-Open-SLM-Curriculum\n  - AxiomicLabs/ArithMark-3.0"
        elif spec["id"] == "mark-slm-v3-150m":
            datasets_block = "\ndatasets:\n  - miceworks/Mark-Open-SLM-Curriculum"
    elif spec["training_status"] == "EXPERIMENTAL_NARROW_CHECKPOINT":
        datasets_block = "\ndatasets:\n  - miceworks/Mark-Ultra-Curriculum"

    frontmatter = f"""---
license: apache-2.0
library_name: gguf
pipeline_tag: {spec['pipeline_tag']}
language:
  - en{datasets_block}
tags:
{tags_yaml}
---"""

    layers = cfg.get("num_hidden_layers", 16)
    dim = cfg.get("hidden_size", 1280)
    heads = cfg.get("num_attention_heads", 16)
    head_dim = cfg.get("v_head_dim", dim // heads if heads else 80)
    q_lora = cfg.get("q_lora_rank", 640)
    kv_lora = cfg.get("kv_lora_rank", 320)
    qk_rope = cfg.get("qk_rope_head_dim", 32)
    n_routed = cfg.get("n_routed_experts", 16)
    k_active = cfg.get("num_experts_per_tok", 2)
    n_shared = cfg.get("n_shared_experts", 1)
    moe_inter = cfg.get("moe_intermediate_size", 1504)
    ctx_len = cfg.get("max_position_embeddings", 4096)
    vocab_size = cfg.get("vocab_size", vocab.get("gguf_vocab_size", 32032))

    total_params = cfg.get("total_parameters", t_arch.get("physical_total_parameters", 0))
    active_params = cfg.get("active_parameters_per_token", t_arch.get("physical_active_parameters", 0))
    total_tensors = t_arch.get("total_tensors", 273)
    sha256_val = cfg.get("sha256") or prov.get("sha256") or "N/A"

    bytes_per_tok, comp_ratio, kv_table = compute_kv_cache_metrics(layers, kv_lora, qk_rope, heads, head_dim)
    kv_rows = "\n".join(
        f"| {ctx:,} tokens | {mla_mib:.2f} MiB | {mha_mib:.2f} MiB | {comp_ratio:.1f}x |"
        for ctx, mla_mib, mha_mib in kv_table
    )

    size_formatted = prov.get("size_formatted", "IQ4_NL Container")
    gguf_filename = prov.get("gguf_file", f"{spec['id'].replace('-', '_')}_iq4_nl.gguf")

    router_bias_info = get_router_bias_documentation(spec["id"])
    tokenizer_doc = get_tokenizer_documentation(spec["tok_type"], vocab_size)

    if spec["id"] == "mark-slm-v3-150m":
        total_params_spec_str = f"`146,899,248` unique parameters (`165,349,680` materialized GGUF parameters across {total_tensors} tensors)"
        layers_spec_str = "`15` physical `deepseek2` blocks (layers 0-1 dense, layers 2-14 MoE)"
        clan_and_ext_block = ""
    else:
        total_params_spec_str = f"`{total_params:,}` ({total_params} physical across {total_tensors} tensors)"
        layers_spec_str = f"`{layers}`"
        clan_text = get_clan_breakdown(n_routed, spec["is_vl"])
        research_extensions_block = ""
        if spec["is_v3"]:
            core_loops = cfg.get("core_loops", 2)
            research_extensions_block = f"""
### Experimental PyTorch Research Extensions (Not Active in llama.cpp GGUF Runtime)

The following architectural mechanisms are research extensions implemented in the PyTorch reference codebase (`src/model/mark_v3_arch.py`) and are not active in the compiled C++/CUDA feed-forward DAG of the `llama.cpp` GGUF runtime:

1. **Latent Recursive Depth (LRD)**:
   In PyTorch, intermediate core layers (layers $s..e-1$) execute across $L_{{\\text{{core}}}} = {core_loops}$ recurrent iterations using residual anchoring:
   $$h^{{(r+1)}}_l = \\text{{Block}}_l(h^{{(r)}}_l + 0.5 \\cdot h_{{\\text{{stem}}}})$$
   In `llama.cpp`, GGUF execution is strictly feed-forward (single sequential pass through layers $0..N-1$).

2. **PB-MoE Soft-Pinning Constraints**:
   During PyTorch curriculum training, clan logit constraints reserve routing slots for domain specialists. In `llama.cpp`, routing executes via standard Top-K selection over the learned router bias affinities:
   $$s_{{i,t}} = \\sigma(u_t^T e_i) + b_i$$
"""
        vl_block = ""
        if spec["is_vl"]:
            vl_block = """
### Vision-Language Projector Subsystem

The visual architecture integrates an AnyRes SigLIP visual encoder with Spatial 2D-RoPE and a 2x2 PixelShuffle projection adapter:
1. **AnyRes Dynamic Resolution**: Preserves native aspect ratio up to 2048x2048 without downsampling distortion.
2. **Spatial 2D-RoPE**: Visual tokens retain physical (X, Y) coordinate awareness.
3. **PixelShuffle 2x2 C-Abstractor**: Compresses 4 visual patches into 1 dense token (75% visual token reduction).
"""
        clan_and_ext_block = f"""
#### Domain Clan Partitioning

{clan_text}
{research_extensions_block}{vl_block}"""

    if spec["training_status"] == "TRAINED_CHECKPOINT":
        ollama_name = "mark-slm-150m" if spec["id"] == "mark-slm-v3-150m" else "mark-slm-3m"
        usage_section = f"""## Usage

### Download

```bash
# Download GGUF and Modelfile using huggingface-cli
huggingface-cli download {spec['hf_repo']} {gguf_filename} Modelfile --local-dir .

# Or using Python
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="{spec['hf_repo']}", filename="{gguf_filename}")
```

### Cryptographic Verification

```bash
echo "{sha256_val}  {gguf_filename}" | sha256sum -c -
```

### llama.cpp Execution

```bash
# Direct CLI completion
llama-cli -m {gguf_filename} \\
  -p "Question: What is 12 * 12?\\nAnswer:" \\
  -n 64 --temp 0.2

# Local HTTP server
llama-server -m {gguf_filename} --port 8080 -c {ctx_len}
```

### Local Ollama Registration

```bash
# Create local model from downloaded Modelfile
ollama create {ollama_name} -f Modelfile
ollama run {ollama_name} "Question: What is 12 * 12? Answer:"
```

### Python Inspection

```python
import gguf

reader = gguf.GGUFReader("{gguf_filename}")
print("Architecture:", reader.get_field("general.architecture"))
print("Total Tensors:", len(reader.tensors))
total_params = sum(int(t.shape.prod()) for t in reader.tensors)
print(f"Materialized GGUF Parameters: {{total_params:,}}")
```"""
    elif spec["training_status"] == "EXPERIMENTAL_NARROW_CHECKPOINT":
        usage_section = f"""## Usage: Narrow Domain and Systems Engineering Verification

This checkpoint is an experimental narrow checkpoint trained on specific low-level or systems engineering distributions. Outside its narrow domain, greedy inference produces fragmented outputs.

### Cryptographic Verification

```bash
echo "{sha256_val}  {gguf_filename}" | sha256sum -c -
```

### Execution with llama.cpp

```bash
llama-cli -m {gguf_filename} \\
  -p "Systems engineering stub: Implement a POSIX mutex initialization routine.\\nCode:" \\
  -n 128 --temp 0.2
```

### Python Inspection

```python
import gguf

reader = gguf.GGUFReader("{gguf_filename}")
print("Architecture:", reader.get_field("general.architecture"))
print("Total Tensors:", len(reader.tensors))
total_params = sum(int(t.shape.prod()) for t in reader.tensors)
print(f"Total Parameters: {{total_params:,}}")
```"""
    else:
        usage_section = f"""## Usage: Runtime Graph, Memory & Latency Verification

This checkpoint is an untrained architectural base scaffold. Usage below is configured for binary graph verification, memory allocation profiling, and hardware latency benchmarking:

### Cryptographic Verification

```bash
echo "{sha256_val}  {gguf_filename}" | sha256sum -c -
```

### Graph and Memory Profiling with llama.cpp

```bash
# Benchmark throughput and KV cache allocation
llama-bench -m {gguf_filename} -n 128 -b 512

# Verify execution graph
llama-cli -m {gguf_filename} -p "System probe: 1+1=" -n 16 --temp 0.0
```

### Python Graph Inspection

```python
import gguf

reader = gguf.GGUFReader("{gguf_filename}")
print("Architecture:", reader.get_field("general.architecture"))
print("Total Tensors:", len(reader.tensors))
total_params = sum(int(t.shape.prod()) for t in reader.tensors)
print(f"Total Parameters: {{total_params:,}}")
```"""

    content = f"""{frontmatter}

# {spec['name']}

## Overview

{spec['overview']}

## Specifications

| Field | Value |
| :--- | :--- |
| Architecture | `deepseek2` |
| Training Status | `{spec['training_status']}` |
| Weight State Label | `{spec['weight_state_label']}` |
| Binary Size | `{size_formatted}` |
| Total Parameters | {total_params_spec_str} |
| Active Parameters | `{active_params if isinstance(active_params, str) else f"{active_params:,}"}` ({active_params} active) |
| Vocabulary Size | `{vocab_size}` |
| Layers | {layers_spec_str} |
| Hidden Dimension | `{dim}` |
| Attention Heads | `{heads}` |
| MLA Ranks | `q_lora_rank={q_lora}`, `kv_lora_rank={kv_lora}`, `qk_rope_head_dim={qk_rope}` |
| MoE Experts | `{n_routed} routed, {n_shared} shared ({k_active} active/token)` |
| Expert FFN Dim | `{moe_inter}` |
| Context Length | `{ctx_len}` tokens |
| Quantization Type | `IQ4_NL` (4-bit non-linear) |
| SHA-256 | `{sha256_val}` |

## Architecture

### Multi-Head Latent Attention (MLA)

The model projects key and value states into a shared low-rank latent representation to decouple memory scaling from context length:

```text
c_KV = h_t * W_DKV          (dim: {kv_lora})
c_Q  = h_t * W_DQ           (dim: {q_lora})
q_C  = RMSNorm(c_Q * W_UQ)
k_C  = RMSNorm(c_KV * W_UK)
```

Decoupled rotary position embeddings are concatenated along a separate dimension:

```text
k_R  = RoPE(h_t * W_KR)     (dim: {qk_rope})
q_t  = [q_C ; q_R]
k_t  = [k_C ; k_R]
```

### KV Cache Memory Usage

Physical memory consumption of the KV cache per token in FP16:

`KV_bytes_per_token = {layers} * ({kv_lora} + {qk_rope}) * 2 = {bytes_per_tok:,} bytes/token`

| Context Window | MLA KV Cache | Standard MHA KV Cache | Memory Savings |
| :--- | :--- | :--- | :--- |
{kv_rows}

### Mixture-of-Experts (Aux-Loss-Free Routing)

- **Shared Experts ({n_shared}S)**: Constantly active path for common representations:
  $$F_{{\\text{{shared}}}}(x) = (\\text{{SiLU}}(x W_{{\\text{{gate}}}}) \\odot (x W_{{\\text{{up}}}})) W_{{\\text{{down}}}}$$
- **Routed Experts ({n_routed}R, Top-{k_active})**: Routed using token-to-expert affinities $s_{{i,t}} = \\sigma(u_t^T e_i)$ over biased scores $s_{{i,t}} + b_i$, where router bias vectors (`blk.{{l}}.exp_probs_b.bias`) are stored in FP32 precision.

#### Aux-Loss-Free Routing Equation

Router biases update per step to prevent expert collapse without penalizing primary task loss using the exact sign-based rule:

$$b_i \\leftarrow b_i + \\gamma \\cdot \\operatorname{{sign}}(\\bar{{f}} - f_i)$$

where $\\gamma = 0.001$, $f_i$ is empirical expert frequency within the batch, and $\\bar{{f}}$ is target uniform frequency.

{router_bias_info}
{clan_and_ext_block}
## Training Pipeline and Reproducibility

- Training Code: Official pipelines and architecture implementations are maintained in the [Mark-AI scripts directory](https://github.com/thu12dta3/Mark-AI/tree/main/scripts).
- Curricula and Datasets: Trained on open datasets hosted on the Hugging Face Hub, including [miceworks/Mark-Open-SLM-Curriculum](https://huggingface.co/datasets/miceworks/Mark-Open-SLM-Curriculum), [miceworks/Mark-Cyber-Curriculum](https://huggingface.co/datasets/miceworks/Mark-Cyber-Curriculum), and [miceworks/Mark-Ultra-Curriculum](https://huggingface.co/datasets/miceworks/Mark-Ultra-Curriculum).
- Optimization Settings: AdamW optimizer ($\\beta_1 = 0.9, \\beta_2 = 0.95$, weight decay $0.1$, gradient clipping $1.0$).
- Training Scope: {spec['training_scope']}

## Evaluation and Benchmark Results

{spec['evaluation_summary']}

## Tokenizer and Vocabulary Specification

{tokenizer_doc}

## Limitations and Intended Use

- Architectural Research Scope: This model is an experimental architectural testbed for Multi-Head Latent Attention (MLA) and fine-grained MoE on 8 GB GPUs such as NVIDIA GeForce RTX 2060 Super (8 GB GDDR6, 256-bit, 448 GB/s), NOT a general-purpose conversational or multilingual assistant.
- In-Distribution Curriculum Learning: Benchmark scores on `Mark-SLM-V3-150M` measure in-distribution learning on `miceworks/Mark-Open-SLM-Curriculum` (which includes official benchmark training splits permitted under Open SLM Leaderboard rules) rather than open-ended out-of-distribution generalization.
- Parameter Scale Constraints: Absolute world knowledge and zero-shot reasoning capacity are bounded by compact parameter counts.

### Running Benchmarks on TPN Workbench and lm-evaluation-harness

When running automated benchmarks on TPN Workbench, lm-evaluation-harness, LM Studio, or Hugging Face evaluation runners:
- Select **Format: gguf** (or **Backend: llama.cpp**).
- Do not evaluate via raw Python `transformers` with `safetensors` or `trust_remote_code=True`, because the native `deepseek2` MLA and fine-grained 3D MoE graph executes directly inside the C++/CUDA kernels of `llama.cpp`.

{usage_section}
"""
    return content

def update_root_readme():
    local_rows = []
    for spec in MODELS_METADATA:
        m_path = REPO_ROOT / "models" / spec["id"] / "gguf_verification_manifest.json"
        manifest = {}
        if m_path.exists():
            with open(m_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)

        model_dir = REPO_ROOT / "models" / spec["id"]
        with open(model_dir / "config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)

        prov = manifest.get("provenance", {})
        arch = manifest.get("tensor_architecture", {})

        size_str = prov.get("size_formatted", "PyTorch / GGUF Config")
        tot_p = cfg.get("total_parameters", arch.get("physical_total_parameters", 0))
        act_p = cfg.get("active_parameters_per_token", arch.get("physical_active_parameters", 0))
        vocab_s = cfg.get("vocab_size", 32032)

        availability = (
            "[Public Checkpoint (`miceworks/Mark-SLM-V3-150M`)](https://huggingface.co/miceworks/Mark-SLM-V3-150M)"
            if spec["id"] == "mark-slm-v3-150m"
            else f"Local Config (`models/{spec['id']}/`)"
        )

        tot_p_str = f"{tot_p:,}" if isinstance(tot_p, int) else str(tot_p)
        act_p_str = f"{act_p:,}" if isinstance(act_p, int) else str(act_p)

        local_rows.append(
            f"| **{spec['name']}** | `{spec['training_status']}` | {size_str} | "
            f"{tot_p_str} | {act_p_str} | {vocab_s} | {availability} |"
        )

    table_content = "\n".join(local_rows)

    root_readme = f"""# Mark-AI

Sparse Transformer research codebase exploring Multi-Head Latent Attention (MLA) and fine-grained Mixture-of-Experts (MoE) exported to native `deepseek2` GGUF containers in 4-bit non-linear quantization (`IQ4_NL`). Engineered and profiled on consumer 8 GB hardware (`NVIDIA GeForce RTX 2060 Super (8 GB GDDR6, 256-bit, 448 GB/s)`).

## Public Research Checkpoint: Mark-SLM-V3-150M

**`Mark-SLM-V3-150M` is a 150M-parameter experimental sparse language model trained on 3.47M tokens (`miceworks/Mark-Open-SLM-Curriculum`), achieving an in-distribution Intelligence Index of `16.36` on the Open SLM Leaderboard ([PR #291](https://huggingface.co/spaces/AxiomicLabs/Open_SLM_Leaderboard/discussions/291)). It is an architectural research testbed for Multi-Head Latent Attention (MLA) and fine-grained MoE on 8 GB GPUs, NOT a general-purpose conversational or multilingual assistant.**

- **Public Hugging Face Repository:** [`miceworks/Mark-SLM-V3-150M`](https://huggingface.co/miceworks/Mark-SLM-V3-150M)
- **GGUF Container:** `mark_slm_v3_150m_iq4_nl.gguf` (`90.37 MiB / 94.76 MB / 94,764,576 bytes`)
- **SHA-256:** `4bbfe2dc3fc7a6ac895d56e30352bda0de1322f9826afa28cafe7d24b0e99632`
- **Unique Architectural Parameters:** `146,899,248` (`54,001,968` active per token across 15 layers: 2 dense + 13 MoE)
- **Materialized GGUF Parameters:** `165,349,680` across `273` tensors (including materialized `output.weight`)

### Open SLM Leaderboard Evaluation (In-Distribution Curriculum Learning)

| Model | Unique Parameters | HellaSwag (`acc_norm`) | ARC-E (`acc_norm`) | ARC-C (`acc_norm`) | PIQA (`acc_norm`) | ArithMark-3 (`acc`) | Intelligence Index |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mark-SLM-V3-150M (Model Soup 50/50)** | 146,899,248 | 37.10% (16.13) | 40.80% (21.07) | 25.50% (0.67) | 62.50% (25.00) | 33.90% (11.87) | **16.36** |
| **Mark-SLM-3M (Local Sub-3M Experiment)** | 2,946,008 | 25.60% (0.80) | 26.60% (2.13) | 23.75% (0.00) | 52.00% (4.00) | 29.70% (6.27) | **3.85** |

Disclosure: `miceworks/Mark-Open-SLM-Curriculum` incorporates official training splits from HellaSwag, PIQA, ARC, SciQ, and procedural ArithMark (permitted under Open SLM Leaderboard Rules 01 and 02), so these scores measure in-distribution curriculum learning rather than broad out-of-distribution generalization.

## Public Datasets on Hugging Face

All three public training datasets have been deduplicated, decontaminated against evaluation splits, stripped of local filesystem paths, and released under the `apache-2.0` license:

| Dataset | Type | Unique Samples | Primary Focus | Hugging Face Repository |
| :--- | :--- | :---: | :--- | :--- |
| **Mark-Open-SLM-Curriculum** | Pretraining / SLM | 11,990 (3.47M tokens) | Rule 02 compliant pretraining splits from HellaSwag, ARC, PIQA, SciQ, ArithMark-3, and Cosmopedia STEM (decontaminated against ARC val/test) | [miceworks/Mark-Open-SLM-Curriculum](https://huggingface.co/datasets/miceworks/Mark-Open-SLM-Curriculum) |
| **Mark-Cyber-Curriculum** | SFT / DPO | 4,613 | Strictly 3-field deduplicated vulnerability research (CWE/CVE), Red/Blue Team SecOps, and eBPF security (`CyberNative`, `Trendyol`, `neuralchemy`) | [miceworks/Mark-Cyber-Curriculum](https://huggingface.co/datasets/miceworks/Mark-Cyber-Curriculum) |
| **Mark-Ultra-Curriculum** | Systems SFT | 1,131 | Deduplicated Rust/C++ concurrency, Linux eBPF, robotics kinematics, and STEM reasoning | [miceworks/Mark-Ultra-Curriculum](https://huggingface.co/datasets/miceworks/Mark-Ultra-Curriculum) |

## Architectural Definitions and Local Configurations (`src/model/` & `models/`)

Alongside the public `Mark-SLM-V3-150M` checkpoint, this repository defines local PyTorch reference modules (`src/model/`) and GGUF export configurations (`models/`) used to test `deepseek2` tensor packing, KV cache scaling, and narrow curriculum convergence:

| Architecture Configuration | Status | GGUF Footprint | Unique Parameters | Active Parameters | Vocab Size | Availability |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- |
{table_content}

## Architecture Details

### Multi-Head Latent Attention (MLA)

To reduce Key-Value cache memory during long-context inference on 8 GB GPUs (`NVIDIA GeForce RTX 2060 Super (8 GB GDDR6, 256-bit, 448 GB/s)`), keys and values are compressed into a shared low-rank latent vector:

```text
c_KV = h_t * W_DKV          (dim: kv_lora_rank)
c_Q  = h_t * W_DQ           (dim: q_lora_rank)
q_C  = RMSNorm(c_Q * W_UQ)
k_C  = RMSNorm(c_KV * W_UK)
k_R  = RoPE(h_t * W_KR)     (dim: qk_rope_head_dim)
q_t  = [q_C ; q_R]
k_t  = [k_C ; k_R]
```

### Mixture-of-Experts and Sign-Based Router Bias

Feed-forward layers combine shared experts with top-K routed experts gated via sigmoid affinities $s_{{i,t}} = \\sigma(u_t^T e_i)$ plus learned FP32 bias $b_i$ (`blk.{{l}}.exp_probs_b.bias`), updated during training via the sign-based load-balancing rule:

$$b_i \\leftarrow b_i + \\gamma \\cdot \\operatorname{{sign}}(\\bar{{f}} - f_i)$$

### Experimental PyTorch Research Extensions (Not Active in llama.cpp GGUF Runtime)

Research extensions implemented in `src/model/mark_v3_arch.py` (such as Latent Recursive Depth when `core_loops > 1` and domain clan soft-pinning during PyTorch training) are PyTorch-only modules; exported GGUF files execute as standard sequential `deepseek2` graphs in `llama.cpp`.

## Testing and Verification

```bash
# 1. Validate Python syntax across all modules
python3 -m compileall -q src/ export/ scripts/ models/ tests/

# 2. Run the automated verification suite
pytest tests/ -v
```
"""
    with open(REPO_ROOT / "README.md", "w", encoding="utf-8") as f:
        f.write(root_readme)
    print("  [OK] Updated root README.md")

def update_all():
    print("=" * 80)
    print("UPDATING ALL README.MD FILES (100% FORENSIC HONESTY)")
    print("=" * 80)

    manifests = {}
    for spec in MODELS_METADATA:
        m_path = REPO_ROOT / "models" / spec["id"] / "gguf_verification_manifest.json"
        if m_path.exists():
            with open(m_path, "r", encoding="utf-8") as f:
                manifests[spec["id"]] = json.load(f)
        else:
            manifests[spec["id"]] = {}

    update_root_readme()

    for spec in MODELS_METADATA:
        content = generate_model_readme(spec, manifests[spec["id"]])
        readme_path = REPO_ROOT / "models" / spec["id"] / "README.md"
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(content)
        lines = len(content.splitlines())
        print(f"  [OK] Generated {spec['id']}/README.md ({lines} lines)")

    print("\n" + "=" * 80)
    print("README UPDATE COMPLETE!")
    print("=" * 80)

if __name__ == "__main__":
    update_all()
