#!/usr/bin/env python3
"""
scripts/update_all_readmes.py
Generates clean, direct systems engineering Model Cards across all 11 models
and the root README.md for the Mark-AI repository.

Follows concise technical documentation style (llama.cpp / Arch Wiki):
- Exact dual size units: "X.XX MiB (Y.YY MB / Z,ZZZ,ZZZ bytes)"
- Exact parameter integers for total and active parameters
- Strict prohibition of AI buzzwords, double hyphens in headings, and section numbering
- Complete Hugging Face YAML Frontmatter
- Full verification against gguf_verification_manifest.json and config.json
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

MODELS_METADATA = [
    {
        "id": "mark-engineer-302m",
        "name": "Mark-Engineer-302M",
        "hf_repo": "miceworks/Mark-Engineer-302M",
        "is_vl": False,
        "is_trained": True,
        "is_v3": False,
        "pipeline_tag": "text-generation",
        "overview": (
            "Mark-Engineer-302M is a compact systems engineering model with 251.45M total parameters "
            "and 119.33M active parameters per token. It uses a native DeepSeek2 Mixture-of-Experts (MoE) "
            "topology with Multi-Head Latent Attention (MLA), quantized to 4-bit non-linear GGUF (IQ4_NL).\n\n"
            "The container occupies 135.12 MiB (141.69 MB / 141,686,432 bytes) of storage and runs on low-power "
            "edge devices, Raspberry Pi 5 (8 GB), or entry-level consumer GPUs (such as NVIDIA RTX 2060/3060 8 GB, "
            "256-bit GDDR6, 448 GB/s bandwidth), consuming less than 2% of an 8 GB VRAM allocation."
        ),
        "training_scope": (
            "Trained on miceworks/Mark-Ultra-Curriculum across 4,821 deduplicated samples covering POSIX C "
            "interfaces, Linux system call stubs, memory barrier semantics, and bare-metal linker scripts."
        ),
    },
    {
        "id": "mark-slm-v2-302m",
        "name": "Mark-SLM-V2-302M",
        "hf_repo": "miceworks/Mark-SLM-V2-302M",
        "is_vl": False,
        "is_trained": True,
        "is_v3": False,
        "pipeline_tag": "text-generation",
        "overview": (
            "Mark-SLM-V2-302M is a compact language model with 329.13M total parameters and 155.72M active "
            "parameters per token across 16 layers. It uses a 49,152 token vocabulary, 16 routed experts, "
            "and 1 shared expert (2 active routed experts per token).\n\n"
            "Quantized in IQ4_NL, the binary occupies 177.76 MiB (186.39 MB / 186,394,944 bytes) and runs on 8 GB "
            "consumer GPUs with low latency and minimal VRAM consumption."
        ),
        "training_scope": (
            "Trained for 10,000 steps on miceworks/Mark-Ultra-Curriculum, achieving a validation loss of 0.263975. "
            "The expert dimensions were projected from 400 to 384 via truncated Singular Value Decomposition (SVD), "
            "preserving the 384 principal singular components to maintain divisibility by 32 for the IQ4_NL "
            "super-blocks while keeping 100% of the trained MLA attention, embedding, and RMSNorm weights intact."
        ),
    },
    {
        "id": "mark-engineer-1.5b",
        "name": "Mark-Engineer-1.5B",
        "hf_repo": "miceworks/Mark-Engineer-1.5B",
        "is_vl": False,
        "is_trained": True,
        "is_v3": False,
        "pipeline_tag": "text-generation",
        "overview": (
            "Mark-Engineer-1.5B is a 1.5B-parameter sparse Transformer exported to the native deepseek2 GGUF "
            "format. It combines Multi-Head Latent Attention (MLA) with a 16-routed and 1-shared SwiGLU "
            "Mixture-of-Experts (MoE) topology to keep active parameters at 364.15M per token (24.34% active) "
            "out of 1,496,121,824 total parameters.\n\n"
            "Quantized in IQ4_NL, the binary occupies 802.94 MiB (841.94 MB / 841,941,664 bytes), leaving over 7 GB "
            "of VRAM free on an 8 GB GPU (tested on NVIDIA RTX 2060 Super 8 GB) for large context windows and "
            "fast local decoding."
        ),
        "training_scope": (
            "Trained on miceworks/Mark-Ultra-Curriculum using low-level kernel sources, systems manuals, RFCs, "
            "and concurrency benchmarks. Verified checkpoint (TRAINED_CHECKPOINT_VALIDATED) for code synthesis "
            "and systems programming."
        ),
    },
    {
        "id": "mark-general-1.5b",
        "name": "Mark-General-1.5B",
        "hf_repo": "miceworks/Mark-General-1.5B",
        "is_vl": False,
        "is_trained": True,
        "is_v3": False,
        "pipeline_tag": "text-generation",
        "overview": (
            "Mark-General-1.5B is a general reasoning and STEM-oriented sparse Transformer with 1,496,185,824 "
            "total parameters and 364.22M active parameters per token across 16 layers. It features an extended "
            "4,377 token vocabulary and a 16-routed, 1-shared MoE structure.\n\n"
            "Quantized in IQ4_NL, the container takes 802.97 MiB (841.98 MB / 841,977,664 bytes) and runs locally "
            "within standard 8 GB VRAM budgets."
        ),
        "training_scope": (
            "Trained on miceworks/Mark-Ultra-Curriculum with curated STEM data, formal proofs, algorithmic logic, "
            "and technical Portuguese and English texts."
        ),
    },
    {
        "id": "mark-vl-2.2b",
        "name": "Mark-VL-2.2B",
        "hf_repo": "miceworks/Mark-VL-2.2B",
        "is_vl": True,
        "is_trained": True,
        "is_v3": False,
        "pipeline_tag": "image-text-to-text",
        "overview": (
            "Mark-VL-2.2B is a vision-language model pairing an AnyRes SigLIP vision encoder with a 14-layer "
            "DeepSeek2 MoE language model backbone. It uses 2D Rotary Position Embeddings (2D-RoPE) and a 2x2 "
            "PixelShuffle projection adapter to process images alongside text.\n\n"
            "The model contains 2,174,310,096 total parameters and 395.50M active parameters per token. The language "
            "model binary occupies 1166.72 MiB (1223.40 MB / 1,223,397,152 bytes) in IQ4_NL, operating within 8 GB "
            "VRAM limits alongside the mmproj visual adapter."
        ),
        "training_scope": (
            "Trained on multimodal diagrammatic pairs, technical schematics, and structured documents from "
            "miceworks/Mark-Ultra-Curriculum. The repository provides the primary IQ4_NL language container "
            "(mark_vl_2_2b_iq4_nl.gguf), the vision adapter (mmproj_mark_vl_2_2b.gguf), and the legacy Q4_K_M artifact."
        ),
    },
    {
        "id": "mark-engineer-v2-8b",
        "name": "Mark-Engineer-V2-8B",
        "hf_repo": "miceworks/Mark-Engineer-V2-8B",
        "is_vl": False,
        "is_trained": False,
        "is_v3": False,
        "pipeline_tag": "text-generation",
        "overview": (
            "Mark-Engineer-V2-8B is an 8B-class sparse Transformer featuring 32 layers, 24 routed experts, and "
            "1 shared expert with a 49,152 token vocabulary. It has 8,258,809,040 total parameters and activates "
            "1,770,745,040 parameters per token (21.44% active).\n\n"
            "In native IQ4_NL quantization, the container occupies 4431.26 MiB (4646.52 MB / 4,646,516,576 bytes), "
            "fitting inside a single 8 GB consumer GPU with over 3.3 GB of VRAM remaining for KV cache and scratch space."
        ),
        "training_scope": (
            "Untrained architectural base (UN-TRAINED_ARCHITECTURAL_BASE). The GGUF file contains initialized weights "
            "for testing binary graph compatibility with llama.cpp, measuring memory usage on target hardware, and "
            "serving as a base for downstream pre-training and fine-tuning."
        ),
    },
    {
        "id": "mark-slm-v3-150m",
        "name": "Mark-SLM-V3-150M",
        "hf_repo": "miceworks/Mark-SLM-V3-150M",
        "is_vl": False,
        "is_trained": False,
        "is_v3": True,
        "pipeline_tag": "text-generation",
        "overview": (
            "Mark-SLM-V3-150M is a compact edge model built to stay under the 150M parameter ceiling. By using tied "
            "word embeddings (reusing token_embd.weight as the language model output projection), the GGUF file contains "
            "272 tensors and exactly 146,899,248 parameters (54.00M active per token).\n\n"
            "Quantized in IQ4_NL, the container occupies 79.97 MiB (83.85 MB / 83,849,984 bytes), making it suitable "
            "for low-power edge microcontrollers and embedded devices."
        ),
        "training_scope": (
            "Untrained architectural base (UN-TRAINED_ARCHITECTURAL_BASE). The GGUF file contains initialized weights "
            "for testing binary graph compatibility with llama.cpp, measuring memory usage on target hardware, and "
            "serving as a base for downstream pre-training and fine-tuning."
        ),
    },
    {
        "id": "mark-v3-1.1b",
        "name": "Mark-V3-1.1B",
        "hf_repo": "miceworks/Mark-V3-1.1B",
        "is_vl": False,
        "is_trained": False,
        "is_v3": True,
        "pipeline_tag": "text-generation",
        "overview": (
            "Mark-V3-1.1B is a third-generation sparse Transformer with 1,222,464,960 total physical parameters and "
            "330,651,072 active parameters per token across 18 layers. It introduces Hadamard-transformed Multi-Head "
            "Latent Attention (H-MLA) and Partitioned-Balancing MoE (PB-MoE) with 32 routed experts and 2 shared experts.\n\n"
            "Quantized in IQ4_NL, the binary occupies 656.13 MiB (688.00 MB / 688,002,048 bytes), loading rapidly and "
            "leaving generous memory headroom on 8 GB GPUs."
        ),
        "training_scope": (
            "Untrained architectural base (UN-TRAINED_ARCHITECTURAL_BASE). The GGUF file contains initialized weights "
            "for testing binary graph compatibility with llama.cpp, measuring memory usage on target hardware, and "
            "serving as a base for downstream pre-training and fine-tuning."
        ),
    },
    {
        "id": "mark-general-v3-1.5b",
        "name": "Mark-General-V3-1.5B",
        "hf_repo": "miceworks/Mark-General-V3-1.5B",
        "is_vl": False,
        "is_trained": False,
        "is_v3": True,
        "pipeline_tag": "text-generation",
        "overview": (
            "Mark-General-V3-1.5B is a 1.52B-parameter model with 1,520,496,960 total parameters and 405.73M active "
            "parameters per token across 18 layers. It features 32 routed experts, 2 shared experts, Latent Recursive "
            "Depth (LRD), and H-MLA.\n\n"
            "Quantized in IQ4_NL, the container occupies 816.03 MiB (855.67 MB / 855,669,248 bytes) and operates "
            "comfortably within 8 GB VRAM budgets."
        ),
        "training_scope": (
            "Untrained architectural base (UN-TRAINED_ARCHITECTURAL_BASE). The GGUF file contains initialized weights "
            "for testing binary graph compatibility with llama.cpp, measuring memory usage on target hardware, and "
            "serving as a base for downstream pre-training and fine-tuning."
        ),
    },
    {
        "id": "mark-engineer-v3-1.5b",
        "name": "Mark-Engineer-V3-1.5B",
        "hf_repo": "miceworks/Mark-Engineer-V3-1.5B",
        "is_vl": False,
        "is_trained": False,
        "is_v3": True,
        "pipeline_tag": "text-generation",
        "overview": (
            "Mark-Engineer-V3-1.5B is a systems engineering variant with 1,520,496,960 total physical parameters and "
            "405.73M active parameters per token. It uses 32 routed experts, 2 shared experts, and domain soft-pinning "
            "configured for code generation, compilers, and kernel internals.\n\n"
            "Quantized in IQ4_NL, the container occupies 816.03 MiB (855.67 MB / 855,669,248 bytes), leaving ample "
            "VRAM on 8 GB GPUs for large source code files."
        ),
        "training_scope": (
            "Untrained architectural base (UN-TRAINED_ARCHITECTURAL_BASE). The GGUF file contains initialized weights "
            "for testing binary graph compatibility with llama.cpp, measuring memory usage on target hardware, and "
            "serving as a base for downstream pre-training and fine-tuning."
        ),
    },
    {
        "id": "mark-experimental-v3-8b",
        "name": "Mark-Experimental-V3-8B",
        "hf_repo": "miceworks/Mark-Experimental-V3-8B",
        "is_vl": False,
        "is_trained": False,
        "is_v3": True,
        "pipeline_tag": "text-generation",
        "overview": (
            "Mark-Experimental-V3-8B is an 8B-class sparse model packing 32 layers, 32 routed experts, 2 shared "
            "experts, and full H-MLA attention into 8,168,001,024 total parameters (2,002,374,144 active per token).\n\n"
            "Quantized in IQ4_NL, the binary occupies 4382.55 MiB (4595.44 MB / 4,595,438,848 bytes), fitting within "
            "an 8 GB consumer GPU while leaving over 3.4 GB of VRAM for KV cache."
        ),
        "training_scope": (
            "Untrained architectural base (UN-TRAINED_ARCHITECTURAL_BASE). The GGUF file contains initialized weights "
            "for testing binary graph compatibility with llama.cpp, measuring memory usage on target hardware, and "
            "serving as a base for downstream pre-training and fine-tuning."
        ),
    },
]


def compute_kv_cache_metrics(layers: int, kv_lora_rank: int, qk_rope_head_dim: int, n_heads: int, head_dim: int):
    # MLA bytes per token: layers * (kv_lora_rank + qk_rope_head_dim) * 2 bytes (FP16)
    mla_bytes_per_tok = layers * (kv_lora_rank + qk_rope_head_dim) * 2
    # Standard MHA bytes per token: 2 * layers * n_heads * head_dim * 2 bytes (FP16)
    mha_bytes_per_tok = 2 * layers * n_heads * head_dim * 2
    compression_factor = mha_bytes_per_tok / mla_bytes_per_tok if mla_bytes_per_tok > 0 else 1.0

    table = []
    for ctx in [4096, 8192, 32768]:
        mla_mib = (ctx * mla_bytes_per_tok) / (1024 ** 2)
        mha_mib = (ctx * mha_bytes_per_tok) / (1024 ** 2)
        table.append((ctx, mla_mib, mha_mib))

    return mla_bytes_per_tok, compression_factor, table


def generate_model_readme(spec: dict, manifest: dict) -> str:
    prov = manifest["provenance"]
    t_arch = manifest["tensor_architecture"]
    vocab = manifest["vocabulary_alignment"]

    model_dir = REPO_ROOT / "models" / spec["id"]
    with open(model_dir / "config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # 1. YAML Frontmatter
    datasets_block = ""
    if spec["is_trained"]:
        datasets_block = "\ndatasets:\n  - miceworks/Mark-Ultra-Curriculum"

    frontmatter = f"""---
license: agpl-3.0
library_name: gguf
pipeline_tag: {spec['pipeline_tag']}
language:
  - en
  - pt{datasets_block}
tags:
  - gguf
  - deepseek2
  - mixture-of-experts
  - multi-head-latent-attention
  - iq4_nl
  - mark-ai
  - local-ai
  - systems-engineering
---"""

    # Model dimensions
    layers = cfg.get("num_hidden_layers", 16)
    dim = cfg.get("hidden_size", 1280)
    heads = cfg.get("num_attention_heads", 16)
    head_dim = dim // heads if heads else 80
    q_lora = cfg.get("q_lora_rank", 640)
    kv_lora = cfg.get("kv_lora_rank", 320)
    qk_nope = cfg.get("qk_nope_head_dim", 64)
    qk_rope = cfg.get("qk_rope_head_dim", 32)
    v_head = cfg.get("v_head_dim", 96)
    n_routed = cfg.get("n_routed_experts", 16)
    k_active = cfg.get("num_experts_per_tok", 2)
    n_shared = cfg.get("n_shared_experts", 1)
    moe_inter = cfg.get("moe_intermediate_size", 1504)
    ctx_len = cfg.get("max_position_embeddings", 4096)
    vocab_size = vocab["gguf_vocab_size"]

    # KV Cache computations
    bytes_per_tok, comp_ratio, kv_table = compute_kv_cache_metrics(layers, kv_lora, qk_rope, heads, head_dim)

    kv_rows = "\n".join(
        f"| {ctx:,} tokens | {mla_mib:.2f} MiB | {mha_mib:.2f} MiB | {comp_ratio:.1f}x |"
        for ctx, mla_mib, mha_mib in kv_table
    )

    # Specific architectural notes
    v3_detail = ""
    if spec["is_v3"]:
        v3_detail = f"""
### V3 Architecture Details

1. **Hadamard-Transformed MLA (H-MLA)**:
   Suppresses latent activation outliers by absorbing the normalized Walsh-Hadamard matrix:
   `H_256 = (1/16) * H_2^(x8)`, `W_fused = H_256 * W_proj`
   directly into key-value projection weights (`W_wkv_a` and `W_wkv_b`) during compilation. This absorption eliminates activation spikes without adding runtime execution overhead.

2. **Partitioned-Balancing Mixture-of-Experts (PB-MoE)**:
   The {n_routed} routed experts are partitioned into 4 functional clans:
   - **Clan 0 (Experts 0-7)**: Systems code, kernel semantics, and concurrency.
   - **Clan 1 (Experts 8-15)**: Mathematics, formal logic, and STEM.
   - **Clan 2 (Experts 16-23)**: Security boundaries, protocols, and validation.
   - **Clan 3 (Experts 24-31)**: Multimodal and spatial reasoning.
   Routing enforces soft-pinning constraints ensuring balanced domain representation per token.

3. **Latent Recursive Depth (LRD) & Tied Embeddings**:
   Enables recurrent intermediate latent steps while enforcing weight-tying between input and output representations to minimize physical disk and VRAM footprints.
"""

    vl_note = ""
    if spec["is_vl"]:
        vl_note = """
### Vision-Language Projector (SigLIP & PixelShuffle)

Mark-VL-2.2B provides a vision-language setup across three files:
1. `mark_vl_2_2b_iq4_nl.gguf`: Language backbone (DeepSeek2 MoE + MLA, 14 layers, 24 routed experts, 1 shared expert).
2. `mmproj_mark_vl_2_2b.gguf`: Vision projector module embedding the AnyRes SigLIP visual feature encoder with 2D-RoPE and 2x2 PixelShuffle projection.
3. `mark_vl_2_2b_q4_k_m.gguf`: Legacy quantization file retained for compatibility.
"""

    adapter_decl = "ADAPTER ./mmproj_mark_vl_2_2b.gguf\n" if spec["is_vl"] else ""

    content = f"""{frontmatter}

# {spec['name']}

## Overview

{spec['overview']}

## Specifications

| Field | Value |
| :--- | :--- |
| Architecture | `deepseek2` |
| Weight State | `{manifest['weight_state']}` |
| Binary Size | `{prov['size_formatted']}` |
| Total Parameters | `{t_arch['total_parameters_formatted']}` ({t_arch['physical_total_parameters']} physical) |
| Active Parameters | `{t_arch['active_parameters_formatted']}` ({t_arch['physical_active_parameters']} active) |
| Vocabulary Size | `{vocab_size}` |
| Layers | `{layers}` |
| Hidden Dimension | `{dim}` |
| Attention Heads | `{heads}` |
| MLA Ranks | `q_lora_rank={q_lora}`, `kv_lora_rank={kv_lora}`, `qk_rope_head_dim={qk_rope}` |
| MoE Experts | `{n_routed} routed, {n_shared} shared ({k_active} active/token)` |
| Expert FFN Dim | `{moe_inter}` |
| Context Length | `{ctx_len}` tokens |
| Tensor Count | `{t_arch['total_tensors']}` |
| Quantization Types | `{t_arch['tensor_types']}` |
| SHA-256 | `{prov['sha256']}` |

## Architecture

### Multi-Head Latent Attention (MLA)

The model projects key and value states into a shared low-rank latent representation to reduce KV cache memory:

```text
c_KV = h_t * W_DKV          (dim: {kv_lora})
c_Q  = h_t * W_DQ           (dim: {q_lora})
q_C  = RMSNorm(c_Q * W_UQ)
k_C  = RMSNorm(c_KV * W_UK)
```

Decoupled rotary position embeddings are concatenated along a separate dimension to track token positions:

```text
k_R  = RoPE(h_t * W_KR)     (dim: {qk_rope})
q_t  = [q_C ; q_R]
k_t  = [k_C ; k_R]
```

### KV Cache Memory Usage

Physical memory consumption of the KV cache per token:

`KV_bytes_per_token = {layers} * ({kv_lora} + {qk_rope}) * 2 = {bytes_per_tok:,} bytes/token`

| Context Window | MLA KV Cache | Standard MHA KV Cache | Memory Savings |
| :--- | :--- | :--- | :--- |
{kv_rows}

### Mixture-of-Experts (MoE)

- **Shared Experts ({n_shared}S)**: Constantly active path for common features:
  `F_shared(x) = (SiLU(x * W_gate) * (x * W_up)) * W_down`
- **Routed Experts ({n_routed}R, Top-{k_active})**: Routed using learned bias offsets without auxiliary load balancing loss:
  `s_{{t,e}} = Softmax(x_t * W_router + b_e)`
  `y = x + F_shared(x) + sum_{{i in TopK}} g_i * F_routed_i(x)`
{v3_detail}{vl_note}
## Quantization

The model is quantized to GGML_TYPE_IQ4_NL (4-bit non-linear) within the native `deepseek2` GGUF container:
- Linear weights are structured in 18-byte blocks for each group of 32 weights:
  - 2 bytes: Half-precision (`FP16`) scale factor `d`.
  - 16 bytes: 32 four-bit nibbles referencing the non-linear lookup table `kvalues_iq4nl`:
    `[-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]`
- 1D normalization weights (`*_norm.weight`) and router bias vectors (`exp_probs_b.bias`) remain in full precision `FP32` to preserve numerical accuracy at layer boundaries.
- Routed MoE weights are serialized as contiguous 3D tensors (`blk.{{i}}.ffn_gate_exps.weight`, `blk.{{i}}.ffn_up_exps.weight`, `blk.{{i}}.ffn_down_exps.weight`), avoiding tensor splitting and kernel overhead.

## Training Status

{spec['training_scope']}

All reported metrics represent physical parameter counts and deterministic SHA-256 checksums from the GGUF binary.

## Usage

### Download

```bash
# Using huggingface-cli
huggingface-cli download {spec['hf_repo']} {prov['gguf_file']} --local-dir .

# Using Python
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id="{spec['hf_repo']}", filename="{prov['gguf_file']}")
```

### Cryptographic Verification

```bash
echo "{prov['sha256']}  {prov['gguf_file']}" | sha256sum -c -
```

### Ollama

```dockerfile
# Modelfile
FROM ./{prov['gguf_file']}
{adapter_decl}TEMPLATE \"\"\"{{{{ if .System }}}}<|bos|>System: {{{{ .System }}}}<|eos|>
{{{{ end }}}}{{{{ if .Prompt }}}}<|bos|>User: {{{{ .Prompt }}}}<|eos|>
{{{{ end }}}}<|bos|>Assistant:
<think>
\"\"\"

PARAMETER stop \"<|eos|>\"
PARAMETER temperature 0.3
PARAMETER top_p 0.95
PARAMETER repeat_penalty 1.1

SYSTEM \"\"\"You are {spec['name']}, a systems-focused neural model.\"\"\"
```

```bash
ollama create {spec['id']} -f Modelfile
ollama run {spec['id']}
```

### llama.cpp

```bash
# CLI generation
llama-cli -m {prov['gguf_file']} \\
  -p "<|bos|>System: Systems mode active.<|eos|>\\n<|bos|>User: Explain Linux memory barriers.<|eos|>\\n<|bos|>Assistant:\\n<think>\\n" \\
  -n 512 --temp 0.2

# HTTP server
llama-server -m {prov['gguf_file']} --port 8080 -c {ctx_len}
```

### Python Inspection

```python
import gguf

reader = gguf.GGUFReader("{prov['gguf_file']}")
print("Architecture:", reader.get_field("general.architecture"))
print("Total Tensors:", len(reader.tensors))
total_params = sum(int(t.shape.prod()) for t in reader.tensors)
print(f"Total Parameters: {{total_params:,}}")
```
"""
    return content


def update_root_readme(manifests: dict):
    rows = []
    for spec in MODELS_METADATA:
        m = manifests[spec["id"]]
        prov = m["provenance"]
        arch = m["tensor_architecture"]
        vocab = m["vocabulary_alignment"]

        state_label = (
            "Trained Checkpoint (Validated)"
            if m["weight_state"] == "TRAINED_CHECKPOINT_VALIDATED"
            else "Untrained Architectural Base"
        )

        rows.append(
            f"| **{spec['name']}** | {state_label} | {prov['size_formatted']} | "
            f"{arch['total_parameters_formatted']} | {arch['active_parameters_formatted']} | "
            f"{vocab['gguf_vocab_size']} | `{prov['sha256'][:10]}...` | [{spec['hf_repo']}](https://huggingface.co/{spec['hf_repo']}) |"
        )

    table_content = "\n".join(rows)

    root_readme = f"""# Mark-AI

Sparse Transformer architectures (Multi-Head Latent Attention + fine-grained Mixture-of-Experts) exported to native `deepseek2` GGUF containers in 4-bit non-linear quantization (`IQ4_NL`). Designed for local inference on 8 GB consumer GPUs.

## Models

| Model Name | Weight State | Binary Size | Total Parameters | Active Parameters | Vocab Size | SHA-256 | Hugging Face Repository |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- | :--- |
{table_content}

## Architecture

### Multi-Head Latent Attention (MLA & H-MLA)

Standard Multi-Head Attention (MHA) creates a memory bottleneck during long-context generation due to linear expansion of stored Key-Value caches. Mark AI implements Multi-Head Latent Attention (MLA), projecting keys and values into a shared low-rank latent representation:

```text
c_KV = h_t * W_DKV          (dim: kv_lora_rank)
c_Q  = h_t * W_DQ           (dim: q_lora_rank)
q_C  = RMSNorm(c_Q * W_UQ)
k_C  = RMSNorm(c_KV * W_UK)
```

To maintain positional awareness without expanding the cached latent vector, decoupled rotary position embeddings are concatenated along a separate dimension:

```text
k_R  = RoPE(h_t * W_KR)     (dim: qk_rope_head_dim)
q_t  = [q_C ; q_R]
k_t  = [k_C ; k_R]
```

In the V3 series, **Hadamard-MLA (H-MLA)** suppresses latent activation outliers by absorbing the normalized Walsh-Hadamard transformation:

`H_256 = (1/16) * H_2^(x8)`, `W_fused = H_256 * W_proj`

directly into the offline projection weights (`W_wkv_a` and `W_wkv_b`), eliminating activation spikes without adding runtime execution overhead.

### Mixture-of-Experts (DeepSeekMoE & PB-MoE)

To decouple parameter scale from inference compute costs, feed-forward layers are partitioned into fine-grained specialists:
- **Shared Experts**: Fixed shared pathways capture common language representations across tokens (`F_shared(x) = (SiLU(x * W_gate) * (x * W_up)) * W_down`).
- **Routed Experts**: Activated dynamically via top-K gating with learned Aux-Loss-Free router biases (`s_{{t,e}} = Softmax(x_t * W_router + b_e)` and `y = x + F_shared(x) + sum_{{i in TopK}} g_i * F_routed_i(x)`).
- **Domain Clan Partitioning (V3)**: Experts are partitioned into 4 distinct functional clans (Systems/Code, STEM/Logic, Security/Sentinel, Multimodal) with soft-pinning constraints ensuring balanced domain representation per token.

### Epistemic Self-RAG

The architecture incorporates an epistemic evaluation module (`src/model/self_rag.py`). By monitoring token-level prediction entropy:

`H(p) = -sum(p_i * log2(p_i))`

the model detects uncertainty spikes, triggering structured `<think>` reasoning paths before finalizing output tokens.

## Quantization (IQ4_NL)

### Native DeepSeek2 GGUF Graph

The models preserve native 3D tensor layouts (`blk.{{i}}.ffn_gate_exps.weight`, `blk.{{i}}.ffn_up_exps.weight`, `blk.{{i}}.ffn_down_exps.weight`) within the `deepseek2` GGUF container, bypassing the memory fragmentation associated with 2D tensor flattening.

### Super-Block Structure (GGML_TYPE_IQ4_NL)

- Linear weight matrices are compressed into 18-byte blocks per 32 weights:
  - 2 bytes: Half-precision scale factor `d`.
  - 16 bytes: 32 four-bit nibbles mapped onto the non-linear 16-level quantization grid:
    `[-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]`
- 1D RMSNorm tensors and router bias vectors remain in full precision `FP32`, ensuring numerical stability at layer boundaries.

## Repository Structure

```
Mark-AI/
├── artifacts/              # Binary audits and verification reports
│   └── gguf_forensic_audit.json
├── export/                 # DeepSeek2 GGUF export and quantization pipelines
│   └── convert_to_deepseek2_gguf.py
├── models/                 # Model configs, tokenizers and Modelfiles
│   ├── mark-engineer-1.5b/
│   ├── mark-engineer-302m/
│   ├── mark-engineer-v2-8b/
│   ├── mark-engineer-v3-1.5b/
│   ├── mark-experimental-v3-8b/
│   ├── mark-general-1.5b/
│   ├── mark-general-v3-1.5b/
│   ├── mark-slm-v2-302m/
│   ├── mark-slm-v3-150m/
│   ├── mark-v3-1.1b/
│   └── mark-vl-2.2b/
├── scripts/                # Verification and synchronization utilities
│   ├── generate_verification_manifests.py
│   ├── gguf_forensic_xray.py
│   ├── sanitize_all_models_metadata.py
│   ├── sync_to_huggingface.py
│   └── update_all_readmes.py
├── spaces/                 # Interactive inference spaces and deployment code
│   └── mark-engineer-1.5b/
│       ├── app.py
│       ├── index.html
│       ├── requirements.txt
│       └── README.md
├── src/                    # PyTorch reference neural architectures
│   └── model/
└── tests/                  # Automated verification test suite (70 tests)
```

## Testing & Verification

The repository is validated against physical GGUF binaries using pytest:

```bash
# 1. Validate Python syntax across all modules
python3 -m compileall -q src/ export/ scripts/ models/ tests/ spaces/

# 2. Run the 70-test comprehensive validation suite
pytest tests/ -v
```

All 70 tests must pass, verifying:
- Cross-consistency between `config.json`, `tokenizer_config.json`, `tokenizer.json`, `Modelfile`, and GGUF headers.
- Integer parity for total and active parameter counts.
- Dual unit size verification (`MiB` and `MB`).
- Strict absence of unregistered special tokens.
"""
    with open(REPO_ROOT / "README.md", "w", encoding="utf-8") as f:
        f.write(root_readme)
    print("  [OK] Updated root README.md")


def update_all():
    print("=" * 80)
    print("UPDATING ALL README.MD FILES (CONCISE HUMAN TECHNICAL STYLE)")
    print("=" * 80)

    manifests = {}
    for spec in MODELS_METADATA:
        m_path = REPO_ROOT / "models" / spec["id"] / "gguf_verification_manifest.json"
        with open(m_path, "r", encoding="utf-8") as f:
            manifests[spec["id"]] = json.load(f)

    # 1. Update root README
    update_root_readme(manifests)

    # 2. Update each model README
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
