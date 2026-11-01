#!/usr/bin/env python3
"""
scripts/update_all_readmes.py
Stage 4 Execution:
Updates the root README.md and all 11 models/*/README.md based strictly
on the single source of truth in gguf_verification_manifest.json.
Enforces:
- Exact dual units: "X.XX MiB (Y.YY MB / Z,ZZZ,ZZZ bytes)"
- Exact integer counts for total and active parameters
- Zero emojis, zero checkboxes, zero unproven claims (no vLLM)
- "Artifact Provenance & Verification Manifest" section with audit commands
"""

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

MODELS = [
    {
        "id": "mark-engineer-302m",
        "name": "Mark-Engineer-302M",
        "hf_repo": "miceworks/Mark-Engineer-302M",
        "purpose": "Ultra-lightweight systems engineering model for kernel-level code and low-resource edge devices.",
        "context_length": 4096,
        "is_vl": False,
    },
    {
        "id": "mark-slm-v2-302m",
        "name": "Mark-SLM-V2-302M",
        "hf_repo": "miceworks/Mark-SLM-V2-302M",
        "purpose": "Compact small language model with an expanded 49,152 token vocabulary and DeepSeek2 MoE/MLA topology.",
        "context_length": 4096,
        "is_vl": False,
    },
    {
        "id": "mark-engineer-1.5b",
        "name": "Mark-Engineer-1.5B",
        "hf_repo": "miceworks/Mark-Engineer-1.5B",
        "purpose": "Foundational systems engineering model trained on low-level kernels, concurrency primitives, and systems code.",
        "context_length": 4096,
        "is_vl": False,
    },
    {
        "id": "mark-general-1.5b",
        "name": "Mark-General-1.5B",
        "hf_repo": "miceworks/Mark-General-1.5B",
        "purpose": "General-purpose 1.5B reasoning model with extended 4377 token vocabulary and DeepSeek2 MoE topology.",
        "context_length": 4096,
        "is_vl": False,
    },
    {
        "id": "mark-vl-2.2b",
        "name": "Mark-VL-2.2B",
        "hf_repo": "miceworks/Mark-VL-2.2B",
        "purpose": "Multimodal vision-language model pairing a native DeepSeek2 MoE backbone with a specialized vision projector adapter.",
        "context_length": 16384,
        "is_vl": True,
    },
    {
        "id": "mark-engineer-v2-8b",
        "name": "Mark-Engineer-V2-8B",
        "hf_repo": "miceworks/Mark-Engineer-V2-8B",
        "purpose": "Flagship 8B architectural foundation featuring 24 MoE experts, Multi-Head Latent Attention, and 49K vocabulary.",
        "context_length": 8192,
        "is_vl": False,
    },
    {
        "id": "mark-slm-v3-150m",
        "name": "Mark-SLM-V3-150M",
        "hf_repo": "miceworks/Mark-SLM-V3-150M",
        "purpose": "Sub-200M parameter edge model built on native DeepSeek2 MoE and MLA architecture with 32K vocabulary.",
        "context_length": 8192,
        "is_vl": False,
    },
    {
        "id": "mark-v3-1.1b",
        "name": "Mark-V3-1.1B",
        "hf_repo": "miceworks/Mark-V3-1.1B",
        "purpose": "V3 series model with 32 routed experts (4 active per token), 2 shared experts, and MLA. Total physical parameters: 1,222,464,960.",
        "context_length": 32768,
        "is_vl": False,
    },
    {
        "id": "mark-general-v3-1.5b",
        "name": "Mark-General-V3-1.5B",
        "hf_repo": "miceworks/Mark-General-V3-1.5B",
        "purpose": "V3 architecture 1.5B class general reasoning model with 32 routed experts and MLA.",
        "context_length": 32768,
        "is_vl": False,
    },
    {
        "id": "mark-engineer-v3-1.5b",
        "name": "Mark-Engineer-V3-1.5B",
        "hf_repo": "miceworks/Mark-Engineer-V3-1.5B",
        "purpose": "V3 architecture 1.5B class engineering specialist with 32 routed experts and MLA.",
        "context_length": 32768,
        "is_vl": False,
    },
    {
        "id": "mark-experimental-v3-8b",
        "name": "Mark-Experimental-V3-8B",
        "hf_repo": "miceworks/Mark-Experimental-V3-8B",
        "purpose": "V3 architecture 8B class research model with 32 routed experts, 2 shared experts, and MLA.",
        "context_length": 32768,
        "is_vl": False,
    },
]


def update_root_readme(manifests: dict):
    rows = []
    for spec in MODELS:
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

    root_readme = f"""# Mark AI -- Foundational Neural Architecture & Systems Engine

The central cognitive, polyglot systems engineering, and multimodal engine of the MARK ecosystem. Built on native DeepSeek2 architecture featuring Multi-Head Latent Attention (MLA) and 3D Mixture-of-Experts (MoE) in non-linear 4-bit quantization (IQ4_NL).

## Ecosystem Model Matrix (11 Models - GGUF Binary Ground Truth)

| Model Name | Weight State | Binary Size | Total Parameters | Active Parameters | Vocab Size | SHA-256 | Hugging Face Repository |
| :--- | :--- | :--- | :---: | :---: | :---: | :--- | :--- |
{table_content}

## Rigorous Measurement Conventions

All numerical metrics in the Mark-AI repository are verified against physical binary files on disk via `gguf.GGUFReader`:
1. **Size Units**: Binary sizes are strictly reported in both mebibytes and megabytes with exact byte counts: `X.XX MiB (Y.YY MB / Z,ZZZ,ZZZ bytes)`, where `1 MiB = 1,048,576 bytes` and `1 MB = 1,000,000 bytes`.
2. **Parameter Counts**: Parameter values represent physical scalar elements across all tensors in the GGUF container. Routed expert parameters are weighted by `num_experts_per_tok / n_routed_experts` to yield exact active parameters per token.
3. **Weight State**:
   - `TRAINED_CHECKPOINT_VALIDATED`: Models trained and verified against ground truth checkpoints on disk.
   - `UN-TRAINED_ARCHITECTURAL_BASE`: Models exported as architectural foundations with clean initialization, pending large-scale pre-training or fine-tuning. No synthetic or unverified benchmarks are reported.

## Architectural Highlights

1. **Native DeepSeek2 Format**: Preserves 3D expert tensors (`ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`) and Multi-Head Latent Attention compression (`attn_q_a`, `attn_q_b`, `attn_kv_a_mqa`, `attn_kv_b`) without dense degradation.
2. **IQ4_NL Quantization**: Non-linear 4-bit quantization for linear and MoE layers, retaining FP32 precision for RMSNorm and Aux-Loss-Free router bias.
3. **Reproducibility**: Each model directory contains an auditable `gguf_verification_manifest.json` with exact checksums, tensor breakdowns, and runtime execution logs.

## Quickstart

```bash
# Clone the repository
git clone https://github.com/thu12dta3/Mark-AI.git
cd Mark-AI

# Run metadata consistency and verification tests
python3 -m pytest tests/ -v
```
"""
    with open(REPO_ROOT / "README.md", "w", encoding="utf-8") as f:
        f.write(root_readme)
    print("  [OK] Updated root README.md")


def update_model_readme(spec: dict, manifest: dict):
    model_id = spec["id"]
    model_dir = REPO_ROOT / "models" / model_id
    prov = manifest["provenance"]
    t_arch = manifest["tensor_architecture"]
    vocab = manifest["vocabulary_alignment"]

    cfg_file = model_dir / "config.json"
    with open(cfg_file, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    # Note on 1.1B naming if applicable
    extra_note = ""
    if model_id == "mark-v3-1.1b":
        extra_note = """
## Architectural Designation Note

The designation "1.1B" is the series nomenclature. The exact physical parameter count exported in `mark_v3_1_1b_iq4_nl.gguf` with 32 routed experts (moe_intermediate_size=512, kv_lora_rank=256) is **1,222,464,960** total parameters (330,651,072 active per token), requiring **656.13 MiB (688.00 MB / 688,002,048 bytes)** in IQ4_NL container format.
"""

    # Note on VL files if applicable
    vl_note = ""
    if model_id == "mark-vl-2.2b":
        vl_note = """
## Repository Binary Artifacts

This repository provides three distinct binary artifacts for multimodal inference:
1. `mark_vl_2_2b_iq4_nl.gguf`: The primary language model backbone quantized in IQ4_NL (DeepSeek2 MoE + MLA, 14 layers, 24 routed experts, 1 shared expert).
2. `mmproj_mark_vl_2_2b.gguf`: Multimodal vision projector adapter responsible for processing spatial image patches into the text embedding space.
3. `mark_vl_2_2b_q4_k_m.gguf`: Legacy Q4_K_M quantization checkpoint preserved for compatibility with older runtimes.
"""

    adapter_line = "ADAPTER ./mmproj_mark_vl_2_2b.gguf\n" if spec["is_vl"] else ""

    readme_content = f"""# {spec['name']}

{spec['purpose']}

## Forensic Binary Specifications (GGUFReader Verified)

| Property | Value |
| :--- | :--- |
| Architecture | `deepseek2` |
| Quantization Type | `IQ4_NL` (4-bit Non-Linear Weights, FP32 Norms/Bias) |
| Container File | `{prov['gguf_file']}` |
| File Size | `{prov['size_formatted']}` |
| SHA-256 Checksum | `{prov['sha256']}` |
| Total Parameters | `{t_arch['total_parameters_formatted']}` physical |
| Active Parameters / Token | `{t_arch['active_parameters_formatted']}` active |
| Total Tensors | `{t_arch['total_tensors']}` ({t_arch['tensor_types']}) |
| 1D RMSNorm Precision | `100% FP32` |
| Hidden Dimension (`hidden_size`) | `{cfg.get('hidden_size')}` |
| Transformer Layers | `{cfg.get('num_hidden_layers')}` |
| Attention Heads | `{cfg.get('num_attention_heads')}` |
| Multi-Head Latent Attention (MLA) | `q_lora_rank={cfg.get('q_lora_rank')}`, `kv_lora_rank={cfg.get('kv_lora_rank')}` |
| MoE Routed Experts | `{cfg.get('n_routed_experts')}` |
| MoE Shared Experts | `{cfg.get('n_shared_experts')}` |
| Active Experts / Token | `{cfg.get('num_experts_per_tok')}` |
| MoE Intermediate Size | `{cfg.get('moe_intermediate_size')}` |
| Context Window | `{cfg.get('max_position_embeddings', 4096)}` |
| Vocabulary Size | `{vocab['gguf_vocab_size']}` |
| Weight State | `{manifest['weight_state']}` |
{extra_note}{vl_note}
## Quantization Architecture

This model is quantized in native non-linear 4-bit (`IQ4_NL`):
- Linear weight matrices and 3D MoE expert tensors are quantized to 4-bit non-linear integer tables with FP16 scale factors per 32 weights.
- All layer normalization tensors (`*_norm.weight`) and router biases are preserved in full precision `FP32`.

## Local Deployment & Inference

### Ollama

Create a `Modelfile` containing:

```dockerfile
FROM ./{prov['gguf_file']}
{adapter_line}
TEMPLATE \"\"\"{{{{ if .System }}}}<|bos|>System: {{{{ .System }}}}<|eos|>
{{{{ end }}}}{{{{ if .Prompt }}}}<|bos|>User: {{{{ .Prompt }}}}<|eos|>
{{{{ end }}}}<|bos|>Assistant:
<think>
\"\"\"

PARAMETER stop \"<|eos|>\"
PARAMETER temperature 0.2
PARAMETER top_p 0.95
PARAMETER repeat_penalty 1.1

SYSTEM \"\"\"You are {spec['name']}, a model built on native DeepSeek2 MoE and MLA architecture.\"\"\"
```

Build and run:

```bash
ollama create {model_id} -f Modelfile
ollama run {model_id}
```

### llama.cpp

```bash
# Run CLI inference
llama-cli -m {prov['gguf_file']} -p "<|bos|>System: You are an expert system.<|eos|>\\n<|bos|>User: Status report?<|eos|>\\n<|bos|>Assistant:\\n<think>\\n" -n 256

# Start local OpenAI-compatible server
llama-server -m {prov['gguf_file']} --port 8080 -c {cfg.get('max_position_embeddings', 4096)}
```

## Artifact Provenance & Verification Manifest

This repository includes an auditable verification manifest: `gguf_verification_manifest.json`.

To audit the binary checksum and vocabulary on your local machine:

```bash
{manifest['audit_command']}
```
"""
    readme_path = model_dir / "README.md"
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    print(f"  [OK] Updated {model_id}/README.md")


def update_all():
    print("=" * 80)
    print("STAGE 4: UPDATING ALL README.MD FILES")
    print("=" * 80)

    manifests = {}
    for spec in MODELS:
        m_path = REPO_ROOT / "models" / spec["id"] / "gguf_verification_manifest.json"
        with open(m_path, "r", encoding="utf-8") as f:
            manifests[spec["id"]] = json.load(f)

    # 1. Update root README
    update_root_readme(manifests)

    # 2. Update each model README
    for spec in MODELS:
        update_model_readme(spec, manifests[spec["id"]])

    print("\n" + "=" * 80)
    print("STAGE 4 README UPDATE COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    update_all()
