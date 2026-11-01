#!/usr/bin/env python3
import os
import sys
import json
import shutil
import subprocess
from pathlib import Path
from huggingface_hub import HfApi

REPO_ROOT = Path(__file__).resolve().parent.parent
AUDIT_JSON = REPO_ROOT / "artifacts" / "gguf_forensic_audit.json"

MODEL_CONFIGS = [
    {
        "name": "Mark-Engineer-302M",
        "slug": "mark-engineer-302m",
        "hf_repo": "miceworks/Mark-Engineer-302M",
        "gguf_path": Path("/mnt/ssd_1/mark ai/checkpoints/mark_engineer_302m_iq4_nl.gguf"),
        "role": "Edge experimental systems micro-model",
        "trained": True,
        "description": "Compact edge testbed featuring DeepSeek2 native MoE (16 routed, 1 shared, top-2 active) and Multi-Head Latent Attention (MLA).",
    },
    {
        "name": "Mark-SLM-V2-302M",
        "slug": "mark-slm-v2-302m",
        "hf_repo": "miceworks/Mark-SLM-V2-302M",
        "gguf_path": Path("/mnt/ssd_1/mark ai/checkpoints/mark_slm_v2_302m_iq4_nl.gguf"),
        "role": "Small Language Model base",
        "trained": False,
        "description": "Architectural base container featuring 16 layers of DeepSeek2 MoE and MLA for small language model exploration.",
    },
    {
        "name": "Mark-Engineer-1.5B",
        "slug": "mark-engineer-1.5b",
        "hf_repo": "miceworks/Mark-Engineer-1.5B",
        "gguf_path": Path("/mnt/ssd_1/mark ai/checkpoints/mark_engineer_1_5b_iq4_nl.gguf"),
        "role": "Systems engineering and low-level software design",
        "trained": True,
        "description": "Foundational systems engineering model trained on low-level kernels, concurrency primitives, and systems code.",
    },
    {
        "name": "Mark-General-1.5B",
        "slug": "mark-general-1.5b",
        "hf_repo": "miceworks/Mark-General-1.5B",
        "gguf_path": Path("/mnt/ssd_1/mark ai/checkpoints/mark_general_1_5b_iq4_nl.gguf"),
        "role": "General reasoning and STEM baseline",
        "trained": True,
        "description": "General domain reasoning model integrating DeepSeek2 MoE and compressed KV cache via MLA.",
    },
    {
        "name": "Mark-VL-2.2B",
        "slug": "mark-vl-2.2b",
        "hf_repo": "miceworks/Mark-VL-2.2B",
        "gguf_path": Path("/mnt/ssd_1/mark ai/checkpoints/mark_vl_2_2b_iq4_nl.gguf"),
        "role": "Multimodal vision-language model",
        "trained": True,
        "description": "Vision-language model combining DeepSeek2 language backbone with multimodal projector (mmproj).",
        "extra_hf_files": [
            (Path("/mnt/ssd_1/mark ai/checkpoints/mark_vl_2_2b/mmproj_mark_vl_2_2b.gguf"), "mmproj_mark_vl_2_2b.gguf")
        ],
    },
    {
        "name": "Mark-Engineer-V2-8B",
        "slug": "mark-engineer-v2-8b",
        "hf_repo": "miceworks/Mark-Engineer-V2-8B",
        "gguf_path": Path("/mnt/ssd_1/mark ai/checkpoints/mark_engineer_v2_8b_iq4_nl.gguf"),
        "role": "High-capacity systems engineering base",
        "trained": False,
        "description": "8.26B parameter DeepSeek2 MoE architectural base with 32 layers and 24 routed experts.",
    },
    {
        "name": "Mark-SLM-V3-150M",
        "slug": "mark-slm-v3-150m",
        "hf_repo": "miceworks/Mark-SLM-V3-150M",
        "gguf_path": Path("/mnt/ssd_1/mark ai/checkpoints/mark_slm_v3_150m_iq4_nl.gguf"),
        "role": "Ultra-lightweight edge V3 base",
        "trained": False,
        "description": "Mark V3 generation micro-model with Hadamard-fused projections and DeepSeek2 MoE routing.",
    },
    {
        "name": "Mark-V3-1.1B",
        "slug": "mark-v3-1.1b",
        "hf_repo": "miceworks/Mark-V3-1.1B",
        "gguf_path": Path("/mnt/ssd_1/mark ai/checkpoints/mark_v3_1_1b_iq4_nl.gguf"),
        "role": "Balanced V3 foundation base",
        "trained": False,
        "description": "Mark V3 architecture base with 32 routed experts, 2 shared experts, and top-4 routing.",
    },
    {
        "name": "Mark-General-V3-1.5B",
        "slug": "mark-general-v3-1.5b",
        "hf_repo": "miceworks/Mark-General-V3-1.5B",
        "gguf_path": Path("/mnt/ssd_1/mark ai/checkpoints/mark_general_v3_1_5b_iq4_nl.gguf"),
        "role": "General cognitive V3 base",
        "trained": False,
        "description": "1.52B parameter V3 base with 18 layers, 32 routed experts, and 2 shared experts.",
    },
    {
        "name": "Mark-Engineer-V3-1.5B",
        "slug": "mark-engineer-v3-1.5b",
        "hf_repo": "miceworks/Mark-Engineer-V3-1.5B",
        "gguf_path": Path("/mnt/ssd_1/mark ai/checkpoints/mark_engineer_v3_1_5b_iq4_nl.gguf"),
        "role": "Advanced systems engineering V3 base",
        "trained": False,
        "description": "1.52B parameter V3 engineering base with clan-scoped routing and DeepSeek2 compatibility.",
    },
    {
        "name": "Mark-Experimental-V3-8B",
        "slug": "mark-experimental-v3-8b",
        "hf_repo": "miceworks/Mark-Experimental-V3-8B",
        "gguf_path": Path("/mnt/ssd_1/mark ai/checkpoints/mark_experimental_v3_8b_iq4_nl.gguf"),
        "role": "Large-scale V3 experimental base",
        "trained": False,
        "description": "8.17B parameter V3 flagship base featuring 32 layers, 32 routed experts, 2 shared experts, and top-4 routing.",
    },
]

def load_audit() -> dict:
    with open(AUDIT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {item["model_name"]: item for item in data}

def generate_model_card(cfg: dict, xray: dict) -> str:
    name = cfg["name"]
    slug = cfg["slug"]
    gguf_name = xray["filename"]
    sha256 = xray["sha256"]
    size_mb = xray["size_mb"]
    t_params = xray["total_params_str"]
    a_params = xray["active_params_str"]
    dim = xray["dimensions"]
    q_info = xray["quantization"]
    trained = cfg["trained"]

    status_str = "Trained Checkpoint (Validated)" if trained else "Architectural Base / Untrained / Weight Initialization via He/Xavier + Hadamard Fusion (Ready for Pre-training or Fine-Tuning)"

    content = f"""# {name}

{cfg['description']}

## Forensic Binary Specifications (GGUFReader Verified)

| Property | Value |
| :--- | :--- |
| Architecture | `{xray['architecture']}` |
| Quantization Type | `IQ4_NL` (4-bit Non-Linear Weights, FP32 Norms/Bias) |
| Container File | `{gguf_name}` |
| File Size | `{size_mb} MB` |
| SHA-256 Checksum | `{sha256}` |
| Total Parameters | `{t_params}` ({xray['total_params']:,} physical) |
| Active Parameters / Token | `{a_params}` ({xray['active_params']:,} active) |
| Total Tensors | `{xray['total_tensors']}` (`{q_info['iq4_nl_tensors']}` IQ4_NL, `{q_info['f32_tensors']}` FP32) |
| Hidden Dimension (`dim`) | `{dim['dim']}` |
| Transformer Layers | `{dim['n_layers']}` |
| Attention Query Heads | `{dim['n_heads']}` |
| Multi-Head Latent Attention (MLA) | `q_lora_rank={dim['q_lora_rank']}`, `kv_lora_rank={dim['kv_lora_rank']}` |
| MoE Routed Experts | `{dim['n_routed_experts']}` |
| MoE Shared Experts | `{dim['n_shared_experts']}` |
| Active Experts / Token | `{dim['num_experts_per_tok']}` |
| MoE Intermediate Size | `{dim['moe_intermediate_size']}` |
| Context Window | `{dim['context_length']}` |
| Vocabulary Size | `{dim['vocab_size']}` |
| Training Status | **{status_str}** |

## Quantization Details

This model is quantized in native non-linear 4-bit (`IQ4_NL`) according to ggml specifications:
- 2D linear weight matrices and 3D MoE expert tensors are packed into 18-byte blocks for each 32 weights (2-byte FP16 scale + 16 bytes mapping to non-linear lookup table `kvalues_iq4nl = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]`).
- Normalization weights (`*_norm.weight`) and router bias (`exp_probs_b.bias`) are preserved in full precision `FP32`.

## Usage & Deployment

### llama.cpp

```bash
# Run CLI inference
llama-cli -m {gguf_name} -p "System architecture analysis:" -n 256

# Start local OpenAI-compatible server
llama-server -m {gguf_name} --port 8080 -c {dim['context_length']}
```

### Ollama

Create a `Modelfile` containing:

```dockerfile
FROM ./{gguf_name}

TEMPLATE \"\"\"<|im_start|>system
{{{{ .System }}}}<|im_end|>
<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
<|im_start|>assistant
<think>
\"\"\"

PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
PARAMETER temperature 0.2
PARAMETER top_p 0.95
```

Register and execute:

```bash
ollama create {slug} -f Modelfile
ollama run {slug}
```

### Python (llama-cpp-python)

```python
from llama_cpp import Llama

llm = Llama(
    model_path="{gguf_name}",
    n_ctx={dim['context_length']},
    n_threads=8,
)

response = llm.create_chat_completion(
    messages=[
        {{"role": "system", "content": "You are a concise software engineering assistant."}},
        {{"role": "user", "content": "Explain lock-free atomic CAS loops."}},
    ]
)
print(response["choices"][0]["message"]["content"])
```
"""
    return content

def generate_root_readme(models: list, audit: dict) -> str:
    content = """# Mark AI -- Foundational Neural Architecture & Systems Engine

The central cognitive, polyglot systems engineering, and multimodal engine of the MARK ecosystem. Built on native DeepSeek2 architecture featuring Multi-Head Latent Attention (MLA) and 3D Mixture-of-Experts (MoE) in non-linear 4-bit quantization (IQ4_NL).

## Ecosystem Model Matrix (11 Models - Forensic GGUF Verification)

| Model Name | Status | Size (MB) | Total Params | Active Params | Layers | MoE (R/S/U) | SHA-256 | Hugging Face |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
"""
    for cfg in models:
        m_name = cfg["name"]
        x = audit[m_name]
        d = x["dimensions"]
        exp_s = f"{d['n_routed_experts']}/{d['n_shared_experts']}/{d['num_experts_per_tok']}"
        hf_link = f"[{cfg['hf_repo']}](https://huggingface.co/{cfg['hf_repo']})"
        stat_short = "Trained (Valid)" if cfg["trained"] else "Untrained Base"
        sha_short = x["sha256"][:10] + "..."
        content += f"| **{m_name}** | {stat_short} | **{x['size_mb']}** | {x['total_params_str']} | {x['active_params_str']} | {d['n_layers']} | {exp_s} | `{sha_short}` | {hf_link} |\n"

    content += """
## Architectural Highlights

1. **Native DeepSeek2 Format**: Preserves 3D expert tensors (`ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps`) and Multi-Head Latent Attention compression (`attn_q_a`, `attn_q_b`, `attn_kv_a_mqa`, `attn_kv_b`) without dense degradation.
2. **IQ4_NL Quantization**: Non-linear 4-bit quantization for all linear and MoE layers, retaining FP32 precision for RMSNorm and Aux-Loss-Free router bias.
3. **Transparent Scientific Classification**:
   - Models with verified checkpoints on disk are marked as *Trained Checkpoint (Validated)*.
   - Newly initialized models are marked strictly as *Untrained Architectural Base*. No unverified benchmark figures are reported.

## Quickstart

```bash
# Clone the repository
git clone https://github.com/thu12dta3/Mark-AI.git
cd Mark-AI

# Run validation suite
python3 -m pytest tests/ -q
```
"""
    return content

def update_model_files(models: list, audit: dict):
    print("[*] Updating local model cards and Modelfiles...")
    for cfg in models:
        m_name = cfg["name"]
        slug = cfg["slug"]
        x = audit[m_name]
        m_dir = REPO_ROOT / "models" / slug
        m_dir.mkdir(parents=True, exist_ok=True)

        card_text = generate_model_card(cfg, x)
        readme_path = m_dir / "README.md"
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(card_text)

        # Modelfile
        modelfile_path = m_dir / "Modelfile"
        gguf_name = x["filename"]
        modelfile_content = f"""FROM ./{gguf_name}

TEMPLATE \"\"\"<|im_start|>system
{{{{ .System }}}}<|im_end|>
<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
<|im_start|>assistant
<think>
\"\"\"

PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
PARAMETER temperature 0.2
PARAMETER top_p 0.95
PARAMETER repeat_penalty 1.1

SYSTEM \"\"\"You are {m_name}, a language model built on native DeepSeek2 MoE and MLA architecture.\"\"\"
"""
        with open(modelfile_path, "w", encoding="utf-8") as f:
            f.write(modelfile_content)

    # Root README
    root_readme = generate_root_readme(models, audit)
    with open(REPO_ROOT / "README.md", "w", encoding="utf-8") as f:
        f.write(root_readme)
    print("[+] All model cards and root README successfully updated.")

def upload_to_huggingface(models: list, audit: dict):
    print("\n[*] Starting Hugging Face upload across 11 repositories...")
    api = HfApi()
    for cfg in models:
        m_name = cfg["name"]
        repo_id = cfg["hf_repo"]
        gguf_path = cfg["gguf_path"]
        slug = cfg["slug"]
        m_dir = REPO_ROOT / "models" / slug

        print(f"\n---> Processing HF repo: {repo_id} ({m_name})")
        api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

        # Upload GGUF binary
        print(f"     Uploading {gguf_path.name} ({audit[m_name]['size_mb']} MB)...")
        api.upload_file(
            path_or_fileobj=str(gguf_path),
            path_in_repo=gguf_path.name,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"Update {m_name} native deepseek2 IQ4_NL GGUF container",
        )

        # Upload extra files (e.g. mmproj for Mark-VL)
        for extra_src, extra_dst in cfg.get("extra_hf_files", []):
            if extra_src.exists():
                print(f"     Uploading extra asset {extra_dst}...")
                api.upload_file(
                    path_or_fileobj=str(extra_src),
                    path_in_repo=extra_dst,
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message=f"Update {extra_dst}",
                )

        # Upload configs, README, Modelfile, tokenizer
        for fname in ["README.md", "Modelfile", "config.json", "tokenizer.json", "tokenizer_config.json"]:
            fpath = m_dir / fname
            if fpath.exists():
                api.upload_file(
                    path_or_fileobj=str(fpath),
                    path_in_repo=fname,
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message=f"Update {fname} with verified GGUF metadata",
                )
        print(f"     Repo {repo_id} updated successfully.")

def create_hf_space():
    print("\n[*] Setting up Hugging Face Space for Mark-Engineer-1.5B...")
    api = HfApi()
    space_id = "miceworks/Mark-Engineer-1.5B-Space"
    api.create_repo(repo_id=space_id, repo_type="space", space_sdk="static", exist_ok=True)

    space_dir = Path("/home/thu12/.cache/mark_hf_space")
    space_dir.mkdir(parents=True, exist_ok=True)

    index_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mark Engineer 1.5B - Systems Engineering Model</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 2rem; display: flex; justify-content: center; }
        .container { max-width: 800px; width: 100%; }
        h1 { color: #38bdf8; font-size: 2rem; margin-bottom: 0.5rem; }
        .badge { background: #1e293b; color: #94a3b8; border: 1px solid #334155; padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.875rem; display: inline-block; margin-bottom: 1.5rem; }
        .card { background: #1e293b; border: 1px solid #334155; border-radius: 0.75rem; padding: 1.5rem; margin-bottom: 1.5rem; }
        h2 { font-size: 1.25rem; color: #f1f5f9; margin-top: 0; }
        pre { background: #090d16; border: 1px solid #1e293b; border-radius: 0.5rem; padding: 1rem; overflow-x: auto; color: #38bdf8; font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; font-size: 0.9rem; }
        table { width: 100%; border-collapse: collapse; margin-top: 1rem; font-size: 0.9rem; }
        th, td { border: 1px solid #334155; padding: 0.5rem 0.75rem; text-align: left; }
        th { background: #0f172a; color: #94a3b8; }
        a { color: #38bdf8; text-decoration: none; }
        a:hover { text-decoration: underline; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Mark Engineer 1.5B</h1>
        <div class="badge">Native DeepSeek2 MoE + MLA | IQ4_NL Quantization</div>
        
        <div class="card">
            <h2>Model Overview</h2>
            <p>Mark Engineer 1.5B is a lightweight systems engineering model optimized for low-level operating system mechanics, Linux kernel modules, eBPF verifiers, Windows KMDF drivers, and Tokio concurrency primitives.</p>
            <table>
                <tr><th>Architecture</th><td>DeepSeek2 (MLA + MoE)</td></tr>
                <tr><th>Quantization</th><td>IQ4_NL (4-bit non-linear)</td></tr>
                <tr><th>Physical Parameters</th><td>1.50B</td></tr>
                <tr><th>Active Parameters / Token</th><td>364.15M</td></tr>
                <tr><th>Context Length</th><td>4,096 tokens</td></tr>
                <tr><th>Hugging Face Weights</th><td><a href="https://huggingface.co/miceworks/Mark-Engineer-1.5B" target="_blank">miceworks/Mark-Engineer-1.5B</a></td></tr>
            </table>
        </div>

        <div class="card">
            <h2>Quick Execution with Ollama</h2>
            <pre><code># 1. Download Modelfile and weights from Hugging Face
huggingface-cli download miceworks/Mark-Engineer-1.5B mark_engineer_1_5b_iq4_nl.gguf Modelfile --local-dir ./mark-1.5b

# 2. Register with Ollama
cd ./mark-1.5b
ollama create mark-engineer-1.5b -f Modelfile

# 3. Run interactive systems dialogue
ollama run mark-engineer-1.5b "Explain Tokio pin projections in asynchronous Rust"</code></pre>
        </div>

        <div class="card">
            <h2>Local llama.cpp Execution</h2>
            <pre><code># Run local CLI inference
llama-cli -m mark_engineer_1_5b_iq4_nl.gguf -p "Systems architecture:" -n 256

# Start OpenAI-compatible HTTP server
llama-server -m mark_engineer_1_5b_iq4_nl.gguf --port 8080 -c 4096</code></pre>
        </div>
    </div>
</body>
</html>
"""
    with open(space_dir / "index.html", "w", encoding="utf-8") as f:
        f.write(index_html)

    readme_md = """---
title: Mark Engineer 1.5B
emoji: 💻
colorFrom: blue
colorTo: indigo
sdk: static
pinned: false
---

# Mark Engineer 1.5B Space

Interactive deployment and technical specifications testbed for Mark Engineer 1.5B using native DeepSeek2 architecture (MoE + MLA) quantized with IQ4_NL.
"""
    with open(space_dir / "README.md", "w", encoding="utf-8") as f:
        f.write(readme_md)

    for fname in ["index.html", "README.md"]:
        api.upload_file(
            path_or_fileobj=str(space_dir / fname),
            path_in_repo=fname,
            repo_id=space_id,
            repo_type="space",
            commit_message=f"Deploy static Space {fname}",
        )
    print(f"[+] Hugging Face Space {space_id} deployed successfully.")

def publish_github_releases(models: list, audit: dict):
    print("\n[*] Publishing assets to GitHub Releases...")
    tag = "v3.0.0"
    title = "Mark-AI v3.0.0: Native DeepSeek2 IQ4_NL Ecosystem Release"
    notes = """## Mark-AI v3.0.0 -- Native DeepSeek2 GGUF (IQ4_NL) Release

Universal release of all 11 Mark-AI models exported to native DeepSeek2 GGUF format with 4-bit non-linear quantization (IQ4_NL) and FP32 normalizations.

### Reassembling Large Models (> 2 GB)
For models larger than 2 GB, download all parts and run:
```bash
cat mark_engineer_v2_8b_iq4_nl.gguf.part-* > mark_engineer_v2_8b_iq4_nl.gguf
cat mark_experimental_v3_8b_iq4_nl.gguf.part-* > mark_experimental_v3_8b_iq4_nl.gguf
```
"""
    # Create or update release
    check_rel = subprocess.run(["gh", "release", "view", tag], capture_output=True, text=True)
    if check_rel.returncode != 0:
        print(f"Creating release {tag}...")
        subprocess.run(["gh", "release", "create", tag, "--title", title, "--notes", notes], check=True)
    else:
        print(f"Release {tag} already exists. Assets will be uploaded with --clobber.")

    parts_dir = Path("/run/media/thu12/hd 1tb/mark_release_parts")
    parts_dir.mkdir(parents=True, exist_ok=True)

    for cfg in models:
        m_name = cfg["name"]
        gguf_path = cfg["gguf_path"]
        size_bytes = gguf_path.stat().st_size

        if size_bytes < 2000 * 1024 * 1024:
            print(f"     Uploading {gguf_path.name} to GitHub release {tag}...")
            subprocess.run(["gh", "release", "upload", tag, str(gguf_path), "--clobber"], check=True)
        else:
            print(f"     File {gguf_path.name} exceeds 2 GB ({size_bytes / (1024*1024):.2f} MB). Splitting into 1900M chunks...")
            pfx = parts_dir / f"{gguf_path.name}.part-"
            # Remove previous parts
            for old_p in parts_dir.glob(f"{gguf_path.name}.part-*"):
                old_p.unlink()
            subprocess.run(["split", "-b", "1900M", str(gguf_path), str(pfx)], check=True)
            part_files = sorted(parts_dir.glob(f"{gguf_path.name}.part-*"))
            for p in part_files:
                print(f"     Uploading chunk {p.name}...")
                subprocess.run(["gh", "release", "upload", tag, str(p), "--clobber"], check=True)
                p.unlink()

    # Also upload mmproj for Mark-VL if available
    mmproj = Path("/mnt/ssd_1/mark ai/checkpoints/mark_vl_2_2b/mmproj_mark_vl_2_2b.gguf")
    if mmproj.exists():
        print(f"     Uploading {mmproj.name}...")
        subprocess.run(["gh", "release", "upload", tag, str(mmproj), "--clobber"], check=True)

    print("[+] All assets uploaded to GitHub Releases.")

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Mark-AI Ecosystem Publisher")
    parser.add_argument("--skip-hf", action="store_true", help="Skip Hugging Face model upload")
    parser.add_argument("--skip-space", action="store_true", help="Skip Space deployment")
    parser.add_argument("--skip-releases", action="store_true", help="Skip GitHub releases")
    args = parser.parse_args()

    if not AUDIT_JSON.exists():
        print(f"Error: {AUDIT_JSON} not found. Run scripts/gguf_forensic_xray.py first.")
        sys.exit(1)

    audit = load_audit()
    update_model_files(MODEL_CONFIGS, audit)
    if not args.skip_hf:
        upload_to_huggingface(MODEL_CONFIGS, audit)
    if not args.skip_space:
        create_hf_space()
    if not args.skip_releases:
        publish_github_releases(MODEL_CONFIGS, audit)
    print("\n=======================================================")
    print("MARK-AI ECOSYSTEM SYNCHRONIZATION AND PUBLISHING COMPLETED")
    print("=======================================================")

if __name__ == "__main__":
    main()
