#!/usr/bin/env python3
"""
Mark AI - Automated Hugging Face Model Hub Sync and Publisher.
Syncs all Mark AI models (Mark-Engineer-1.5B, Mark-General-1.5B, Mark-Engineer-302M, Mark-VL-2.2B)
to Hugging Face Hub with comprehensive Model Cards, YAML frontmatter, benchmark tables,
training logs, and GGUF / PyTorch weights.
"""

import os
import sys
import argparse
from pathlib import Path
import huggingface_hub
from huggingface_hub import HfApi

SSD_CHECKPOINTS = Path("/mnt/ssd_1/mark ai/checkpoints")
REPO_ROOT = Path("/home/thu12/.gemini/antigravity/scratch/Mark-AI")

MODELS_CONFIG = {
    "Mark-Engineer-1.5B": {
        "repo_name": "Mark-Engineer-1.5B",
        "pipeline_tag": "text-generation",
        "tags": ["mark-ai", "code", "systems-engineering", "rust", "cpp", "c", "linux-kernel", "ollama", "gguf"],
        "gguf": SSD_CHECKPOINTS / "mark_engineer_1_5b_iq4_nl.gguf",
        "pt": SSD_CHECKPOINTS / "mark_engineer_1_5b_iq4_nl.pt",
        "metadata": SSD_CHECKPOINTS / "mark_engineer_1_5b_metadata.json",
        "modelfile": REPO_ROOT / "models" / "mark-engineer-1.5b" / "Modelfile",
        "local_readme": REPO_ROOT / "models" / "mark-engineer-1.5b" / "README.md",
        "description": "Polyglot Systems Engineering Model with Multi-Head Latent Attention (MLA), 16+1 Mixture of Experts (MoE), and Multi-Token Prediction (MTP). Built for low-level systems programming, Linux/Windows/macOS kernels, and high-performance infrastructure.",
    },
    "Mark-General-1.5B": {
        "repo_name": "Mark-General-1.5B",
        "pipeline_tag": "text-generation",
        "tags": ["mark-ai", "stem", "self-rag", "reasoning", "generalist", "epistemic-search", "ollama", "gguf"],
        "gguf": SSD_CHECKPOINTS / "mark_general_1_5b_iq4_nl.gguf",
        "pt": SSD_CHECKPOINTS / "mark_general_1_5b_iq4_nl.pt",
        "modelfile": REPO_ROOT / "models" / "mark-general-1.5b" / "Modelfile",
        "local_readme": REPO_ROOT / "models" / "mark-general-1.5b" / "README.md",
        "description": "Generalist STEM and Scientific Reasoning Model with Self-RAG Epistemic Tokens ([UNKNOWN], [WEB_SEARCH], [FACT_CHECK]), Multi-Head Latent Attention (MLA), and 16+1 MoE architecture.",
    },
    "Mark-Engineer-302M": {
        "repo_name": "Mark-Engineer-302M",
        "pipeline_tag": "text-generation",
        "tags": ["mark-ai", "edge-kernel", "synthetic-microbenchmarks", "experimental", "non-conversational", "gguf"],
        "gguf": SSD_CHECKPOINTS / "mark_engineer_302m_iq4_nl.gguf",
        "pt": SSD_CHECKPOINTS / "mark_engineer_302m_iq4_nl.pt",
        "modelfile": REPO_ROOT / "models" / "mark-engineer-302m" / "Modelfile",
        "local_readme": REPO_ROOT / "models" / "mark-engineer-302m" / "README.md",
        "description": "Ultra-lightweight 302M edge testbed model for synthetic micro-kernel execution. Non-conversational model (does not conduct natural open dialogue, high hallucination rate in prose; strictly tuned for deterministic micro-benchmarks).",
    },
    "Mark-VL-2.2B": {
        "repo_name": "Mark-VL-2.2B",
        "pipeline_tag": "image-text-to-text",
        "tags": ["mark-ai", "multimodal", "vision-language", "ocr", "spatial-reasoning", "pixel-shuffle", "ollama", "gguf"],
        "gguf": SSD_CHECKPOINTS / "mark_vl_2_2b" / "mark_vl_2_2b_q4_k_m.gguf",
        "mmproj": SSD_CHECKPOINTS / "mark_vl_2_2b" / "mmproj_mark_vl_2_2b.gguf",
        "metadata": SSD_CHECKPOINTS / "mark_vl_2_2b" / "mark_vl_2_2b_metadata.json",
        "modelfile": REPO_ROOT / "models" / "mark-vl-2.2b" / "Modelfile",
        "local_readme": REPO_ROOT / "models" / "mark-vl-2.2b" / "README.md",
        "description": "High-Efficiency Multimodal Vision-Language Model featuring 2D Pixel-Shuffle compression, SigLIP ViT visual encoder, and 1.5B MLA+MoE language backbone. Specialized in high-density document OCR, systems diagram parsing, and spatial coordinate grounding.",
    },
    "Mark-Engineer-V2-8B": {
        "repo_name": "Mark-Engineer-V2-8B",
        "pipeline_tag": "text-generation",
        "tags": ["mark-ai", "systems-engineering", "linux-kernel", "rust", "cpp", "c", "reverse-engineering", "robotics", "electrical-engineering", "mla", "moe", "mtp", "gguf", "ollama"],
        "gguf": SSD_CHECKPOINTS / "mark_engineer_v2_8b" / "mark_engineer_v2_8b_iq4_nl.gguf",
        "pt": SSD_CHECKPOINTS / "mark_engineer_v2_8b" / "mark_engineer_v2_8b_step_15000.pt",
        "metadata": SSD_CHECKPOINTS / "mark_engineer_v2_8b" / "training_metadata.json",
        "modelfile": REPO_ROOT / "models" / "mark-engineer-v2-8b" / "Modelfile",
        "local_readme": REPO_ROOT / "models" / "mark-engineer-v2-8b" / "README.md",
        "config_json": REPO_ROOT / "models" / "mark-engineer-v2-8b" / "config.json",
        "modeling_py": REPO_ROOT / "models" / "mark-engineer-v2-8b" / "modeling_mark_engineer_v2_8b.py",
        "configuration_py": REPO_ROOT / "models" / "mark-engineer-v2-8b" / "configuration_mark_engineer_v2_8b.py",
        "tokenizer_json": REPO_ROOT / "models" / "mark-engineer-v2-8b" / "tokenizer.json",
        "tokenizer_config": REPO_ROOT / "models" / "mark-engineer-v2-8b" / "tokenizer_config.json",
        "description": "Mark-Engineer-V2-8B: High-Density Systems & Cognitive Architecture (8.01B MoE + MLA + MTP, Converged Loss: 1.9047, Trained but Untested).",
    }
}

def generate_model_card(model_key: str, config: dict, namespace: str) -> str:
    """Generate YAML frontmatter + rich Markdown model card."""
    tags_str = "\n".join(f"- {t}" for t in config["tags"])
    
    # Read local readme content if available
    local_content = ""
    if config.get("local_readme") and config["local_readme"].exists():
        with open(config["local_readme"], "r", encoding="utf-8") as f:
            local_content = f.read()
    else:
        local_content = f"# {config['repo_name']}\n\n{config['description']}\n"
    
    frontmatter = f"""---
language:
- en
- pt
license: agpl-3.0
tags:
{tags_str}
pipeline_tag: {config['pipeline_tag']}
---
"""
    return frontmatter + "\n" + local_content

def sync_model(api: HfApi, model_key: str, config: dict, namespace: str, dry_run: bool = False):
    repo_id = f"{namespace}/{config['repo_name']}"
    print(f"\n=======================================================")
    print(f"Target Repo: {repo_id}")
    print(f"Description: {config['description']}")
    print(f"=======================================================")
    
    # Verify weights on disk
    files_to_upload = []
    for file_key in ["gguf", "pt", "mmproj", "metadata", "modelfile", "config_json", "modeling_py", "configuration_py", "tokenizer_json", "tokenizer_config"]:
        if file_key in config and config[file_key].exists():
            files_to_upload.append((file_key, config[file_key]))
        elif file_key in config:
            print(f"WARNING: File {file_key} not found at {config[file_key]}")

    print(f"Found {len(files_to_upload)} asset files ready for upload:")
    for key, path in files_to_upload:
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"  - [{key.upper()}] {path.name} ({size_mb:.2f} MiB)")

    # Prepare README.md / Model Card
    card_content = generate_model_card(model_key, config, namespace)
    temp_card_path = REPO_ROOT / f".hf_card_{model_key}.md"
    with open(temp_card_path, "w", encoding="utf-8") as f:
        f.write(card_content)
    
    if dry_run:
        print(f"[DRY RUN] Would create repo: {repo_id}")
        print(f"[DRY RUN] Would upload {len(files_to_upload)} assets and README.md")
        return

    # Create Repo on Hugging Face Hub
    print(f"Ensuring repository exists on Hugging Face Hub: {repo_id}...")
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=False)

    # Upload README.md (Model Card)
    print("Uploading Model Card (README.md)...")
    api.upload_file(
        path_or_fileobj=temp_card_path,
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="model",
        commit_message="docs: publish official Mark AI model card and benchmarks"
    )

    # Upload files
    for key, path in files_to_upload:
        print(f"Uploading {path.name} ({path.stat().st_size / (1024*1024):.2f} MiB)...")
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=path.name,
            repo_id=repo_id,
            repo_type="model",
            commit_message=f"feat: upload {key.upper()} weights ({path.name})"
        )

    # Clean temporary card
    if temp_card_path.exists():
        temp_card_path.unlink()

    print(f"SUCCESS: {repo_id} published successfully.")
    print(f"URL: https://huggingface.co/{repo_id}")

def main():
    parser = argparse.ArgumentParser(description="Sync Mark AI Models to Hugging Face Hub")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face API write token")
    parser.add_argument("--org", type=str, default=None, help="Hugging Face organization name (optional)")
    parser.add_argument("--model", type=str, default="all", choices=["all", "Mark-Engineer-1.5B", "Mark-General-1.5B", "Mark-Engineer-302M", "Mark-VL-2.2B", "Mark-Engineer-V2-8B"], help="Specific model to sync")
    parser.add_argument("--dry-run", action="store_true", help="Perform dry run without actual network upload")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN") or huggingface_hub.get_token()
    
    if not token and not args.dry_run:
        print("ERROR: No Hugging Face access token found.")
        print("Please provide your Hugging Face write token using one of the following methods:")
        print("  1. Pass --token hf_yourTokenHere")
        print("  2. Set HF_TOKEN environment variable: export HF_TOKEN=hf_...")
        print("  3. Run: huggingface-cli login")
        print("\nTo preview what will be published without a token, run:")
        print("  python3 scripts/hf_sync_models.py --dry-run")
        sys.exit(1)

    api = None
    namespace = args.org
    if token:
        api = HfApi(token=token)
        user_info = api.whoami()
        username = user_info["name"]
        print(f"Authenticated as Hugging Face user: {username}")
        if not namespace:
            namespace = username
    else:
        namespace = args.org or "mark-ai"

    print(f"Publishing under Hugging Face namespace: {namespace}")
    models_to_sync = MODELS_CONFIG.keys() if args.model == "all" else [args.model]

    for model_key in models_to_sync:
        sync_model(api, model_key, MODELS_CONFIG[model_key], namespace, dry_run=args.dry_run)

if __name__ == "__main__":
    main()
