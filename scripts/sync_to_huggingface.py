#!/usr/bin/env python3
"""
scripts/sync_to_huggingface.py
Uploads verified Model Cards (README.md) and configurations (config.json)
to all 14 official Hugging Face repositories under miceworks/*.
Preserves all existing .gguf and model weight files on the Hub.
"""

import os
from pathlib import Path
from huggingface_hub import HfApi

REPO_ROOT = Path(__file__).resolve().parent.parent

OFFICIAL_14_REPOS = [
    {"id": "mark-engineer-302m", "repo_id": "miceworks/Mark-Engineer-302M"},
    {"id": "mark-slm-v2-302m", "repo_id": "miceworks/Mark-SLM-V2-302M"},
    {"id": "mark-engineer-1.5b", "repo_id": "miceworks/Mark-Engineer-1.5B"},
    {"id": "mark-general-1.5b", "repo_id": "miceworks/Mark-General-1.5B"},
    {"id": "mark-vl-2.2b", "repo_id": "miceworks/Mark-VL-2.2B"},
    {"id": "mark-engineer-v2-8b", "repo_id": "miceworks/Mark-Engineer-V2-8B"},
    {"id": "mark-slm-v3-150m", "repo_id": "miceworks/Mark-SLM-V3-150M"},
    {"id": "mark-v3-1.1b", "repo_id": "miceworks/Mark-V3-1.1B"},
    {"id": "mark-general-v3-1.5b", "repo_id": "miceworks/Mark-General-V3-1.5B"},
    {"id": "mark-engineer-v3-1.5b", "repo_id": "miceworks/Mark-Engineer-V3-1.5B"},
    {"id": "mark-general-v3-8b", "repo_id": "miceworks/Mark-General-V3-8B"},
    {"id": "mark-experimental-v3-8b", "repo_id": "miceworks/Mark-Experimental-V3-8B"},
    {"id": "mark-slm-3m", "repo_id": "miceworks/Mark-SLM-3M"},
    {"id": "mark-v3-32b", "repo_id": "miceworks/Mark-V3-32B"},
]

FILES_TO_SYNC = [
    "README.md",
    "config.json",
    "tokenizer_config.json",
    "gguf_verification_manifest.json",
    "Modelfile",
]


def sync_all(api: HfApi):
    print("=" * 80)
    print("SYNCHRONIZING MODEL CARDS & METADATA ACROSS ALL 14 HF REPOSITORIES")
    print("=" * 80)

    for entry in OFFICIAL_14_REPOS:
        model_id = entry["id"]
        repo_id = entry["repo_id"]
        model_dir = REPO_ROOT / "models" / model_id
        print(f"\n--- Syncing {repo_id} ({model_id}) ---")

        for fname in FILES_TO_SYNC:
            fpath = model_dir / fname
            if fpath.exists():
                print(f"  Uploading {fname} ({fpath.stat().st_size} bytes)...")
                try:
                    api.upload_file(
                        path_or_fileobj=str(fpath),
                        path_in_repo=fname,
                        repo_id=repo_id,
                        commit_message=f"fix(audit): update {fname} with verified forensic metadata",
                    )
                    print(f"  [OK] Uploaded {fname}")
                except Exception as e:
                    print(f"  [ERROR] Failed to upload {fname} to {repo_id}: {e}")
            else:
                print(f"  [SKIP] Local {fname} does not exist in {model_id}")

    print("\n" + "=" * 80)
    print("HUGGING FACE METADATA SYNCHRONIZATION COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    sync_all(api)
