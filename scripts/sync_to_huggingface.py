#!/usr/bin/env python3
"""
scripts/sync_to_huggingface.py
Stage 5 Execution:
Uploads verified and synchronized metadata to all 11 Hugging Face repositories
under miceworks/* and updates the Space miceworks/Mark-Engineer-1.5B-Space.
Safely removes obsolete/contradictory files on remote repos.
"""

from pathlib import Path
from huggingface_hub import HfApi

REPO_ROOT = Path(__file__).resolve().parent.parent

MODELS = [
    {
        "id": "mark-engineer-302m",
        "repo_id": "miceworks/Mark-Engineer-302M",
        "extra_files": [],
        "delete_files": [],
    },
    {
        "id": "mark-slm-v2-302m",
        "repo_id": "miceworks/Mark-SLM-V2-302M",
        "extra_files": [],
        "delete_files": [],
    },
    {
        "id": "mark-engineer-1.5b",
        "repo_id": "miceworks/Mark-Engineer-1.5B",
        "extra_files": ["mark_engineer_1_5b_metadata.json"],
        "delete_files": [],
    },
    {
        "id": "mark-general-1.5b",
        "repo_id": "miceworks/Mark-General-1.5B",
        "extra_files": [],
        "delete_files": [],
    },
    {
        "id": "mark-vl-2.2b",
        "repo_id": "miceworks/Mark-VL-2.2B",
        "extra_files": ["mark_vl_2_2b_metadata.json"],
        "delete_files": [],
    },
    {
        "id": "mark-engineer-v2-8b",
        "repo_id": "miceworks/Mark-Engineer-V2-8B",
        "extra_files": [
            "configuration_mark_engineer_v2_8b.py",
            "mark_engineer_v2_8b_metadata.json",
        ],
        "delete_files": [
            "mark_engineer_v2_8b_step_15000.pt",
            "training_metadata.json",
        ],
    },
    {
        "id": "mark-slm-v3-150m",
        "repo_id": "miceworks/Mark-SLM-V3-150M",
        "extra_files": [],
        "delete_files": [],
    },
    {
        "id": "mark-v3-1.1b",
        "repo_id": "miceworks/Mark-V3-1.1B",
        "extra_files": [],
        "delete_files": [],
    },
    {
        "id": "mark-general-v3-1.5b",
        "repo_id": "miceworks/Mark-General-V3-1.5B",
        "extra_files": [],
        "delete_files": [],
    },
    {
        "id": "mark-engineer-v3-1.5b",
        "repo_id": "miceworks/Mark-Engineer-V3-1.5B",
        "extra_files": [],
        "delete_files": [],
    },
    {
        "id": "mark-experimental-v3-8b",
        "repo_id": "miceworks/Mark-Experimental-V3-8B",
        "extra_files": [],
        "delete_files": [],
    },
]

STANDARD_FILES = [
    "README.md",
    "config.json",
    "tokenizer_config.json",
    "tokenizer.json",
    "Modelfile",
    "gguf_verification_manifest.json",
]


def sync_models(api: HfApi):
    print("=" * 80)
    print("STAGE 5: SYNCHRONIZING METADATA TO HUGGING FACE")
    print("=" * 80)

    for spec in MODELS:
        model_id = spec["id"]
        repo_id = spec["repo_id"]
        model_dir = REPO_ROOT / "models" / model_id
        print(f"\n--- Syncing {repo_id} ---")

        # 1. Delete contradictory/obsolete remote files if present
        remote_files = api.list_repo_files(repo_id=repo_id)
        for del_file in spec["delete_files"]:
            if del_file in remote_files:
                print(f"  Deleting obsolete remote file: {del_file}...")
                try:
                    api.delete_file(
                        path_in_repo=del_file,
                        repo_id=repo_id,
                        commit_message=f"Delete obsolete contradictory artifact: {del_file}",
                    )
                    print(f"  [DELETED] {del_file}")
                except Exception as e:
                    print(f"  [WARNING] Failed to delete {del_file}: {e}")

        # 2. Upload synchronized files
        files_to_upload = STANDARD_FILES + spec["extra_files"]
        for fname in files_to_upload:
            fpath = model_dir / fname
            if fpath.exists():
                print(f"  Uploading {fname} ({fpath.stat().st_size} bytes)...")
                api.upload_file(
                    path_or_fileobj=str(fpath),
                    path_in_repo=fname,
                    repo_id=repo_id,
                    commit_message=f"Synchronize {fname} with GGUF binary single source of truth",
                )
                print(f"  [OK] Uploaded {fname}")
            else:
                print(f"  [SKIP] Local file {fname} does not exist")


def sync_space(api: HfApi):
    space_id = "miceworks/Mark-Engineer-1.5B-Space"
    print(f"\n--- Syncing Space {space_id} ---")

    space_dir = REPO_ROOT / "spaces" / "mark-engineer-1.5b"

    # Remove obsolete static files from remote Space if present
    remote_space_files = api.list_repo_files(repo_id=space_id, repo_type="space")
    for obsolete_file in ["index.html", "style.css"]:
        if obsolete_file in remote_space_files:
            print(f"  Deleting obsolete static file from Space: {obsolete_file}...")
            try:
                api.delete_file(
                    path_in_repo=obsolete_file,
                    repo_id=space_id,
                    repo_type="space",
                    commit_message=f"Remove obsolete static file {obsolete_file} in favor of Gradio app",
                )
                print(f"  [DELETED] {obsolete_file}")
            except Exception as e:
                print(f"  [WARNING] Could not delete {obsolete_file}: {e}")

    # Upload app.py, requirements.txt, and README.md
    for fname in ["README.md", "requirements.txt", "app.py"]:
        fpath = space_dir / fname
        if fpath.exists():
            print(f"  Uploading {fname} to Space...")
            api.upload_file(
                path_or_fileobj=str(fpath),
                path_in_repo=fname,
                repo_id=space_id,
                repo_type="space",
                commit_message=f"Deploy interactive Gradio application ({fname})",
            )
            print(f"  [OK] Uploaded {fname}")


if __name__ == "__main__":
    api = HfApi()
    sync_models(api)
    sync_space(api)
    print("\n" + "=" * 80)
    print("STAGE 5 SYNCHRONIZATION COMPLETE!")
    print("=" * 80)
