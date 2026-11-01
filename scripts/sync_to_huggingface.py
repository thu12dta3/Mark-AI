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

    space_readme = """---
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

    space_index_html = """<!DOCTYPE html>
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
                <tr><th>Quantization</th><td>IQ4_NL (4-bit non-linear weights, FP32 Norms/Bias)</td></tr>
                <tr><th>Binary Size</th><td>802.94 MiB (841.94 MB / 841,941,664 bytes)</td></tr>
                <tr><th>SHA-256 Checksum</th><td><code>fa7d3061f33a20beaae5c2b85aceaf720c75b811efd8bceae0c57c57037e838e</code></td></tr>
                <tr><th>Physical Parameters</th><td>1,496,121,824 physical (364,151,264 active per token)</td></tr>
                <tr><th>Context Window</th><td>4,096 tokens</td></tr>
                <tr><th>Vocabulary Size</th><td>4,311 tokens</td></tr>
                <tr><th>Weight State</th><td>TRAINED_CHECKPOINT_VALIDATED</td></tr>
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
llama-cli -m mark_engineer_1_5b_iq4_nl.gguf -p "&lt;|bos|&gt;System: You are an expert system.&lt;|eos|&gt;\\n&lt;|bos|&gt;User: Systems architecture:&lt;|eos|&gt;\\n&lt;|bos|&gt;Assistant:\\n&lt;think&gt;\\n" -n 256

# Start OpenAI-compatible HTTP server
llama-server -m mark_engineer_1_5b_iq4_nl.gguf --port 8080 -c 4096</code></pre>
        </div>
    </div>
</body>
</html>
"""

    api.upload_file(
        path_or_fileobj=space_readme.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=space_id,
        repo_type="space",
        commit_message="Synchronize Space README with forensic GGUF metrics",
    )
    print("  [OK] Uploaded Space README.md")

    api.upload_file(
        path_or_fileobj=space_index_html.encode("utf-8"),
        path_in_repo="index.html",
        repo_id=space_id,
        repo_type="space",
        commit_message="Synchronize Space index.html with forensic GGUF metrics",
    )
    print("  [OK] Uploaded Space index.html")


if __name__ == "__main__":
    api = HfApi()
    sync_models(api)
    sync_space(api)
    print("\n" + "=" * 80)
    print("STAGE 5 SYNCHRONIZATION COMPLETE!")
    print("=" * 80)
