import json
import re
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

OFFICIAL_MODELS = [
    "mark-engineer-302m",
    "mark-slm-v2-302m",
    "mark-engineer-1.5b",
    "mark-general-1.5b",
    "mark-vl-2.2b",
    "mark-engineer-v2-8b",
    "mark-slm-v3-150m",
    "mark-v3-1.1b",
    "mark-general-v3-1.5b",
    "mark-engineer-v3-1.5b",
    "mark-experimental-v3-8b",
]


@pytest.mark.parametrize("model_id", OFFICIAL_MODELS)
def test_model_files_exist(model_id):
    model_dir = REPO_ROOT / "models" / model_id
    assert model_dir.is_dir(), f"Model directory {model_dir} does not exist"
    assert (model_dir / "config.json").is_file(), f"Missing config.json in {model_id}"
    assert (model_dir / "tokenizer_config.json").is_file(), f"Missing tokenizer_config.json in {model_id}"
    assert (model_dir / "tokenizer.json").is_file(), f"Missing tokenizer.json in {model_id}"
    assert (model_dir / "Modelfile").is_file(), f"Missing Modelfile in {model_id}"
    assert (model_dir / "README.md").is_file(), f"Missing README.md in {model_id}"
    assert (model_dir / "gguf_verification_manifest.json").is_file(), f"Missing verification manifest in {model_id}"


@pytest.mark.parametrize("model_id", OFFICIAL_MODELS)
def test_vocab_size_cross_consistency(model_id):
    model_dir = REPO_ROOT / "models" / model_id

    with open(model_dir / "config.json") as f:
        cfg = json.load(f)

    with open(model_dir / "tokenizer_config.json") as f:
        tok_cfg = json.load(f)

    with open(model_dir / "tokenizer.json") as f:
        tok_json = json.load(f)

    with open(model_dir / "gguf_verification_manifest.json") as f:
        manifest = json.load(f)

    cfg_vocab = cfg.get("vocab_size")
    tok_cfg_vocab = tok_cfg.get("vocab_size")
    tok_json_vocab = len(tok_json.get("model", {}).get("vocab", {}))
    gguf_vocab = manifest.get("vocabulary_alignment", {}).get("gguf_vocab_size")

    assert cfg_vocab is not None, f"config.json missing vocab_size in {model_id}"
    assert tok_cfg_vocab == cfg_vocab, (
        f"Vocab mismatch in {model_id}: tokenizer_config.json={tok_cfg_vocab} != config.json={cfg_vocab}"
    )
    assert tok_json_vocab == cfg_vocab, (
        f"Vocab mismatch in {model_id}: tokenizer.json vocab len={tok_json_vocab} != config.json={cfg_vocab}"
    )
    assert gguf_vocab == cfg_vocab, (
        f"Vocab mismatch in {model_id}: GGUF header vocab={gguf_vocab} != config.json={cfg_vocab}"
    )


@pytest.mark.parametrize("model_id", OFFICIAL_MODELS)
def test_parameter_counts_cross_consistency(model_id):
    model_dir = REPO_ROOT / "models" / model_id

    with open(model_dir / "config.json") as f:
        cfg = json.load(f)

    with open(model_dir / "gguf_verification_manifest.json") as f:
        manifest = json.load(f)

    with open(model_dir / "README.md") as f:
        readme_text = f.read()

    cfg_total = cfg.get("total_parameters")
    manifest_total = manifest["tensor_architecture"]["physical_total_parameters"]
    cfg_active = cfg.get("active_parameters_per_token")
    manifest_active = manifest["tensor_architecture"]["physical_active_parameters"]

    assert isinstance(cfg_total, int), f"config.json total_parameters must be integer in {model_id}"
    assert cfg_total == manifest_total, (
        f"Total params mismatch in {model_id}: config.json={cfg_total} != manifest={manifest_total}"
    )
    assert isinstance(cfg_active, int), f"config.json active_parameters_per_token must be integer in {model_id}"
    assert cfg_active == manifest_active, (
        f"Active params mismatch in {model_id}: config.json={cfg_active} != manifest={manifest_active}"
    )

    # Check total params integer formatted or exact appears in README.md
    assert f"{cfg_total:,}" in readme_text or str(cfg_total) in readme_text, (
        f"Exact total parameters count {cfg_total:,} not found in {model_id}/README.md"
    )


@pytest.mark.parametrize("model_id", OFFICIAL_MODELS)
def test_sha256_and_units_cross_consistency(model_id):
    model_dir = REPO_ROOT / "models" / model_id

    with open(model_dir / "config.json") as f:
        cfg = json.load(f)

    with open(model_dir / "gguf_verification_manifest.json") as f:
        manifest = json.load(f)

    with open(model_dir / "README.md") as f:
        readme_text = f.read()

    real_sha256 = manifest["provenance"]["sha256"]
    cfg_sha256 = cfg.get("sha256") or (cfg.get("download", {}).get("sha256") if isinstance(cfg.get("download"), dict) else None)

    assert cfg_sha256 == real_sha256, (
        f"SHA256 mismatch in {model_id}: config.json={cfg_sha256} != real GGUF SHA256={real_sha256}"
    )
    assert real_sha256 in readme_text, (
        f"SHA256 {real_sha256} not found in {model_id}/README.md"
    )

    # Check dual units (MiB and MB) present in README.md
    assert "MiB" in readme_text and "MB" in readme_text, (
        f"Dual size units (MiB and MB) missing in {model_id}/README.md"
    )


@pytest.mark.parametrize("model_id", OFFICIAL_MODELS)
def test_modelfile_token_integrity(model_id):
    model_dir = REPO_ROOT / "models" / model_id

    with open(model_dir / "Modelfile") as f:
        mf = f.read()

    # Never use unregistered tokens like <|im_start|> or <|im_end|>
    assert "<|im_start|>" not in mf, f"Unregistered <|im_start|> found in {model_id}/Modelfile"
    assert "<|im_end|>" not in mf, f"Unregistered <|im_end|> found in {model_id}/Modelfile"

    # Stop parameter must be <|eos|>
    assert 'PARAMETER stop "<|eos|>"' in mf, f"Missing PARAMETER stop \"<|eos|>\" in {model_id}/Modelfile"

    # For Mark-VL-2.2B, check ADAPTER
    if model_id == "mark-vl-2.2b":
        assert "ADAPTER ./mmproj_mark_vl_2_2b.gguf" in mf, "Missing ADAPTER in mark-vl-2.2b Modelfile"


def test_root_readme_consistency():
    root_readme = REPO_ROOT / "README.md"
    assert root_readme.is_file()
    with open(root_readme) as f:
        content = f.read()

    for m in OFFICIAL_MODELS:
        assert m.lower() in content.lower() or m.replace("-", " ").lower() in content.lower(), f"Model {m} missing from root README.md"
    assert "MiB" in content and "MB" in content
    # No emojis in technical specification tables
    assert "🚀" not in content and "🔥" not in content and "✅" not in content
