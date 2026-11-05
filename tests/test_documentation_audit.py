import json
import re
from pathlib import Path
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

OFFICIAL_14_MODELS = [
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
    "mark-general-v3-8b",
    "mark-experimental-v3-8b",
    "mark-slm-3m",
    "mark-v3-32b",
]

CATEGORY_A_MODELS = {
    "mark-slm-v3-150m",
    "mark-engineer-v3-1.5b",
    "mark-slm-3m",
    "mark-engineer-302m",
    "mark-slm-v2-302m",
    "mark-engineer-1.5b",
    "mark-general-1.5b",
    "mark-vl-2.2b",
}

CATEGORY_B_MODELS = {
    "mark-v3-1.1b",
    "mark-general-v3-1.5b",
    "mark-vl-v3-4b",
    "mark-engineer-v2-8b",
    "mark-experimental-v3-8b",
    "mark-general-v3-8b",
    "mark-v3-32b",
}

VISION_LANGUAGE_MODELS = {
    "mark-vl-2.2b",
    "mark-vl-v3-4b",
    "mark-general-2.2b",
}


@pytest.mark.parametrize("model_id", OFFICIAL_14_MODELS)
def test_library_name_gguf_and_tags(model_id):
    model_dir = REPO_ROOT / "models" / model_id
    readme = (model_dir / "README.md").read_text(encoding="utf-8")

    # Frontmatter exists
    assert readme.startswith("---"), f"{model_id}/README.md missing YAML frontmatter opening"
    parts = readme.split("---", 2)
    assert len(parts) >= 3, f"{model_id}/README.md malformed YAML frontmatter"
    fm = parts[1]

    # library_name: gguf
    assert "library_name: gguf" in fm, f"Missing library_name: gguf in {model_id} YAML frontmatter"

    # tags contain gguf and llama.cpp
    assert "gguf" in fm, f"Missing gguf tag in {model_id} YAML frontmatter"
    assert "llama.cpp" in fm, f"Missing llama.cpp tag in {model_id} YAML frontmatter"


@pytest.mark.parametrize("model_id", OFFICIAL_14_MODELS)
def test_training_status_vs_conversational(model_id):
    model_dir = REPO_ROOT / "models" / model_id
    readme = (model_dir / "README.md").read_text(encoding="utf-8")
    parts = readme.split("---", 2)
    fm = parts[1]

    # No model may contain UN-TRAINED_ARCHITECTURAL_BASE
    assert "UN-TRAINED_ARCHITECTURAL_BASE" not in readme, (
        f"{model_id}/README.md contains legacy UN-TRAINED_ARCHITECTURAL_BASE"
    )

    if model_id in CATEGORY_B_MODELS:
        # Category B models MUST NOT carry conversational tag
        assert "conversational" not in fm, f"Category B model {model_id} has conversational tag in YAML"
        # Must declare SVD_INITIALIZED_BASE or EARLY_CONVERGENCE_CHECKPOINT
        assert (
            "SVD_INITIALIZED_BASE" in readme or "EARLY_CONVERGENCE_CHECKPOINT" in readme
        ), f"Category B model {model_id} missing SVD_INITIALIZED_BASE or EARLY_CONVERGENCE_CHECKPOINT"
        # Usage must be framed as Runtime Graph, Memory & Latency Verification
        assert "Runtime Graph, Memory & Latency Verification" in readme, (
            f"Category B model {model_id} Usage section not framed as Runtime Graph, Memory & Latency Verification"
        )
    else:
        # Category A models must declare TRAINED_CHECKPOINT
        assert "TRAINED_CHECKPOINT" in readme, f"Category A model {model_id} missing TRAINED_CHECKPOINT"


@pytest.mark.parametrize("model_id", OFFICIAL_14_MODELS)
def test_expert_count_and_clan_ranges(model_id):
    model_dir = REPO_ROOT / "models" / model_id
    cfg = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    readme = (model_dir / "README.md").read_text(encoding="utf-8")

    n_routed = cfg.get("n_routed_experts")
    assert n_routed is not None

    if n_routed == 4:
        assert "Clan 0 (Expert 0)" in readme
        assert "Clan 1 (Expert 1)" in readme
        assert "Clan 2 (Expert 2)" in readme
        assert "Clan 3 (Expert 3)" in readme
    elif n_routed == 16:
        assert "Clan 0 (Experts 0-3)" in readme
        assert "Clan 1 (Experts 4-7)" in readme
        assert "Clan 2 (Experts 8-11)" in readme
        assert "Clan 3 (Experts 12-15)" in readme
    elif n_routed == 24:
        assert "Clan 0 (Experts 0-5)" in readme
        assert "Clan 1 (Experts 6-11)" in readme
        assert "Clan 2 (Experts 12-17)" in readme
        assert "Clan 3 (Experts 18-23)" in readme
    elif n_routed == 32:
        assert "Clan 0 (Experts 0-7)" in readme
        assert "Clan 1 (Experts 8-15)" in readme
        assert "Clan 2 (Experts 16-23)" in readme
        assert "Clan 3 (Experts 24-31)" in readme
    elif n_routed == 64:
        assert "Clan 0 (Experts 0-15)" in readme
        assert "Clan 1 (Experts 16-31)" in readme
        assert "Clan 2 (Experts 32-47)" in readme
        assert "Clan 3 (Experts 48-63)" in readme


@pytest.mark.parametrize("model_id", OFFICIAL_14_MODELS)
def test_text_only_models_no_multimodal_in_clan3(model_id):
    model_dir = REPO_ROOT / "models" / model_id
    readme = (model_dir / "README.md").read_text(encoding="utf-8")

    clan3_line = ""
    for line in readme.splitlines():
        if "Clan 3" in line:
            clan3_line = line
            break

    assert clan3_line, f"Clan 3 definition not found in {model_id}/README.md"

    if model_id in VISION_LANGUAGE_MODELS:
        assert "multimodal" in clan3_line.lower() or "spatial" in clan3_line.lower()
    else:
        assert "multimodal" not in clan3_line.lower(), (
            f"Text-only model {model_id} claims multimodal in Clan 3: {clan3_line}"
        )
        assert "Structured Data, Tool Calling, and Symbolic Logic" in clan3_line, (
            f"Text-only model {model_id} missing standard Clan 3 title: {clan3_line}"
        )


@pytest.mark.parametrize("model_id", OFFICIAL_14_MODELS)
def test_required_sections_exist(model_id):
    model_dir = REPO_ROOT / "models" / model_id
    readme = (model_dir / "README.md").read_text(encoding="utf-8")

    required = [
        "## Training Pipeline and Reproducibility",
        "## Evaluation and Benchmark Results",
        "## Tokenizer and Vocabulary Specification",
        "## Limitations and Intended Use",
        "### Running Benchmarks on TPN Workbench and lm-evaluation-harness",
    ]

    for req in required:
        assert req in readme, f"Missing required section '{req}' in {model_id}/README.md"


@pytest.mark.parametrize("model_id", OFFICIAL_14_MODELS)
def test_zero_emojis_checkboxes_and_heading_dashes(model_id):
    model_dir = REPO_ROOT / "models" / model_id
    readme = (model_dir / "README.md").read_text(encoding="utf-8")

    # Check for emojis
    for char in readme:
        code = ord(char)
        # Unicode emoji ranges: 0x1F300-0x1F9FF, 0x2600-0x27BF, etc.
        assert not (
            0x1F300 <= code <= 0x1FAFF
            or 0x2700 <= code <= 0x27BF
            or code in [0x26A0, 0x26A1, 0x2600, 0x2601, 0x2602, 0x2705, 0x274C, 0x2728]
        ), f"Emoji character '{char}' (code {hex(code)}) found in {model_id}/README.md"

    # Check for checkboxes
    assert "[ ]" not in readme, f"Checkbox '[ ]' found in {model_id}/README.md"
    assert "[x]" not in readme, f"Checkbox '[x]' found in {model_id}/README.md"

    # Check for '--' in headings
    for line in readme.splitlines():
        if line.startswith("#"):
            assert "--" not in line, f"Double hyphen '--' found in heading: '{line}' in {model_id}/README.md"


def test_root_readme_has_all_14_models():
    root_readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    for m in OFFICIAL_14_MODELS:
        assert m.lower() in root_readme.lower() or m.replace("-", " ").lower() in root_readme.lower(), (
            f"Model {m} missing from root README.md"
        )

    # Check for checkboxes and emojis in root README
    assert "[ ]" not in root_readme
    assert "[x]" not in root_readme
    for char in root_readme:
        code = ord(char)
        assert not (
            0x1F300 <= code <= 0x1FAFF
            or 0x2700 <= code <= 0x27BF
            or code in [0x26A0, 0x26A1, 0x2600, 0x2601, 0x2602, 0x2705, 0x274C, 0x2728]
        ), f"Emoji character '{char}' found in root README.md"

    # Check for '--' in headings in root README
    for line in root_readme.splitlines():
        if line.startswith("#"):
            assert "--" not in line, f"Double hyphen '--' found in heading: '{line}' in root README.md"
