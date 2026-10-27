#!/usr/bin/env bash
# ==============================================================================
# MarkVL-2.2B: Automated Plug-and-Play Ollama Installer & Model Registrar
# ==============================================================================
# Strict Rule: Zero emojis in all output logs and scripts.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

echo "=============================================================================="
echo "MarkVL-2.2B: Ollama Plug-and-Play Installer"
echo "Model: MarkVL 2.2B Dense Visual Perception Engine"
echo "Quantization: Q4_K_M (Language: 2.33 GB | Multimodal Projector: 801 MB)"
echo "Target Inference VRAM: ~1.85 GB on NVIDIA GeForce RTX 2060 Super (8GB GDDR6)"
echo "=============================================================================="

# 1. Verify Ollama CLI
if ! command -v ollama &>/dev/null; then
    echo "[Error] Ollama binary not found in PATH."
    echo "Please install Ollama from https://ollama.com before continuing."
    exit 1
fi

OLLAMA_VERSION=$(ollama --version 2>&1 || true)
echo "[Status] Detected Ollama: ${OLLAMA_VERSION}"

# 2. Verify or locate GGUF checkpoints
GGUF_LANG="${SCRIPT_DIR}/mark_vl_2_2b.gguf"
GGUF_PROJ="${SCRIPT_DIR}/mmproj_mark_vl_2_2b.gguf"

if [[ ! -f "${GGUF_LANG}" ]]; then
    echo "[Info] Language GGUF not found locally at ${GGUF_LANG}."
    echo "[Info] Checking local SSD cache at /mnt/ssd_1/mark ai/checkpoints/mark_vl_2_2b/ ..."
    if [[ -f "/mnt/ssd_1/mark ai/checkpoints/mark_vl_2_2b/mark_vl_2_2b_q4_k_m.gguf" ]]; then
        ln -sf "/mnt/ssd_1/mark ai/checkpoints/mark_vl_2_2b/mark_vl_2_2b_q4_k_m.gguf" "${GGUF_LANG}"
        echo "[Status] Linked language model from SSD cache."
    else
        echo "[Download] Retrieving weights via GitHub Release..."
        python3 "${REPO_ROOT}/scripts/download_model.py" --model markvl-2.2b --dest "${SCRIPT_DIR}"
    fi
fi

if [[ ! -f "${GGUF_PROJ}" ]]; then
    if [[ -f "/mnt/ssd_1/mark ai/checkpoints/mark_vl_2_2b/mmproj_mark_vl_2_2b.gguf" ]]; then
        ln -sf "/mnt/ssd_1/mark ai/checkpoints/mark_vl_2_2b/mmproj_mark_vl_2_2b.gguf" "${GGUF_PROJ}"
        echo "[Status] Linked projector from SSD cache."
    fi
fi

# 3. Create Ollama Model via Modelfile
echo "[Status] Compiling MarkVL-2.2B container in Ollama..."
cd "${SCRIPT_DIR}"
ollama create markvl -f Modelfile
ollama cp markvl mark-vl || true
ollama cp markvl markvl:2.2b || true
ollama cp markvl markvl-2.2b:latest || true

echo "=============================================================================="
echo "[Verification] MarkVL-2.2B successfully registered in Ollama."
echo "Available aliases:"
echo "  - ollama run markvl"
echo "  - ollama run mark-vl"
echo "  - ollama run markvl:2.2b"
echo ""
echo "Interactive test command:"
echo "  ollama run markvl \"Identifique sua arquitetura e capacidades perceptivas.\""
echo "=============================================================================="
