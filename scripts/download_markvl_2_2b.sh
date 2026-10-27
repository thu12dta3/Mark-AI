#!/usr/bin/env bash
# ==============================================================================
# Mark AI - 1-Click MarkVL-2.2B Weights Downloader & Reassembler
# ==============================================================================
# Strict Constraint: Zero emojis in all output and logs.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TARGET_DIR="${REPO_ROOT}/models/mark-vl-2.2b"

mkdir -p "${TARGET_DIR}"

RELEASE_BASE="https://github.com/thu12dta3/Mark-AI/releases/download/v1.0.0-markvl-2.2b"

echo "=============================================================================="
echo "Mark AI: 1-Click MarkVL-2.2B Downloader"
echo "Target Destination: ${TARGET_DIR}"
echo "=============================================================================="

# Check if local SSD cache exists
LOCAL_CACHE="/mnt/ssd_1/mark ai/checkpoints/mark_vl_2_2b"
if [[ -f "${LOCAL_CACHE}/mark_vl_2_2b_q4_k_m.gguf" && -f "${LOCAL_CACHE}/mmproj_mark_vl_2_2b.gguf" ]]; then
    echo "[Local Cache Detected] Linking weights directly from SSD..."
    ln -sf "${LOCAL_CACHE}/mark_vl_2_2b_q4_k_m.gguf" "${TARGET_DIR}/mark_vl_2_2b.gguf"
    ln -sf "${LOCAL_CACHE}/mmproj_mark_vl_2_2b.gguf" "${TARGET_DIR}/mmproj_mark_vl_2_2b.gguf"
    echo "[Success] Symlinked MarkVL-2.2B weights to ${TARGET_DIR}."
    exit 0
fi

echo "[Step 1/3] Downloading Multimodal Projector (mmproj_mark_vl_2_2b.gguf)..."
if [[ ! -f "${TARGET_DIR}/mmproj_mark_vl_2_2b.gguf" ]]; then
    curl -L --progress-bar -o "${TARGET_DIR}/mmproj_mark_vl_2_2b.gguf" "${RELEASE_BASE}/mmproj_mark_vl_2_2b.gguf"
fi

echo "[Step 2/3] Downloading Language Model Part 00..."
if [[ ! -f "${TARGET_DIR}/mark_vl_2_2b_q4_k_m.gguf.part-00" && ! -f "${TARGET_DIR}/mark_vl_2_2b.gguf" ]]; then
    curl -L --progress-bar -o "${TARGET_DIR}/mark_vl_2_2b_q4_k_m.gguf.part-00" "${RELEASE_BASE}/mark_vl_2_2b_q4_k_m.gguf.part-00"
fi

echo "[Step 3/3] Downloading Language Model Part 01..."
if [[ ! -f "${TARGET_DIR}/mark_vl_2_2b_q4_k_m.gguf.part-01" && ! -f "${TARGET_DIR}/mark_vl_2_2b.gguf" ]]; then
    curl -L --progress-bar -o "${TARGET_DIR}/mark_vl_2_2b_q4_k_m.gguf.part-01" "${RELEASE_BASE}/mark_vl_2_2b_q4_k_m.gguf.part-01"
fi

if [[ ! -f "${TARGET_DIR}/mark_vl_2_2b.gguf" ]]; then
    echo "[Reassembly] Concatenating multipart GGUF archives into mark_vl_2_2b.gguf..."
    cat "${TARGET_DIR}/mark_vl_2_2b_q4_k_m.gguf.part-00" "${TARGET_DIR}/mark_vl_2_2b_q4_k_m.gguf.part-01" > "${TARGET_DIR}/mark_vl_2_2b.gguf"
    rm -f "${TARGET_DIR}/mark_vl_2_2b_q4_k_m.gguf.part-00" "${TARGET_DIR}/mark_vl_2_2b_q4_k_m.gguf.part-01"
    echo "[Verification] GGUF container successfully assembled."
fi

echo "=============================================================================="
echo "[Complete] MarkVL-2.2B weights are ready for plug-and-play inference."
echo "To register in Ollama, run:"
echo "  bash models/mark-vl-2.2b/install_markvl_ollama.sh"
echo "=============================================================================="
