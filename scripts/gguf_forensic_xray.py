#!/usr/bin/env python3
import os
import sys
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, List

import numpy as np
import gguf

MODELS_LIST = [
    ("Mark-Engineer-302M", Path("/mnt/ssd_1/mark ai/checkpoints/mark_engineer_302m_iq4_nl.gguf"), True),
    ("Mark-SLM-V2-302M", Path("/mnt/ssd_1/mark ai/checkpoints/mark_slm_v2_302m_iq4_nl.gguf"), False),
    ("Mark-Engineer-1.5B", Path("/mnt/ssd_1/mark ai/checkpoints/mark_engineer_1_5b_iq4_nl.gguf"), True),
    ("Mark-General-1.5B", Path("/mnt/ssd_1/mark ai/checkpoints/mark_general_1_5b_iq4_nl.gguf"), True),
    ("Mark-VL-2.2B", Path("/mnt/ssd_1/mark ai/checkpoints/mark_vl_2_2b_iq4_nl.gguf"), True),
    ("Mark-Engineer-V2-8B", Path("/mnt/ssd_1/mark ai/checkpoints/mark_engineer_v2_8b_iq4_nl.gguf"), False),
    ("Mark-SLM-V3-150M", Path("/mnt/ssd_1/mark ai/checkpoints/mark_slm_v3_150m_iq4_nl.gguf"), False),
    ("Mark-V3-1.1B", Path("/mnt/ssd_1/mark ai/checkpoints/mark_v3_1_1b_iq4_nl.gguf"), False),
    ("Mark-General-V3-1.5B", Path("/mnt/ssd_1/mark ai/checkpoints/mark_general_v3_1_5b_iq4_nl.gguf"), False),
    ("Mark-Engineer-V3-1.5B", Path("/mnt/ssd_1/mark ai/checkpoints/mark_engineer_v3_1_5b_iq4_nl.gguf"), False),
    ("Mark-Experimental-V3-8B", Path("/mnt/ssd_1/mark ai/checkpoints/mark_experimental_v3_8b_iq4_nl.gguf"), False),
]

def compute_sha256(filepath: Path) -> str:
    sha = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(64 * 1024 * 1024):
            sha.update(chunk)
    return sha.hexdigest()

def get_field_val(reader: gguf.GGUFReader, key: str, default: Any = None) -> Any:
    f = reader.get_field(key)
    if f is None or not f.parts or f.data is None or len(f.data) == 0:
        return default
    idx = f.data[0]
    part = f.parts[idx]
    if f.types and f.types[0] == gguf.GGUFValueType.STRING:
        return bytes(part).decode("utf-8", errors="replace")
    if hasattr(part, "tolist"):
        lst = part.tolist()
        return lst[0] if len(lst) == 1 else lst
    if hasattr(part, "item"):
        return part.item()
    return part

def analyze_gguf(name: str, path: Path, has_pt: bool) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"GGUF file missing: {path}")

    file_size_bytes = path.stat().st_size
    file_size_mb = file_size_bytes / (1024 * 1024)
    sha256_hash = compute_sha256(path)

    reader = gguf.GGUFReader(str(path))
    arch = get_field_val(reader, "general.architecture", "unknown")
    assert arch == "deepseek2", f"Invalid architecture: {arch} (expected deepseek2)"

    prefix = f"{arch}."
    context_length = get_field_val(reader, f"{prefix}context_length", 4096)
    embedding_length = get_field_val(reader, f"{prefix}embedding_length", 1280)
    block_count = get_field_val(reader, f"{prefix}block_count", 16)
    feed_forward_length = get_field_val(reader, f"{prefix}feed_forward_length", 3840)
    head_count = get_field_val(reader, f"{prefix}attention.head_count", 16)
    q_lora_rank = get_field_val(reader, f"{prefix}attention.q_lora_rank", 0)
    kv_lora_rank = get_field_val(reader, f"{prefix}attention.kv_lora_rank", 0)
    key_length = get_field_val(reader, f"{prefix}attention.key_length", 0)
    value_length = get_field_val(reader, f"{prefix}attention.value_length", 0)
    expert_feed_forward_length = get_field_val(reader, f"{prefix}expert_feed_forward_length", 0)
    expert_count = get_field_val(reader, f"{prefix}expert_count", 0)
    expert_shared_count = get_field_val(reader, f"{prefix}expert_shared_count", 0)
    expert_used_count = get_field_val(reader, f"{prefix}expert_used_count", 0)
    rope_dimension_count = get_field_val(reader, f"{prefix}rope.dimension_count", 0)
    vocab_size = get_field_val(reader, f"{prefix}vocab_size", 32000)

    tensor_dict = {t.name: t for t in reader.tensors}
    total_tensors = len(reader.tensors)

    # Count quant types and parameter counts
    iq4_nl_count = 0
    f32_count = 0
    f16_count = 0
    other_quant_count = 0

    total_params = 0
    active_params = 0

    for t in reader.tensors:
        t_params = int(np.prod(t.shape))
        total_params += t_params

        # Categorize tensor quant type
        if t.tensor_type == gguf.GGMLQuantizationType.IQ4_NL:
            iq4_nl_count += 1
        elif t.tensor_type == gguf.GGMLQuantizationType.F32:
            f32_count += 1
        elif t.tensor_type == gguf.GGMLQuantizationType.F16:
            f16_count += 1
        else:
            other_quant_count += 1

        # Calculate active parameters
        if "_exps.weight" in t.name:
            # Routed expert tensor has shape [n_routed, intermediate, dim]
            # Active portion is expert_used_count / expert_count
            if expert_count > 0:
                active_params += int(t_params * (expert_used_count / expert_count))
            else:
                active_params += t_params
        else:
            active_params += t_params

    # Critical architecture node presence
    crit_nodes = [
        "token_embd.weight",
        "blk.0.attn_kv_a_mqa.weight",
        "blk.0.attn_q_a.weight",
        "output_norm.weight",
        "output.weight",
    ]
    has_crit_nodes = all(node in tensor_dict for node in crit_nodes)
    has_moe_nodes = any("ffn_gate_exps.weight" in k for k in tensor_dict) or any("ffn_gate.weight" in k for k in tensor_dict)

    training_status = "Trained Checkpoint (Validated)" if has_pt else "Untrained Architectural Base (IQ4_NL)"

    return {
        "model_name": name,
        "filename": path.name,
        "path": str(path),
        "size_bytes": file_size_bytes,
        "size_mb": round(file_size_mb, 2),
        "sha256": sha256_hash,
        "architecture": arch,
        "training_status": training_status,
        "total_tensors": total_tensors,
        "total_params": total_params,
        "total_params_str": f"{total_params / 1e6:.2f}M" if total_params < 1e9 else f"{total_params / 1e9:.2f}B",
        "active_params": active_params,
        "active_params_str": f"{active_params / 1e6:.2f}M" if active_params < 1e9 else f"{active_params / 1e9:.2f}B",
        "quantization": {
            "iq4_nl_tensors": iq4_nl_count,
            "f32_tensors": f32_count,
            "f16_tensors": f16_count,
            "other_tensors": other_quant_count,
            "format": "IQ4_NL (Linear/MoE) + FP32 (Norms/Bias)"
        },
        "dimensions": {
            "dim": embedding_length,
            "n_layers": block_count,
            "n_heads": head_count,
            "q_lora_rank": q_lora_rank,
            "kv_lora_rank": kv_lora_rank,
            "key_length": key_length,
            "value_length": value_length,
            "feed_forward_length": feed_forward_length,
            "moe_intermediate_size": expert_feed_forward_length,
            "n_routed_experts": expert_count,
            "n_shared_experts": expert_shared_count,
            "num_experts_per_tok": expert_used_count,
            "rope_dimension_count": rope_dimension_count,
            "vocab_size": vocab_size,
            "context_length": context_length
        },
        "critical_nodes_verified": has_crit_nodes and has_moe_nodes
    }

def main():
    print("=" * 90)
    print("MARK-AI BINARY FORENSIC X-RAY (GGUFReader Forensic Inspection)")
    print("=" * 90)

    results = []
    for model_name, path, has_pt in MODELS_LIST:
        print(f"[*] Scanning {model_name} -> {path.name}...")
        report = analyze_gguf(model_name, path, has_pt)
        results.append(report)

    # Save to artifacts/gguf_forensic_audit.json
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    out_json = artifacts_dir / "gguf_forensic_audit.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Forensic audit report saved to {out_json}")

    # Print Comparative Markdown Table
    print("\n" + "=" * 120)
    print("COMPARATIVE FORENSIC AUDIT TABLE (11/11 MARK-AI MODELS - NATIVE DEEPSEEK2 IQ4_NL)")
    print("=" * 120)
    headers = [
        "Model Name", "Arch", "Size (MB)", "Tensors", "Total Params", "Active Params",
        "IQ4_NL / F32", "Layers", "Experts (R/S/U)", "Status"
    ]
    row_fmt = "| {:<23} | {:<9} | {:<9} | {:<7} | {:<12} | {:<13} | {:<12} | {:<6} | {:<13} | {:<25} |"
    print(row_fmt.format(*headers))
    print("|" + "|".join(["-" * 25, "-" * 11, "-" * 11, "-" * 9, "-" * 14, "-" * 15, "-" * 14, "-" * 8, "-" * 15, "-" * 27]) + "|")

    for r in results:
        q_str = f"{r['quantization']['iq4_nl_tensors']} / {r['quantization']['f32_tensors']}"
        d = r["dimensions"]
        exp_str = f"{d['n_routed_experts']}/{d['n_shared_experts']}/{d['num_experts_per_tok']}"
        print(row_fmt.format(
            r["model_name"],
            r["architecture"],
            f"{r['size_mb']} MB",
            r["total_tensors"],
            r["total_params_str"],
            r["active_params_str"],
            q_str,
            d["n_layers"],
            exp_str,
            r["training_status"]
        ))
    print("=" * 120 + "\n")

if __name__ == "__main__":
    main()
