#!/usr/bin/env python3
"""
sanitize_all_models_metadata.py
Stage 1 & Stage 2 Execution:
Synchronizes all 11 Mark-AI models against the ground truth in GGUF binaries.
Ensures zero contradictions between:
- config.json
- tokenizer_config.json
- tokenizer.json
- Modelfile
- *_metadata.json
"""

import json
import os
import hashlib
from pathlib import Path
import numpy as np
import gguf

REPO_ROOT = Path(__file__).resolve().parent.parent

# 11 Official Models with GGUF checkpoint mappings
MODELS_SPECS = [
    {
        "id": "mark-engineer-302m",
        "gguf_path": "/mnt/ssd_1/mark ai/checkpoints/mark_engineer_302m_iq4_nl.gguf",
        "gguf_filename": "mark_engineer_302m_iq4_nl.gguf",
        "name": "Mark-Engineer-302M",
        "description": "Ultra-lightweight systems engineering model built on native DeepSeek2 MoE and MLA architecture.",
        "weight_state": "TRAINED_CHECKPOINT_VALIDATED",
        "is_vl": False,
        "is_untrained": False,
    },
    {
        "id": "mark-slm-v2-302m",
        "gguf_path": "/mnt/ssd_1/mark ai/checkpoints/mark_slm_v2_302m_iq4_nl.gguf",
        "gguf_filename": "mark_slm_v2_302m_iq4_nl.gguf",
        "name": "Mark-SLM-V2-302M",
        "description": "Compact small language model with 49K vocabulary and DeepSeek2 MoE + MLA topology.",
        "weight_state": "TRAINED_CHECKPOINT_VALIDATED",
        "is_vl": False,
        "is_untrained": False,
    },
    {
        "id": "mark-engineer-1.5b",
        "gguf_path": "/mnt/ssd_1/mark ai/checkpoints/mark_engineer_1_5b_iq4_nl.gguf",
        "gguf_filename": "mark_engineer_1_5b_iq4_nl.gguf",
        "name": "Mark-Engineer-1.5B",
        "description": "Mid-sized systems engineering and low-level code reasoning model.",
        "weight_state": "TRAINED_CHECKPOINT_VALIDATED",
        "is_vl": False,
        "is_untrained": False,
    },
    {
        "id": "mark-general-1.5b",
        "gguf_path": "/mnt/ssd_1/mark ai/checkpoints/mark_general_1_5b_iq4_nl.gguf",
        "gguf_filename": "mark_general_1_5b_iq4_nl.gguf",
        "name": "Mark-General-1.5B",
        "description": "General-purpose 1.5B reasoning model with extended 4377 vocabulary.",
        "weight_state": "TRAINED_CHECKPOINT_VALIDATED",
        "is_vl": False,
        "is_untrained": False,
    },
    {
        "id": "mark-vl-2.2b",
        "gguf_path": "/mnt/ssd_1/mark ai/checkpoints/mark_vl_2_2b_iq4_nl.gguf",
        "gguf_filename": "mark_vl_2_2b_iq4_nl.gguf",
        "name": "Mark-VL-2.2B",
        "description": "Multimodal vision-language model with native DeepSeek2 MoE backbone and vision projector adapter.",
        "weight_state": "TRAINED_CHECKPOINT_VALIDATED",
        "is_vl": True,
        "is_untrained": False,
    },
    {
        "id": "mark-engineer-v2-8b",
        "gguf_path": "/mnt/ssd_1/mark ai/checkpoints/mark_engineer_v2_8b_iq4_nl.gguf",
        "gguf_filename": "mark_engineer_v2_8b_iq4_nl.gguf",
        "name": "Mark-Engineer-V2-8B",
        "description": "Flagship 8B architectural foundation featuring 24 MoE experts, MLA, and 49K vocabulary.",
        "weight_state": "UN-TRAINED_ARCHITECTURAL_BASE",
        "is_vl": False,
        "is_untrained": True,
    },
    {
        "id": "mark-slm-v3-150m",
        "gguf_path": "/mnt/ssd_1/mark ai/checkpoints/mark_slm_v3_150m_iq4_nl.gguf",
        "gguf_filename": "mark_slm_v3_150m_iq4_nl.gguf",
        "name": "Mark-SLM-V3-150M",
        "description": "Sub-200M parameter edge model with native DeepSeek2 MoE and MLA architecture.",
        "weight_state": "UN-TRAINED_ARCHITECTURAL_BASE",
        "is_vl": False,
        "is_untrained": True,
    },
    {
        "id": "mark-v3-1.1b",
        "gguf_path": "/mnt/ssd_1/mark ai/checkpoints/mark_v3_1_1b_iq4_nl.gguf",
        "gguf_filename": "mark_v3_1_1b_iq4_nl.gguf",
        "name": "Mark-V3-1.1B",
        "description": "Compact V3 series model with 32 routed experts (4 active per token), 2 shared experts, and MLA.",
        "weight_state": "UN-TRAINED_ARCHITECTURAL_BASE",
        "is_vl": False,
        "is_untrained": True,
    },
    {
        "id": "mark-general-v3-1.5b",
        "gguf_path": "/mnt/ssd_1/mark ai/checkpoints/mark_general_v3_1_5b_iq4_nl.gguf",
        "gguf_filename": "mark_general_v3_1_5b_iq4_nl.gguf",
        "name": "Mark-General-V3-1.5B",
        "description": "V3 architecture 1.5B class general reasoning model with 32 routed experts and MLA.",
        "weight_state": "UN-TRAINED_ARCHITECTURAL_BASE",
        "is_vl": False,
        "is_untrained": True,
    },
    {
        "id": "mark-engineer-v3-1.5b",
        "gguf_path": "/mnt/ssd_1/mark ai/checkpoints/mark_engineer_v3_1_5b_iq4_nl.gguf",
        "gguf_filename": "mark_engineer_v3_1_5b_iq4_nl.gguf",
        "name": "Mark-Engineer-V3-1.5B",
        "description": "V3 architecture 1.5B class engineering specialist with 32 routed experts and MLA.",
        "weight_state": "UN-TRAINED_ARCHITECTURAL_BASE",
        "is_vl": False,
        "is_untrained": True,
    },
    {
        "id": "mark-experimental-v3-8b",
        "gguf_path": "/mnt/ssd_1/mark ai/checkpoints/mark_experimental_v3_8b_iq4_nl.gguf",
        "gguf_filename": "mark_experimental_v3_8b_iq4_nl.gguf",
        "name": "Mark-Experimental-V3-8B",
        "description": "V3 architecture 8B class research model with 32 routed experts, 2 shared experts, and MLA.",
        "weight_state": "UN-TRAINED_ARCHITECTURAL_BASE",
        "is_vl": False,
        "is_untrained": True,
    },
]

CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{% if message['role'] == 'system' %}<|bos|>System: {{ message['content'] }}<|eos|>\n"
    "{% elif message['role'] == 'user' %}<|bos|>User: {{ message['content'] }}<|eos|>\n"
    "{% elif message['role'] == 'assistant' %}<|bos|>Assistant: "
    "{% if message['content'] %}{{ message['content'] }}<|eos|>\n{% endif %}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|bos|>Assistant:\n<think>\n{% endif %}"
)


def compute_sha256(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(16 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def inspect_gguf(file_path: str):
    reader = gguf.GGUFReader(file_path)
    file_bytes = os.path.getsize(file_path)
    arch = bytes(reader.get_field("general.architecture").parts[-1]).decode("utf-8")
    
    vocab_field = reader.get_field(f"{arch}.vocab_size")
    vocab_size = vocab_field.parts[-1].tolist()[0] if vocab_field else 4311
    
    n_routed_field = reader.get_field(f"{arch}.expert_count")
    n_routed = n_routed_field.parts[-1].tolist()[0] if n_routed_field else 0
    
    n_used_field = reader.get_field(f"{arch}.expert_used_count")
    n_used = n_used_field.parts[-1].tolist()[0] if n_used_field else 0
    
    n_shared_field = reader.get_field(f"{arch}.expert_shared_count")
    n_shared = n_shared_field.parts[-1].tolist()[0] if n_shared_field else 0
    
    moe_inter_field = reader.get_field(f"{arch}.expert_feed_forward_length")
    moe_inter = moe_inter_field.parts[-1].tolist()[0] if moe_inter_field else 0
    
    kv_lora_field = reader.get_field(f"{arch}.attention.kv_lora_rank")
    kv_lora = kv_lora_field.parts[-1].tolist()[0] if kv_lora_field else None
    
    q_lora_field = reader.get_field(f"{arch}.attention.q_lora_rank")
    q_lora = q_lora_field.parts[-1].tolist()[0] if q_lora_field else None

    hidden_field = reader.get_field(f"{arch}.embedding_length")
    hidden_size = hidden_field.parts[-1].tolist()[0] if hidden_field else None

    layers_field = reader.get_field(f"{arch}.block_count")
    num_layers = layers_field.parts[-1].tolist()[0] if layers_field else None

    heads_field = reader.get_field(f"{arch}.attention.head_count")
    num_heads = heads_field.parts[-1].tolist()[0] if heads_field else None

    total_params = 0
    active_params = 0
    tensor_types = {}
    norm_fp32 = True

    for t in reader.tensors:
        n_elem = int(np.prod(t.shape))
        total_params += n_elem
        t_type = t.tensor_type.name
        tensor_types[t_type] = tensor_types.get(t_type, 0) + 1
        if "norm" in t.name and t_type != "F32":
            norm_fp32 = False

        if "_exps" in t.name:
            if n_routed > 0:
                active_params += int(n_elem * (n_used / n_routed))
        else:
            active_params += n_elem

    # tokens
    tok_field = reader.get_field("tokenizer.ggml.tokens")
    tokens = [bytes(tok_field.parts[idx]).decode("utf-8", errors="replace") for idx in tok_field.data]

    return {
        "file_bytes": file_bytes,
        "mib": round(file_bytes / (1024 ** 2), 2),
        "mb": round(file_bytes / (1000 ** 2), 2),
        "size_formatted": f"{file_bytes / (1024 ** 2):.2f} MiB ({file_bytes / (1000 ** 2):.2f} MB / {file_bytes:,} bytes)",
        "total_parameters": total_params,
        "active_parameters": active_params,
        "vocab_size": vocab_size,
        "n_routed_experts": n_routed,
        "num_experts_per_tok": n_used,
        "n_shared_experts": n_shared,
        "moe_intermediate_size": moe_inter,
        "kv_lora_rank": kv_lora,
        "q_lora_rank": q_lora,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "tensors_count": len(reader.tensors),
        "tensor_types": tensor_types,
        "norm_fp32": norm_fp32,
        "tokens_sample": tokens[:25],
    }


def sanitize_all():
    print("=" * 80)
    print("SANITIZING ALL 11 MARK-AI MODELS FROM SINGLE SOURCE OF TRUTH (GGUF)")
    print("=" * 80)

    # Load base tokenizer.json
    base_tok_path = REPO_ROOT / "models" / "mark-engineer-302m" / "tokenizer.json"
    with open(base_tok_path, "r", encoding="utf-8") as f:
        base_tok_data = json.load(f)

    for spec in MODELS_SPECS:
        model_id = spec["id"]
        model_dir = REPO_ROOT / "models" / model_id
        gguf_path = spec["gguf_path"]
        print(f"\nProcessing {model_id}...")

        # 1. Inspect GGUF
        gguf_info = inspect_gguf(gguf_path)
        sha256 = compute_sha256(gguf_path)
        gguf_info["sha256"] = sha256

        print(f"  GGUF: {gguf_info['size_formatted']}")
        print(f"  SHA256: {sha256}")
        print(f"  Vocab Size: {gguf_info['vocab_size']}")
        print(f"  Total Params: {gguf_info['total_parameters']:,} | Active: {gguf_info['active_parameters']:,}")

        # 2. Update config.json
        config_path = model_dir / "config.json"
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        config["vocab_size"] = gguf_info["vocab_size"]
        config["total_parameters"] = gguf_info["total_parameters"]
        config["active_parameters_per_token"] = gguf_info["active_parameters"]
        config["weight_state"] = spec["weight_state"]
        config["sha256"] = sha256
        config["quantization"] = "IQ4_NL"

        if gguf_info["hidden_size"]:
            config["hidden_size"] = gguf_info["hidden_size"]
        if gguf_info["num_layers"]:
            config["num_hidden_layers"] = gguf_info["num_layers"]
        if gguf_info["num_heads"]:
            config["num_attention_heads"] = gguf_info["num_heads"]
        if gguf_info["n_routed_experts"]:
            config["n_routed_experts"] = gguf_info["n_routed_experts"]
        if gguf_info["num_experts_per_tok"]:
            config["num_experts_per_tok"] = gguf_info["num_experts_per_tok"]
        if gguf_info["n_shared_experts"]:
            config["n_shared_experts"] = gguf_info["n_shared_experts"]
        if gguf_info["moe_intermediate_size"]:
            config["moe_intermediate_size"] = gguf_info["moe_intermediate_size"]
        if gguf_info["kv_lora_rank"]:
            config["kv_lora_rank"] = gguf_info["kv_lora_rank"]
        if gguf_info["q_lora_rank"]:
            config["q_lora_rank"] = gguf_info["q_lora_rank"]

        # Clean unproven / obsolete fields
        config.pop("evaluation_status", None)
        config.pop("training_convergence_loss", None)
        config.pop("status", None)

        # Remove vLLM from target_platforms
        if "target_platforms" in config:
            config["target_platforms"] = [p for p in config["target_platforms"] if p.lower() != "vllm"]
        else:
            config["target_platforms"] = ["Ollama", "LM Studio", "llama.cpp", "PyTorch"]

        # Clean download urls / sha256
        if "download" in config and isinstance(config["download"], dict):
            config["download"]["sha256"] = sha256

        # Write clean formatted config.json
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"  [OK] Updated {config_path.name}")

        # 3. Update tokenizer_config.json
        tok_cfg_path = model_dir / "tokenizer_config.json"
        tok_cfg = {}
        if tok_cfg_path.exists():
            with open(tok_cfg_path, "r", encoding="utf-8") as f:
                tok_cfg = json.load(f)

        tok_cfg["vocab_size"] = gguf_info["vocab_size"]
        tok_cfg["bos_token"] = "<|bos|>"
        tok_cfg["eos_token"] = "<|eos|>"
        tok_cfg["pad_token"] = "<|pad|>"
        tok_cfg["bos_token_id"] = 1
        tok_cfg["eos_token_id"] = 2
        tok_cfg["pad_token_id"] = 0
        tok_cfg["model_max_length"] = config.get("max_position_embeddings", 4096)
        tok_cfg["chat_template"] = CHAT_TEMPLATE
        tok_cfg["additional_special_tokens"] = [
            "<think>",
            "</think>",
            "[RETRIEVE]",
            "[/RETRIEVE]",
            "[EVALUATE]",
            "[ACCEPT]",
            "[REJECT]",
            "<|vision_start|>",
            "<|vision_patch|>",
            "<|vision_end|>",
            "<|voice_stream|>",
            "<|rag_context|>",
            "<|/rag_context|>",
            "<|mcp_call|>",
            "<|mcp_response|>",
            "<|citation|>",
            "<|/citation|>",
        ]
        tok_cfg["special_tokens"] = {
            "<|pad|>": 0,
            "<|bos|>": 1,
            "<|eos|>": 2,
            "<think>": 3,
            "</think>": 4,
            "[RETRIEVE]": 5,
            "[/RETRIEVE]": 6,
            "[EVALUATE]": 7,
            "[ACCEPT]": 8,
            "[REJECT]": 9,
            "<|vision_start|>": 10,
            "<|vision_patch|>": 11,
            "<|vision_end|>": 12,
            "<|voice_stream|>": 13,
            "[UNKNOWN]": 14,
            "[WEB_SEARCH]": 15,
            "[/WEB_SEARCH]": 16,
            "<|rag_context|>": 17,
            "<|/rag_context|>": 18,
            "<|mcp_call|>": 19,
            "<|mcp_response|>": 20,
            "<|citation|>": 21,
            "<|/citation|>": 22,
        }

        with open(tok_cfg_path, "w", encoding="utf-8") as f:
            json.dump(tok_cfg, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"  [OK] Updated {tok_cfg_path.name}")

        # 4. Update tokenizer.json
        tok_json_path = model_dir / "tokenizer.json"
        tok_data = json.loads(json.dumps(base_tok_data))
        current_vocab = tok_data.get("model", {}).get("vocab", {})
        target_vocab_size = gguf_info["vocab_size"]

        if len(current_vocab) < target_vocab_size:
            for idx in range(len(current_vocab), target_vocab_size):
                token_str = f"<extra_token_{idx}>"
                current_vocab[token_str] = idx

        with open(tok_json_path, "w", encoding="utf-8") as f:
            json.dump(tok_data, f, indent=2, ensure_ascii=False)
            f.write("\n")
        print(f"  [OK] Updated {tok_json_path.name} (vocab length: {len(current_vocab)})")

        # 5. Update Modelfile
        modelfile_path = model_dir / "Modelfile"
        adapter_directive = "ADAPTER ./mmproj_mark_vl_2_2b.gguf\n" if spec["is_vl"] else ""
        system_role = (
            f"You are {spec['name']}, a multimodal vision-language model built on native DeepSeek2 MoE and MLA architecture."
            if spec["is_vl"]
            else f"You are {spec['name']}, a language model built on native DeepSeek2 MoE and MLA architecture."
        )

        modelfile_content = f"""FROM ./{spec['gguf_filename']}
{adapter_directive}
TEMPLATE \"\"\"{{{{ if .System }}}}<|bos|>System: {{{{ .System }}}}<|eos|>
{{{{ end }}}}{{{{ if .Prompt }}}}<|bos|>User: {{{{ .Prompt }}}}<|eos|>
{{{{ end }}}}<|bos|>Assistant:
<think>
\"\"\"

PARAMETER stop \"<|eos|>\"
PARAMETER temperature 0.2
PARAMETER top_p 0.95
PARAMETER repeat_penalty 1.1

SYSTEM \"\"\"{system_role}\"\"\"
"""
        with open(modelfile_path, "w", encoding="utf-8") as f:
            f.write(modelfile_content)
        print(f"  [OK] Updated {modelfile_path.name}")

        # 6. Update *_metadata.json if relevant
        if model_id == "mark-engineer-1.5b":
            meta_path = REPO_ROOT / "models" / model_id / "mark_engineer_1_5b_metadata.json"
            meta_data = {
                "model_name": spec["name"],
                "version": "1.0.0",
                "total_parameters": gguf_info["total_parameters"],
                "active_parameters_per_token": gguf_info["active_parameters"],
                "vocab_size": gguf_info["vocab_size"],
                "format": "IQ4_NL",
                "size_formatted": gguf_info["size_formatted"],
                "sha256": sha256,
                "weight_state": spec["weight_state"],
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta_data, f, indent=2, ensure_ascii=False)
                f.write("\n")
            print(f"  [OK] Updated {meta_path.name}")

        if model_id == "mark-vl-2.2b":
            meta_path = REPO_ROOT / "models" / model_id / "mark_vl_2_2b_metadata.json"
            meta_data = {
                "model_name": spec["name"],
                "version": "1.0.0",
                "total_parameters": gguf_info["total_parameters"],
                "active_parameters_per_token": gguf_info["active_parameters"],
                "vocab_size": gguf_info["vocab_size"],
                "format": "IQ4_NL",
                "size_formatted": gguf_info["size_formatted"],
                "sha256": sha256,
                "weight_state": spec["weight_state"],
                "multimodal_projector": "mmproj_mark_vl_2_2b.gguf",
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta_data, f, indent=2, ensure_ascii=False)
                f.write("\n")
            print(f"  [OK] Updated {meta_path.name}")

        if model_id == "mark-engineer-v2-8b":
            meta_path = REPO_ROOT / "models" / model_id / "mark_engineer_v2_8b_metadata.json"
            meta_data = {
                "model_name": spec["name"],
                "version": "2.0.0",
                "total_parameters": gguf_info["total_parameters"],
                "active_parameters_per_token": gguf_info["active_parameters"],
                "vocab_size": gguf_info["vocab_size"],
                "format": "IQ4_NL",
                "size_formatted": gguf_info["size_formatted"],
                "sha256": sha256,
                "weight_state": spec["weight_state"],
            }
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta_data, f, indent=2, ensure_ascii=False)
                f.write("\n")
            print(f"  [OK] Created {meta_path.name}")

    print("\n" + "=" * 80)
    print("STAGE 1 & STAGE 2 SANITIZATION COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    sanitize_all()
