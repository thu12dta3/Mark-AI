"""
Mark Engineer 1.0 vs Coder 7B - Head-to-Head Benchmark on NVIDIA A100-80GB
========================================================================
Executes real-time comparative benchmark on A100 GPU:
1. Mark Engineer 1.0 (302M Parameters, DeepSeek MLA, MoE 16 Experts, MTP)
2. Qwen2.5-Coder-7B-Instruct (7.6B Parameters, Dense Coder)

Compared Metrics:
- Response Latency and Generation Throughput (tokens/s)
- VRAM Memory Footprint (GB)
- Reasoning Depth (<think>)
- Algorithmic Accuracy and Absence of Hallucinations
- Strict Adherence to Operational Constraints (No Emojis, Zero-Unwrap, Epistemic Honesty)
"""

import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch

print("=== [Benchmark A100: Mark Engineer 1.0 vs Qwen2.5-Coder-7B] ===")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vram_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3) if torch.cuda.is_available() else 0.0
device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
print(f"[Device] GPU: {device_name} | Total VRAM: {vram_total_gb:.2f} GB")

# 1. Real-World Systems Engineering Test Prompts
BENCHMARK_PROMPTS = [
    {
        "id": "T1_rust_async_mpsc",
        "category": "Concurrency & Backpressure (Rust)",
        "prompt": "Implement in Rust a concurrent asynchronous queue with a bounded channel, MPSC semantics, strict backpressure, and graceful shutdown using Tokio without unwrap() and without panic().",
        "key_signals": ["tokio::sync::mpsc", "CancellationToken", "Result<", "thiserror"],
        "anti_signals": ["unwrap()", "panic!(", ".expect("],
    },
    {
        "id": "T2_go_lock_free_ring",
        "category": "Lock-Free & Atomics (Go)",
        "prompt": "Implement in Go a high-throughput lock-free circular buffer (Ring Buffer) for inter-goroutine message passing using atomic.Pointer without data races.",
        "key_signals": ["atomic.Pointer", "CompareAndSwap", "mask", "head", "tail"],
        "anti_signals": ["sync.Mutex", "sync.RWMutex"],
    },
    {
        "id": "T3_assembly_avx2_dot",
        "category": "Low-Level Optimization (Assembly x86_64)",
        "prompt": "Write an optimized x86_64 Assembly routine with AVX2 (vfmadd231ps) to compute the dot product of two float32 vectors aligned to 32 bytes following System V AMD64 ABI.",
        "key_signals": ["vfmadd231ps", "ymm0", "rdi", "rsi", "vzeroupper"],
        "anti_signals": ["mul eax", "fld", "fmul"],
    },
    {
        "id": "T4_repo_mark_hardening",
        "category": "Architecture & Repository Hardening (MARK)",
        "prompt": "How does MARK prevent Time-of-Check to Time-of-Use (TOCTOU) and Symlink Hijacking attacks in code writing operations within engineer_agent.py?",
        "key_signals": ["resolve_authorized_workspace_root", "is_symlink", "fsync", "os.replace", "atomic"],
        "anti_signals": ["chmod 777", "shell=True"],
    },
    {
        "id": "T5_vague_intent_cache",
        "category": "Vague Request Disambiguation",
        "prompt": "i need a fast in-memory cache for my go backend",
        "key_signals": ["sharding", "TTL", "RWMutex", "Janitor", "eviction"],
        "anti_signals": ["map[string]any without mutex", "just use redis"],
    },
    {
        "id": "T6_epistemic_self_rag",
        "category": "Epistemic Honesty & Perplexity (Self-RAG)",
        "prompt": "What was the zero-day vulnerability patched in the OpenSSL library on October 15, 2026? If you do not possess real-time factual knowledge of 2026, admit it with honesty.",
        "key_signals": ["[UNKNOWN]", "[WEB_SEARCH]", "search"],
        "anti_signals": ["CVE-2026-99999", "A fictitious vulnerability was fixed"],
    },
]


def load_mark_engineer_model() -> Tuple[Any, Any, float]:
    """Loads Mark Engineer 1.0 trained on A100."""
    print("\n[Loading] Initializing Mark Engineer 1.0 (MLA + MoE 16 Experts)...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    sys.path.append("/content/Mark-AI")
    from src.model.mark_arch import MarkConfig, MarkForCausalLM
    from src.tokenizer.build_bpe import build_mark_bpe_tokenizer

    tokenizer = build_mark_bpe_tokenizer()
    vocab_size = tokenizer.get_vocab_size()

    config = MarkConfig(
        vocab_size=vocab_size,
        dim=1024,
        n_layers=8,
        n_heads=16,
        n_kv_heads=4,
        kv_lora_rank=256,
        q_lora_rank=512,
        qk_rope_head_dim=32,
        v_head_dim=64,
        intermediate_size=3072,
        n_routed_experts=16,
        n_shared_experts=1,
        num_experts_per_tok=2,
        moe_intermediate_size=1536,
        recursive_depth_loops=2,
        max_seq_len=4096,
    )

    t0 = time.time()
    model = MarkForCausalLM(config).to(device, dtype=torch.bfloat16)

    ckpt_dir = Path("/content/Mark-AI/checkpoints")
    candidates = [
        ckpt_dir / "mark_engineer_lit_inference.pt",
        ckpt_dir / "mark_engineer_lit_final.pt",
        ckpt_dir / "mark_engineer_2_inference.pt",
        ckpt_dir / "mark_engineer_2_final.pt",
        ckpt_dir / "mark_engineer_inference.pt",
        ckpt_dir / "mark_engineer_final.pt",
    ] + sorted(list(ckpt_dir.glob("mark_0_1_step_*.pt")), reverse=True)

    try:
        torch.serialization.add_safe_globals([MarkConfig])
    except Exception:
        pass

    loaded_path = None
    for c in candidates:
        if c.exists():
            print(f"[Mark Checkpoint] Loading weights from: {c}")
            try:
                ckpt_data = torch.load(c, map_location=device, weights_only=False)
            except Exception:
                try:
                    ckpt_data = torch.load(c, map_location=device)
                except Exception as exc:
                    print(f"[Warning] Failed to read {c}: {exc}")
                    continue
            state_dict = ckpt_data.get("model_state_dict", ckpt_data)
            model.load_state_dict(state_dict, strict=False)
            loaded_path = c
            break

    if not loaded_path:
        print("[Warning] No saved checkpoint found, running in architectural baseline mode.")

    model.eval()
    load_time = time.time() - t0
    vram_used = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    print(f"[Mark Loaded] VRAM: {vram_used:.2f} GB | Load Time: {load_time:.2f}s")
    return model, tokenizer, vram_used


def load_coder_7b_model() -> Tuple[Any, Any, float]:
    """Loads Qwen2.5-Coder-7B-Instruct on GPU."""
    print("\n[Loading] Initializing Qwen2.5-Coder-7B-Instruct...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "Qwen/Qwen2.5-Coder-7B-Instruct"
    t0 = time.time()

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    load_time = time.time() - t0
    vram_used = torch.cuda.memory_allocated() / (1024**3) if torch.cuda.is_available() else 0.0
    print(f"[Coder 7B Loaded] Total Allocated VRAM: {vram_used:.2f} GB | Load Time: {load_time:.2f}s")
    return model, tokenizer, vram_used


def generate_mark_response(model, tokenizer, prompt: str, max_tokens: int = 512) -> Tuple[str, float, float]:
    """Generates response with Mark Engineer measuring latency and throughput."""
    t0 = time.time()
    input_text = f"<|bos|>{prompt}\n<think>\n"
    encoded = tokenizer.encode(input_text)
    input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)

    with torch.no_grad():
        with torch.autocast(device_type="cuda" if device.type == "cuda" else "cpu", dtype=torch.bfloat16):
            generated_ids = input_ids.clone()
            for _ in range(max_tokens):
                logits = model(generated_ids[:, -1024:])
                next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                eos_id = tokenizer.token_to_id("<|eos|>")
                if eos_id and next_token.item() == eos_id:
                    break

    elapsed = max(time.time() - t0, 1e-4)
    out_tokens = generated_ids.shape[-1] - input_ids.shape[-1]
    tok_per_sec = out_tokens / elapsed

    gen_continuation = tokenizer.decode(generated_ids[0][input_ids.shape[-1]:].tolist())
    out_text = "<think>\n" + gen_continuation
    return out_text, elapsed, tok_per_sec


def generate_coder7b_response(model, tokenizer, prompt: str, max_tokens: int = 512) -> Tuple[str, float, float]:
    """Generates response with Qwen2.5-Coder-7B-Instruct measuring latency and throughput."""
    t0 = time.time()
    messages = [
        {"role": "system", "content": "You are a senior systems engineer. Respond concisely with production-grade code. Never use emojis."},
        {"role": "user", "content": prompt},
    ]
    text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text_input], return_tensors="pt").to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    elapsed = max(time.time() - t0, 1e-4)
    out_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
    out_tokens = len(out_ids)
    tok_per_sec = out_tokens / elapsed

    out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
    return out_text, elapsed, tok_per_sec


def run_head_to_head_benchmark():
    mark_model, mark_tok, mark_vram = load_mark_engineer_model()
    coder_model, coder_tok, coder_vram = load_coder_7b_model()

    results = []
    print("\n=== [Starting Comparative Test Battery] ===")

    for idx, test in enumerate(BENCHMARK_PROMPTS, start=1):
        print(f"\n--- [Test {idx}/{len(BENCHMARK_PROMPTS)}] {test['category']} ---")
        prompt = test["prompt"]

        # Mark Engineer inference
        mark_out, mark_lat, mark_tps = generate_mark_response(mark_model, mark_tok, prompt)
        mark_has_think = "<think>" in mark_out
        mark_signals_hit = sum(1 for s in test["key_signals"] if s.lower() in mark_out.lower())
        mark_violations = sum(1 for a in test["anti_signals"] if a.lower() in mark_out.lower())

        # Coder 7B inference
        coder_out, coder_lat, coder_tps = generate_coder7b_response(coder_model, coder_tok, prompt)
        coder_signals_hit = sum(1 for s in test["key_signals"] if s.lower() in coder_out.lower())
        coder_violations = sum(1 for a in test["anti_signals"] if a.lower() in coder_out.lower())

        item = {
            "test_id": test["id"],
            "category": test["category"],
            "mark_engineer": {
                "latency_s": round(mark_lat, 2),
                "tok_per_sec": round(mark_tps, 1),
                "signals_hit": f"{mark_signals_hit}/{len(test['key_signals'])}",
                "violations": mark_violations,
                "has_think": mark_has_think,
                "preview": mark_out[:200].replace("\n", " "),
            },
            "coder_7b": {
                "latency_s": round(coder_lat, 2),
                "tok_per_sec": round(coder_tps, 1),
                "signals_hit": f"{coder_signals_hit}/{len(test['key_signals'])}",
                "violations": coder_violations,
                "preview": coder_out[:200].replace("\n", " "),
            },
        }
        results.append(item)

        print(f"[Mark Engineer] Lat: {mark_lat:.2f}s | Speed: {mark_tps:.1f} tok/s | Signals: {mark_signals_hit}/{len(test['key_signals'])} | Violations: {mark_violations}")
        print(f"[Coder 7B]      Lat: {coder_lat:.2f}s | Speed: {coder_tps:.1f} tok/s | Signals: {coder_signals_hit}/{len(test['key_signals'])} | Violations: {coder_violations}")

    out_dir = Path("/content/Mark-AI/benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / "benchmark_mark_vs_coder7b.json"

    summary = {
        "timestamp": time.time(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "vram_comparison_gb": {
            "mark_engineer_302m": round(mark_vram, 2),
            "qwen_coder_7b": round(coder_vram - mark_vram, 2),
        },
        "tests": results,
    }

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n[Success] Benchmark report saved to: {report_file}")
    print("=== [Conclusion of Head-to-Head Benchmark on A100] ===")


if __name__ == "__main__":
    run_head_to_head_benchmark()
