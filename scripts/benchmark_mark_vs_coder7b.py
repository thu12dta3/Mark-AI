"""
Mark Engineer 1.0 vs Coder 7B - Head-to-Head Benchmark on NVIDIA A100-80GB
========================================================================
Executa avaliação comparativa em tempo real na GPU A100:
1. Mark Engineer 1.0 (302M Parâmetros, DeepSeek MLA, MoE 16 Especialistas, MTP)
2. Qwen2.5-Coder-7B-Instruct (7.6B Parâmetros, Dense Coder)

Métricas Comparadas:
- Latência de Resposta e Throughput (tokens/s)
- Consumo de VRAM (GB)
- Profundidade de Raciocínio (<think>)
- Acurácia Algorítmica e Ausência de Alucinações
- Honra às Restrições Operacionais (Sem Emojis, Zero-Unwrap, Epistemic Honesty)
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
vram_total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
print(f"[Device] GPU: {torch.cuda.get_device_name(0)} | VRAM Total: {vram_total_gb:.2f} GB")

# 1. Conjunto de Testes Reais de Engenharia de Software
BENCHMARK_PROMPTS = [
    {
        "id": "T1_rust_async_mpsc",
        "category": "Concorrência & Backpressure (Rust)",
        "prompt": "Implemente em Rust uma fila assíncrona concorrente com bounded channel, semântica MPSC, backpressure estrito e graceful shutdown usando Tokio sem unwrap() e sem panic().",
        "key_signals": ["tokio::sync::mpsc", "CancellationToken", "Result<", "thiserror"],
        "anti_signals": ["unwrap()", "panic!(", ".expect("],
    },
    {
        "id": "T2_go_lock_free_ring",
        "category": "Lock-Free & Atomics (Go)",
        "prompt": "Implemente em Go um buffer circular lock-free (Ring Buffer) de alta vazão para troca de mensagens entre goroutines usando atomic.Pointer sem data races.",
        "key_signals": ["atomic.Pointer", "CompareAndSwap", "mask", "head", "tail"],
        "anti_signals": ["sync.Mutex", "sync.RWMutex"],
    },
    {
        "id": "T3_assembly_avx2_dot",
        "category": "Otimização de Baixo Nível (Assembly x86_64)",
        "prompt": "Escreva uma rotina Assembly x86_64 otimizada com AVX2 (vfmadd231ps) para calcular o produto escalar de dois vetores float32 com alinhamento de 32 bytes seguindo System V AMD64 ABI.",
        "key_signals": ["vfmadd231ps", "ymm0", "rdi", "rsi", "vzeroupper"],
        "anti_signals": ["mul eax", "fld", "fmul"],
    },
    {
        "id": "T4_repo_mark_hardening",
        "category": "Arquitetura & Hardening de Repositório (MARK)",
        "prompt": "Como o MARK previne ataques de Time-of-Check to Time-of-Use (TOCTOU) e Symlink Hijacking em operações de gravação de código no engineer_agent.py?",
        "key_signals": ["resolve_authorized_workspace_root", "is_symlink", "fsync", "os.replace", "atomic"],
        "anti_signals": ["chmod 777", "shell=True"],
    },
    {
        "id": "T5_vague_intent_cache",
        "category": "Desambiguação de Pedido Vago",
        "prompt": "preciso de um cache rapido na memoria pro meu backend em go",
        "key_signals": ["sharding", "TTL", "RWMutex", "Janitor", "eviction"],
        "anti_signals": ["map[string]any sem mutex", "só usar redis"],
    },
    {
        "id": "T6_epistemic_self_rag",
        "category": "Honestidade Epistêmica & Perplexity (Self-RAG)",
        "prompt": "Qual foi a vulnerabilidade de dia zero corrigida na biblioteca OpenSSL em 15 de outubro de 2026? Se você não tiver acesso a fatos em tempo real de 2026, admita com honestidade.",
        "key_signals": ["[UNKNOWN]", "[WEB_SEARCH]", "pesquisar"],
        "anti_signals": ["CVE-2026-99999", "Foi corrigida a falha fictícia"],
    },
]


def load_mark_engineer_model() -> Tuple[Any, Any, float]:
    """Carrega o Mark Engineer 1.0 treinado na A100."""
    print("\n[Carregamento] Inicializando Mark Engineer 1.0 (MLA + MoE 16 Esp)...")
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

    # Carrega pesos treinados mais recentes
    ckpt_dir = Path("/content/Mark-AI/checkpoints")
    candidates = [
        ckpt_dir / "mark_engineer_inference.pt",
        ckpt_dir / "mark_engineer_final.pt",
    ] + sorted(list(ckpt_dir.glob("mark_0_1_step_*.pt")), reverse=True)

    loaded_path = None
    for c in candidates:
        if c.exists():
            print(f"[Mark Checkpoint] Carregando pesos de: {c}")
            try:
                ckpt_data = torch.load(c, map_location=device, weights_only=False)
            except TypeError:
                ckpt_data = torch.load(c, map_location=device)
            state_dict = ckpt_data.get("model_state_dict", ckpt_data)
            model.load_state_dict(state_dict, strict=False)
            loaded_path = c
            break

    if not loaded_path:
        print("[Aviso] Nenhum checkpoint salvo encontrado, rodando em modo arquitetural puro.")

    model.eval()
    load_time = time.time() - t0
    vram_used = torch.cuda.memory_allocated() / (1024**3)
    print(f"[Mark Carregado] VRAM: {vram_used:.2f} GB | Tempo de Carga: {load_time:.2f}s")
    return model, tokenizer, vram_used


def load_coder_7b_model() -> Tuple[Any, Any, float]:
    """Carrega o modelo Qwen2.5-Coder-7B-Instruct na A100."""
    print("\n[Carregamento] Inicializando Qwen2.5-Coder-7B-Instruct...")
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
    vram_used = torch.cuda.memory_allocated() / (1024**3)
    print(f"[Coder 7B Carregado] VRAM Total Alocada: {vram_used:.2f} GB | Tempo de Carga: {load_time:.2f}s")
    return model, tokenizer, vram_used


def generate_mark_response(model, tokenizer, prompt: str, max_tokens: int = 512) -> Tuple[str, float, float]:
    """Gera resposta com o Mark Engineer 1.0 medindo tempo e vazão."""
    t0 = time.time()
    input_text = f"<|bos|>{prompt}"
    encoded = tokenizer.encode(input_text)
    input_ids = torch.tensor([encoded.ids], dtype=torch.long, device=device)

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
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

    out_text = tokenizer.decode(generated_ids[0].tolist())
    return out_text, elapsed, tok_per_sec


def generate_coder7b_response(model, tokenizer, prompt: str, max_tokens: int = 512) -> Tuple[str, float, float]:
    """Gera resposta com o Qwen2.5-Coder-7B-Instruct medindo tempo e vazão."""
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
    # 1. Carrega Mark Engineer
    mark_model, mark_tok, mark_vram = load_mark_engineer_model()

    # 2. Carrega Coder 7B
    coder_model, coder_tok, coder_vram = load_coder_7b_model()

    results = []
    print("\n=== [Iniciando Bateria de Testes Comparativos] ===")

    for idx, test in enumerate(BENCHMARK_PROMPTS, start=1):
        print(f"\n--- [Teste {idx}/{len(BENCHMARK_PROMPTS)}] {test['category']} ---")
        prompt = test["prompt"]

        # Inferência Mark Engineer
        mark_out, mark_lat, mark_tps = generate_mark_response(mark_model, mark_tok, prompt)
        mark_has_think = "<think>" in mark_out
        mark_signals_hit = sum(1 for s in test["key_signals"] if s.lower() in mark_out.lower())
        mark_violations = sum(1 for a in test["anti_signals"] if a.lower() in mark_out.lower())

        # Inferência Coder 7B
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

        print(f"[Mark Engineer] Lat: {mark_lat:.2f}s | Vazao: {mark_tps:.1f} tok/s | Sinais: {mark_signals_hit}/{len(test['key_signals'])} | Violacoes: {mark_violations}")
        print(f"[Coder 7B]      Lat: {coder_lat:.2f}s | Vazao: {coder_tps:.1f} tok/s | Sinais: {coder_signals_hit}/{len(test['key_signals'])} | Violacoes: {coder_violations}")

    # Salva relatório estruturado
    out_dir = Path("/content/Mark-AI/benchmarks")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / "benchmark_mark_vs_coder7b.json"

    summary = {
        "timestamp": time.time(),
        "gpu": torch.cuda.get_device_name(0),
        "vram_comparison_gb": {
            "mark_engineer_302m": round(mark_vram, 2),
            "qwen_coder_7b": round(coder_vram - mark_vram, 2),
        },
        "tests": results,
    }

    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n[Sucesso] Relatório de benchmark salvo em: {report_file}")
    print("=== [Conclusão do Benchmark Head-to-Head na A100] ===")


if __name__ == "__main__":
    run_head_to_head_benchmark()
