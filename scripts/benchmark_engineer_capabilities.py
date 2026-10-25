"""
Mark Engineer - Benchmark de Capacidades & Métricas de Raciocínio (DeepSeek-R1 / MLA)
Avalia rigorosamente o modelo em:
1. Métricas de Raciocínio (<think>) e Resolução Passo-a-Passo
2. Desambiguação de Pedidos Vagos em Arquiteturas Concretas
3. Análise Profunda de Dados e Repositórios (MARK, MARK-PUBLIC, Roblox-LX)
4. Segurança Cibernética Defensiva (AppSec, Hardening P0, Auditoria SAST)
5. Domínio dos Top 20 Idiomas Globais mais Falados
"""

import ast
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class BenchmarkCase:
    case_id: str
    category: str
    prompt: str
    expected_signals: List[str]
    negative_signals: List[str]
    description: str


@dataclass
class BenchmarkResult:
    case_id: str
    category: str
    passed: bool
    score: float
    think_tokens: int
    output_tokens: int
    think_ratio: float
    reasoning_quality: str
    matched_signals: List[str]
    violations: List[str]
    latency_seconds: float


# Top 20 Idiomas Globais mais Falados (ISO 639-1 / Nomes)
TOP_20_GLOBAL_LANGUAGES = [
    ("en", "Inglês", "How to implement a zero-copy ring buffer in Rust?"),
    ("zh-hans", "Mandarim Simplificado", "如何在Go语言中实现高性能无锁循环缓冲区？"),
    ("hi", "Hindi", "पायथन में नॉन-ब्लॉकिंग कनेक्शन पूल कैसे लागू करें?"),
    ("es", "Espanhol", "¿Cómo optimizar una consulta de base de dados con índices compuestos?"),
    ("fr", "Francês", "Comment structurer un pipeline asynchrone avec gestion des erreurs en Rust?"),
    ("ar", "Árabe Padrão", "كيفية تطبيق نموذج الممثلين غير المتزامن باستخدام Tokio؟"),
    ("bn", "Bengali", "সি++ এ মেমরি লিক ছাড়া কাস্টম অ্যালোকেটার কীভাবে তৈরি করবেন?"),
    ("pt", "Português", "Como estruturar o Hardening P0 contra ataques de replay no MARK?"),
    ("ru", "Russo", "Как реализовать атомарный обмен указателями в многопоточном коде на C++?"),
    ("ur", "Urdu", "رسٹ میں محفوظ کنکرنسی کے لیے چینلز کا صحیح استعمال کیسے کریں؟"),
    ("id", "Indonésio", "Bagaimana cara mendesain arsitektur microservice tahan gangguan di Go?"),
    ("de", "Alemão", "Wie implementiert man eine lock-free Queue mit atomaren Operationen in C++20?"),
    ("ja", "Japonês", "RustでTokioを使用した非同期ワーカープールの設計パターンは何ですか？"),
    ("mr", "Marata", "कनकूरंट प्रोग्रामिंगमध्ये डेडलॉक कसा टाळावा?"),
    ("te", "Telugo", "గోలాంగ్‌లో కాంటెక్స్ట్ క్యాన్సిలేషన్‌ను సమర్థవంతంగా ఎలా ఉపయోగించాలి?"),
    ("tr", "Turco", "Python asyncio ile yüksek verimli ağ soketi nasıl yönetilir?"),
    ("ta", "Tâmil", "ரஸ்ட்டில் பாதுகாப்பான நினைவக ஒதுக்கீட்டை எவ்வாறு உறுதி செய்வது?"),
    ("vi", "Vietnamita", "Làm thế nào để xây dựng worker pool không khóa trong C#?"),
    ("ko", "Coreano", "C++20 코루틴을 활용한 비동기 I/O 파이프라인 구현 방법은?"),
    ("it", "Italiano", "Come implementare un circuit breaker resiliente per chiamate gRPC in Go?"),
]


def build_benchmark_test_suite() -> List[BenchmarkCase]:
    cases = [
        # 1. Raciocínio Algorítmico & Métricas de Pensamento (<think>)
        BenchmarkCase(
            case_id="reasoning_lock_free_queue",
            category="reasoning",
            prompt="Implemente uma fila concorrente sem travamento (lock-free) em C++20 com atomic pointers e sem ABA problem.",
            expected_signals=["<think>", "</think>", "std::atomic", "compare_exchange_weak", "ABA", "hazard pointer"],
            negative_signals=["std::mutex", "sleep", "unwrap()"],
            description="Avalia raciocínio algorítmico profundo e resolução do problema ABA com hazard pointers ou tagged pointers.",
        ),
        # 2. Interpretação de Pedido Vago
        BenchmarkCase(
            case_id="vague_intent_fast_cache",
            category="vague_intent",
            prompt="preciso de um cache rapido na memoria pro meu backend em go",
            expected_signals=[
                "<think>",
                "</think>",
                "sync.RWMutex",
                "sharding",
                "TTL",
                "eviction",
                "LRU",
            ],
            negative_signals=["map[string]any", "global lock sem sharding"],
            description="Mede a capacidade de desmembrar um pedido vago em requisitos arquiteturais concretos (sharding de chaves, TTL, política de despejo).",
        ),
        # 3. Análise Profunda de Repositório (MARK / MARK-PUBLIC / Roblox-LX)
        BenchmarkCase(
            case_id="repo_mark_public_tls_bridge",
            category="repo_analysis",
            prompt="Explique como o cliente Android do MARK-PUBLIC valida certificados TLS 1.3 ao falar com o daemon Rust mark_link.",
            expected_signals=[
                "<think>",
                "</think>",
                "Tls13SocketFactory",
                "mobile-android",
                "mark_link",
                "daemon.rs",
                "mobile.token",
                "0600",
            ],
            negative_signals=["TrustAllCerts", "ALLOW_ALL_HOSTNAME_VERIFIER"],
            description="Testa compreensão da integração entre cliente Kotlin e servidor Rust no MARK-PUBLIC.",
        ),
        # 4. Segurança Cibernética Defensiva & Hardening (AppSec P0)
        BenchmarkCase(
            case_id="appsec_race_condition_toctou",
            category="appsec_defensive",
            prompt="Identifique a vulnerabilidade no seguinte código C e forneça a versão endurecida:\n"
                   "if (access(filepath, W_OK) == 0) {\n    FILE *f = fopen(filepath, \"w\");\n    fputs(data, f);\n    fclose(f);\n}",
            expected_signals=[
                "<think>",
                "</think>",
                "TOCTOU",
                "Time-of-Check to Time-of-Use",
                "O_CREAT",
                "O_WRONLY",
                "O_NOFOLLOW",
                "open(",
                "fdopen",
            ],
            negative_signals=["usleep", "access()"],
            description="Detecta condição de corrida TOCTOU e corrige utilizando file descriptors atômicos (O_NOFOLLOW/O_EXCL).",
        ),
        # 5. Módulo Perplexity Self-RAG
        BenchmarkCase(
            case_id="perplexity_self_rag_unknown",
            category="perplexity_rag",
            prompt="Qual foi a vulnerabilidade de dia zero corrigida na biblioteca OpenSSL em 15 de outubro de 2026?",
            expected_signals=[
                "<think>",
                "</think>",
                "[UNKNOWN]",
                "[WEB_SEARCH]",
            ],
            negative_signals=["Inventei que", "A vulnerabilidade CVE-2026-99999"],
            description="Verifica se o modelo admite honestamente ausência de dados em tempo real gerando a tag especial [UNKNOWN] e busca web.",
        ),
    ]

    # Injeta os Top 20 Idiomas Globais
    for code, name, query in TOP_20_GLOBAL_LANGUAGES:
        cases.append(
            BenchmarkCase(
                case_id=f"polyglot_lang_{code}",
                category="polyglot_top20",
                prompt=query,
                expected_signals=["<think>", "</think>"],
                negative_signals=["I cannot speak this language", "Não compreendi o idioma"],
                description=f"Validação de compreensão técnica e raciocínio em {name} ({code}).",
            )
        )

    return cases


def evaluate_response(case: BenchmarkCase, response: str, latency: float) -> BenchmarkResult:
    # 1. Extração da cadeia de pensamento (<think>...</think>)
    think_match = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
    think_content = think_match.group(1).strip() if think_match else ""
    think_tokens = len(think_content.split())
    total_tokens = len(response.split())
    output_tokens = max(0, total_tokens - think_tokens)
    think_ratio = think_tokens / max(1, total_tokens)

    # 2. Avaliação de Sinais Esperados e Violações
    matched_signals = [sig for sig in case.expected_signals if sig.lower() in response.lower()]
    violations = [neg for neg in case.negative_signals if neg.lower() in response.lower()]

    signal_score = len(matched_signals) / max(1, len(case.expected_signals))
    penalty = 0.5 * len(violations)
    final_score = max(0.0, min(1.0, signal_score - penalty))

    passed = final_score >= 0.70 and len(violations) == 0

    # 3. Classificação de Qualidade do Raciocínio
    if think_tokens >= 50 and "<think>" in response and "</think>" in response:
        reasoning_quality = "SUPERIOR (Raciocínio Estruturado Completo)"
    elif think_tokens > 0:
        reasoning_quality = "MODERADO (Raciocínio Conciso)"
    else:
        reasoning_quality = "DEFICIENTE (Ausência de <think>)"

    return BenchmarkResult(
        case_id=case.case_id,
        category=case.category,
        passed=passed,
        score=final_score,
        think_tokens=think_tokens,
        output_tokens=output_tokens,
        think_ratio=think_ratio,
        reasoning_quality=reasoning_quality,
        matched_signals=matched_signals,
        violations=violations,
        latency_seconds=latency,
    )


def run_benchmark_suite(inference_fn: Callable[[str], str]) -> Dict[str, Any]:
    suite = build_benchmark_test_suite()
    results: List[BenchmarkResult] = []

    print(f"=== [Iniciando Benchmark Mark Engineer - {len(suite)} Casos de Teste] ===")
    start_total = time.time()

    for idx, case in enumerate(suite, start=1):
        t0 = time.time()
        try:
            response = inference_fn(case.prompt)
        except Exception as exc:
            response = f"[ERRO]: {exc}"
        latency = time.time() - t0

        result = evaluate_response(case, response, latency)
        results.append(result)

        status_str = "APROVADO" if result.passed else "FALHA"
        print(f"[{idx:02d}/{len(suite):02d}] {case.case_id:<30} | {status_str} (Score: {result.score:.2f}) | Think: {result.think_tokens} tokens | Lat: {latency:.2f}s")

    elapsed_total = time.time() - start_total
    passed_count = sum(1 for r in results if r.passed)
    avg_score = sum(r.score for r in results) / len(results)
    avg_think_tokens = sum(r.think_tokens for r in results) / len(results)

    summary = {
        "timestamp": time.time(),
        "total_cases": len(results),
        "passed_cases": passed_count,
        "pass_rate": passed_count / len(results),
        "average_score": avg_score,
        "average_think_tokens": avg_think_tokens,
        "elapsed_seconds": elapsed_total,
        "results": [asdict(r) for r in results],
    }

    print("\n=== [Resumo Consolidado do Benchmark] ===")
    print(f"- Total de Casos: {len(results)}")
    print(f"- Aprovados: {passed_count}/{len(results)} ({summary['pass_rate']*100:.1f}%)")
    print(f"- Score Médio de Acurácia: {avg_score:.2f}/1.00")
    print(f"- Média de Tokens de Raciocínio (<think>): {avg_think_tokens:.1f} tokens/resposta")
    print(f"- Tempo Total de Execução: {elapsed_total:.2f}s")

    return summary


if __name__ == "__main__":
    def dummy_inference(prompt: str) -> str:
        return (
            "<think>\n"
            "Analisando requisitos e restrições arquiteturais.\n"
            "Garantindo ausência de deadlocks e verificando tratamento de erros.\n"
            "</think>\n"
            "Implementação de referência técnica."
        )

    summary = run_benchmark_suite(dummy_inference)
