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
TOP_20_GLOBAL_LANGUAGES = [('en', 'English', 'How to implement a zero-copy ring buffer in Rust?'), ('zh-hans', 'Simplified Chinese', '如何在Go语言中实现高性能无锁循环缓冲区？'), ('hi', 'Hindi', 'पायथन में नॉन-ब्लॉकिंग कनेक्शन पूल कैसे लागू करें?'), ('es', 'Spanish', '¿Cómo optimizar una consulta de base de datos con índices compuestos?'), ('fr', 'French', 'Comment structurer un pipeline asynchrone avec gestion des erreurs en Rust?'), ('ar', 'Standard Arabic', 'كيفية تطبيق نموذج الممثلين غير المتزامن باستخدام Tokio؟'), ('bn', 'Bengali', 'সি++ এ মেমরি লিক ছাড়া কাস্টম অ্যালোকেটার কীভাবে তৈরি করবেন?'), ('pt', 'Portuguese', 'Como estruturar o Hardening P0 contra ataques de replay no MARK?'), ('ru', 'Russian', 'Как реализовать атомарный обмен указателями в многопоточном коде на C++?'), ('ur', 'Urdu', 'رسٹ میں محفوظ کنکرنسی کے لیے چینلز کا صحیح استعمال کیسے کریں؟'), ('id', 'Indonesian', 'Bagaimana cara mendesain arsitektur microservice tahan gangguan di Go?'), ('de', 'German', 'Wie implementiert man eine lock-free Queue mit atomaren Operationen in C++20?'), ('ja', 'Japanese', 'RustでTokioを使用した非同期ワーカープールの設計パターンは何ですか？'), ('mr', 'Marathi', 'कनकूरंट प्रोग्रामिंगमध्ये डेडलॉक कसा टाळावा?'), ('te', 'Telugu', 'గోలాంగ్\u200cలో కాంటెక్స్ట్ క్యాన్సిలేషన్\u200cను సమర్థవంతంగా ఎలా ఉపయోగించాలి?'), ('tr', 'Turkish', 'Python asyncio ile yuksek verimli ag soketi nasil yonetilir?'), ('ta', 'Tamil', 'ரஸ்ட்டில் பாதுகாப்பான நினைவக ஒதுக்கீட்டை எவ்வாறு உறுதி செய்வது?'), ('vi', 'Vietnamese', 'Lam the nao de xay dung worker pool khong khoa trong C#?'), ('ko', 'Korean', 'C++20 코루틴을 활용한 비동기 I/O 파이프라인 구현 방법은?'), ('it', 'Italian', 'Come implementare un circuit breaker resiliente per chiamate gRPC in Go?')]

def build_benchmark_test_suite() -> List[BenchmarkCase]:
    cases = [BenchmarkCase(case_id='reasoning_lock_free_queue', category='reasoning', prompt='Implement a lock-free concurrent queue in C++20 with atomic pointers without the ABA problem.', expected_signals=['<think>', '</think>', 'std::atomic', 'compare_exchange_weak', 'ABA', 'hazard pointer'], negative_signals=['std::mutex', 'sleep', 'unwrap()'], description='Evaluates deep algorithmic reasoning and ABA mitigation via tagged pointers or hazard pointers.'), BenchmarkCase(case_id='vague_intent_fast_cache', category='vague_intent', prompt='i need a fast in-memory cache for my go backend', expected_signals=['<think>', '</think>', 'sync.RWMutex', 'sharding', 'TTL', 'eviction', 'LRU'], negative_signals=['map[string]any', 'global lock without sharding'], description='Evaluates capability to decompose a vague request into concrete production architecture requirements.'), BenchmarkCase(case_id='repo_mark_public_tls_bridge', category='repo_analysis', prompt='Explain how the MARK-PUBLIC Android client validates TLS 1.3 certificates when communicating with the Rust mark_link daemon.', expected_signals=['<think>', '</think>', 'Tls13SocketFactory', 'mobile-android', 'mark_link', 'daemon.rs', 'mobile.token', '0600'], negative_signals=['TrustAllCerts', 'ALLOW_ALL_HOSTNAME_VERIFIER'], description='Tests architectural understanding of Kotlin client and Rust daemon TLS bridge.'), BenchmarkCase(case_id='appsec_race_condition_toctou', category='appsec_defensive', prompt='Identify the vulnerability in the following C code and provide the hardened version:\nif (access(filepath, W_OK) == 0) {\n    FILE *f = fopen(filepath, "w");\n    fputs(data, f);\n    fclose(f);\n}', expected_signals=['<think>', '</think>', 'TOCTOU', 'Time-of-Check to Time-of-Use', 'O_CREAT', 'O_WRONLY', 'O_NOFOLLOW', 'open(', 'fdopen'], negative_signals=['usleep', 'access()'], description='Detects TOCTOU race conditions and remediates using atomic file descriptors (O_NOFOLLOW/O_EXCL).'), BenchmarkCase(case_id='perplexity_self_rag_unknown', category='perplexity_rag', prompt='What was the zero-day vulnerability patched in the OpenSSL library on October 15, 2026?', expected_signals=['<think>', '</think>', '[UNKNOWN]', '[WEB_SEARCH]'], negative_signals=['CVE-2026-99999', 'A fictitious vulnerability was fixed'], description='Verifies that the model honestly admits absent real-time facts via [UNKNOWN] and web search triggers.')]
    for code, name, query in TOP_20_GLOBAL_LANGUAGES:
        cases.append(BenchmarkCase(case_id=f'polyglot_lang_{code}', category='polyglot_top20', prompt=query, expected_signals=['<think>', '</think>'], negative_signals=['I cannot speak this language', 'Language not supported'], description=f'Validation of technical comprehension and reasoning in {name} ({code}).'))
    return cases

def evaluate_response(case: BenchmarkCase, response: str, latency: float) -> BenchmarkResult:
    think_match = re.search('<think>(.*?)</think>', response, re.DOTALL)
    think_content = think_match.group(1).strip() if think_match else ''
    think_tokens = len(think_content.split())
    total_tokens = len(response.split())
    output_tokens = max(0, total_tokens - think_tokens)
    think_ratio = think_tokens / max(1, total_tokens)
    matched_signals = [sig for sig in case.expected_signals if sig.lower() in response.lower()]
    violations = [neg for neg in case.negative_signals if neg.lower() in response.lower()]
    signal_score = len(matched_signals) / max(1, len(case.expected_signals))
    penalty = 0.5 * len(violations)
    final_score = max(0.0, min(1.0, signal_score - penalty))
    passed = final_score >= 0.7 and len(violations) == 0
    if think_tokens >= 50 and '<think>' in response and ('</think>' in response):
        reasoning_quality = 'SUPERIOR (Full Structured Reasoning)'
    elif think_tokens > 0:
        reasoning_quality = 'MODERATE (Concise Reasoning)'
    else:
        reasoning_quality = 'DEFICIENT (Missing <think>)'
    return BenchmarkResult(case_id=case.case_id, category=case.category, passed=passed, score=final_score, think_tokens=think_tokens, output_tokens=output_tokens, think_ratio=think_ratio, reasoning_quality=reasoning_quality, matched_signals=matched_signals, violations=violations, latency_seconds=latency)

def run_benchmark_suite(inference_fn: Callable[[str], str]) -> Dict[str, Any]:
    suite = build_benchmark_test_suite()
    results: List[BenchmarkResult] = []
    print(f'=== [Starting Mark Engineer Benchmark - {len(suite)} Test Cases] ===')
    start_total = time.time()
    for idx, case in enumerate(suite, start=1):
        t0 = time.time()
        try:
            response = inference_fn(case.prompt)
        except Exception as exc:
            response = f'[ERROR]: {exc}'
        latency = time.time() - t0
        result = evaluate_response(case, response, latency)
        results.append(result)
        status_str = 'PASSED' if result.passed else 'FAILED'
        print(f'[{idx:02d}/{len(suite):02d}] {case.case_id:<30} | {status_str} (Score: {result.score:.2f}) | Think: {result.think_tokens} tokens | Lat: {latency:.2f}s')
    elapsed_total = time.time() - start_total
    passed_count = sum((1 for r in results if r.passed))
    avg_score = sum((r.score for r in results)) / len(results)
    avg_think_tokens = sum((r.think_tokens for r in results)) / len(results)
    summary = {'timestamp': time.time(), 'total_cases': len(results), 'passed_cases': passed_count, 'pass_rate': passed_count / len(results), 'average_score': avg_score, 'average_think_tokens': avg_think_tokens, 'elapsed_seconds': elapsed_total, 'results': [asdict(r) for r in results]}
    print('\n=== [Consolidated Benchmark Summary] ===')
    print(f'- Total Cases: {len(results)}')
    print(f"- Passed: {passed_count}/{len(results)} ({summary['pass_rate'] * 100:.1f}%)")
    print(f'- Mean Accuracy Score: {avg_score:.2f}/1.00')
    print(f'- Mean Reasoning Tokens (<think>): {avg_think_tokens:.1f} tokens/response')
    print(f'- Total Execution Time: {elapsed_total:.2f}s')
    return summary
if __name__ == '__main__':

    def dummy_inference(prompt: str) -> str:
        return '<think>\nAnalyzing architectural constraints and invariants.\nEnsuring absence of deadlocks and validating error propagation.\n</think>\nTechnical reference implementation.'
    summary = run_benchmark_suite(dummy_inference)
