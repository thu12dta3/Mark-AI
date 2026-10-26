# 01. Mark Engineer 300M (302M) Capabilities Benchmark

This document details the test protocols, evaluation methodology, and empirical scores achieved by the **Mark Engineer 300M (302M parameters)** checkpoint across deliberate `<think>` reasoning, architectural request disambiguation, defensive cybersecurity (AppSec P0), and multilingual polyglot coverage.

---

## 1. Overview and Methodology

The benchmark is implemented in [`scripts/benchmark_engineer_capabilities.py`](../../scripts/benchmark_engineer_capabilities.py).

The evaluated model features the following architectural parameters:
- **Total Parameters:** 302,028,912 (~302M)
- **Transformer Layers:** 8 layers
- **Hidden Dimension:** 1024
- **Attention Heads:** 16 heads (MLA with low-rank KV compression)
- **Experts (MoE):** 16 routed experts + 1 shared expert
- **Vocabulary:** 4,377 tokens (custom BPE)

### Approval Criteria:
Each test case defines:
1. `expected_signals`: Structural substrings and code invariants that MUST be present in the output.
2. `negative_signals`: Prohibited patterns (hallucinations, dangerous simplifications, indiscriminate `.unwrap()` in Rust, un-sharded global locks in Go, etc.).
3. `think_ratio`: Minimum proportion of generated tokens allocated inside `<think>...</think>` tags to validate invariants before emitting the final code.

---

## 2. Structured Test Cases

### Case 1: Algorithmic Reasoning & Lock-Free Concurrency (C++20)
- **ID:** `reasoning_lock_free_queue`
- **Prompt:**
  ```text
  Implement a lock-free concurrent queue in C++20 using atomic pointers without the ABA problem.
  ```
- **Expected Invariants:** `<think>`, `std::atomic`, `compare_exchange_weak`, `hazard pointer` or `tagged pointer`.
- **Prohibited Signals:** `std::mutex`, `sleep`, `unwrap()`.
- **Mark 302M Result:** **Passed (Score: 95.0%)**. The model spent 58.2% of its tokens inside `<think>` analyzing the node reclamation hazard window before applying version-tagged pointers.

### Case 2: Vague Request Disambiguation into Concrete Architecture
- **ID:** `vague_intent_fast_cache`
- **Prompt:**
  ```text
  i need a fast in-memory cache for my go backend
  ```
- **Expected Invariants:** `sync.RWMutex`, `sharding`, `TTL`, `eviction`, `LRU`.
- **Prohibited Signals:** `map[string]any without mutex`, `global lock without sharding`.
- **Mark 302M Result:** **Passed (Score: 92.5%)**. Instead of generating a naive race-prone map, the model architected a 32-shard concurrent cache with background janitor cleanup and TTL eviction policies.

### Case 3: Deep Repository Analysis & TOCTOU Hardening
- **ID:** `repo_mark_public_tls_bridge`
- **Prompt:**
  ```text
  How does MARK prevent Time-of-Check to Time-of-Use (TOCTOU) and Symlink Hijacking attacks in code writing operations within engineer_agent.py?
  ```
- **Expected Invariants:** `resolve_authorized_workspace_root`, `is_symlink`, `fsync`, `os.replace`, `atomic`.
- **Prohibited Signals:** `chmod 777`, `shell=True`.
- **Mark 302M Result:** **Passed (Score: 98.0%)**. Documented the defense chain: `O_NOFOLLOW` / symlink verification, writing to a `.tmp` file on the same filesystem, and an atomic `os.replace()` guaranteed by POSIX kernel semantics.

### Case 4: Defensive Cybersecurity (AppSec P0)
- **ID:** `appsec_crypto_timing_attack`
- **Prompt:**
  ```text
  How to validate a session token or HMAC signature without exposing the backend to timing attacks?
  ```
- **Expected Invariants:** `hmac.compare_digest`, `constant time`, `subtle.ConstantTimeCompare`, `side-channel`.
- **Prohibited Signals:** Direct `==` or `===` string equality checks on cryptographic hashes.
- **Mark 302M Result:** **Passed (Score: 100.0%)**. Warned against byte-by-byte short-circuiting in standard equality operators and produced constant-time comparison routines.

---

## 3. Evaluation Across Top 20 Global Languages

Mark Engineer 300M was benchmarked using technical engineering prompts across the 20 most spoken languages globally, ensuring zero semantic degradation:

| Code | Language | Evaluated Engineering Prompt | Status |
| :---: | :--- | :--- | :---: |
| **en** | English | Zero-copy ring buffer in Rust | Passed |
| **zh-hans**| Simplified Mandarin | 高性能无锁循环缓冲区 (Go) | Passed |
| **hi** | Hindi | नॉन-ब्लॉकिंग कनेक्शन पूल (Python) | Passed |
| **es** | Spanish | Indices compuestos en base de datos | Passed |
| **fr** | French | Pipeline asynchrone avec gestion des erreurs | Passed |
| **ar** | Standard Arabic | نموذج الممثلين غير المتزامن باستخدام Tokio | Passed |
| **bn** | Bengali | সি++ এ মেমরি লিক ছাড়া কাস্টম অ্যালোকেটার | Passed |
| **pt** | Portuguese | Hardening P0 contra ataques de replay no MARK | Passed |
| **ru** | Russian | Атомарный обмен указателями в C++ | Passed |
| **ur** | Urdu | چینلز کا صحیح استعمال (Rust) | Passed |
| **id** | Indonesian | Arsitektur microservice tahan gangguan (Go) | Passed |
| **de** | German | Lock-free Queue mit atomaren Operationen (C++20) | Passed |
| **ja** | Japanese | Tokioを使用した非同期ワーカープール | Passed |
| **mr** | Marathi | कनकूरंट प्रोग्रामिंगमध्ये डेडलॉक निवारण | Passed |
| **te** | Telugu | కాంటెక్స్ట్ క్యాన్సిలేషన్ (Go) | Passed |
| **tr** | Turkish | Asenkron ag soketi yonetimi (Python) | Passed |
| **ta** | Tamil | பாதுகாப்பான நினைவக ஒதுக்கீடு (Rust) | Passed |
| **vi** | Vietnamese | Worker pool khong khoa (C#) | Passed |
| **ko** | Korean | C++20 코루틴 비동기 I/O 파이프라인 | Passed |
| **it** | Italian | Circuit breaker resiliente per gRPC (Go) | Passed |

---

## 4. Reasoning Metrics (<think>) and Latency

- **Mean Reasoning Allocation:** 54.8% of generated tokens were allocated inside `<think>...</think>` tags to validate invariants before emitting code.
- **Mean Latency:** 0.42 seconds Time-to-First-Token (TTFT) on local RTX 3060; continuous throughput of ~180 tokens/s.
- **Runtime Memory:** 348 MB VRAM under active inference with native IQ4_NL quantization.
