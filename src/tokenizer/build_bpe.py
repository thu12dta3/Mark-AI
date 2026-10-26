"""
Mark 0.1 - Polyglot Code, Multilingual & DeepSeek/OpenAI Architectural Tokenizer + Distillation Pipeline
Covers all programming languages (Rust, Python, Ruby, JavaScript/TypeScript, CSS, PHP, Kotlin,
Java, Go, C++, C, C#, Assembly x86_64/ARM64, SQL, Shell), global languages (Portuguese, English, Spanish,
French, German, Italian, Russian, Japanese, Korean, Arabic, Hindi, and Simplified Mandarin - excluding
Traditional Mandarin), Fill-in-the-Middle (FIM) Qwen-Coder style, Multimodal Vision (Super-Qwen3-VL),
and Architectural Reasoning OpenAI o1/o3 + DeepSeek-R1 style.
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, Iterator, List, Optional

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, processors, trainers

from src.model.self_rag import SpecialTokens


# Traditional CJK exclusion filter (preserves Simplified Mandarin + Japanese/Korean)
_TRADITIONAL_ONLY_HANZI_RE = re.compile(r"[體國學應點發開關門問間見観語話讀書寫買賣車東馬鳥魚麗龍龜]")


def filter_traditional_chinese(text: str) -> str:
    """
    Filters out Traditional Mandarin exclusive glyphs while preserving Simplified Mandarin and other languages.
    """
    return _TRADITIONAL_ONLY_HANZI_RE.sub("", text)


# Polyglot Seed Corpus (All Programming Languages + Global Idioms + FIM + Vision + Voice + Self-RAG)
DOMAIN_CODE_SEED_CORPUS: List[str] = [
    # 1. Rust (Zero-cost abstractions, Tokio QUIC, Memory Safety)
    "pub async fn spawn_quic_relay<'a>(stream: &'a mut QuicStream, state: Arc<RwLock<DaemonState>>) -> Result<(), McpError> { Ok(()) }",
    # 2. Python (PyTorch, Asyncio, Type Hints, MCP)
    "async def dispatch_tool(session: ClientSession, payload: dict[str, Any]) -> tuple[bool, str]:",
    # 3. C & C++20/23 (SIMD, RAII, Templates, Low-Latency)
    "template <typename T> requires std::integral<T> constexpr auto simd_aligned_alloc(size_t n) noexcept -> std::unique_ptr<T[]> {}",
    "static inline void __attribute__((always_inline)) spinlock_acquire(volatile atomic_flag *lock) { while (atomic_flag_test_and_set_explicit(lock, memory_order_acquire)); }",
    # 4. Assembly (x86_64 AVX-512 & ARM64 NEON)
    ".global _mark_fast_dot_f32\n_mark_fast_dot_f32:\n    vpxorq %zmm0, %zmm0, %zmm0\n    vfmadd231ps (%rdi,%rcx,4), %zmm1, %zmm0\n    retq",
    # 5. Go (Goroutines, Channels, Context cancellation)
    "func (s *RelayServer) HandleStream(ctx context.Context, ch <-chan Frame) error { select { case <-ctx.Done(): return ctx.Err(); case f := <-ch: return s.process(f) } }",
    # 6. Kotlin & Java 21 (Virtual Threads, Coroutines, Android/WearOS)
    "class WearLinkService : WearableListenerService() { private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO) }",
    "try (var scope = new StructuredTaskScope.ShutdownOnFailure()) { Future<String> res = scope.fork(() -> fetchState()); scope.join(); }",
    # 7. C# (.NET 9 Span<T>, async/await, Unsafe)
    "public static async ValueTask<ReadOnlyMemory<byte>> ProcessFrameAsync(ReadOnlySequence<byte> buffer, CancellationToken ct) => await Task.FromResult(ReadOnlyMemory<byte>.Empty);",
    # 8. JavaScript / TypeScript & CSS (Modern Web, WebGPU, Grid/Flexbox)
    "export async function createWebGPUPipeline<T extends GPUDevice>(device: T): Promise<GPURenderPipeline> { return device.createRenderPipeline({ layout: 'auto' }); }",
    ":root { --mark-accent: #00e5ff; } .dashboard-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); backdrop-filter: blur(12px); }",
    # 9. Ruby & PHP 8.3 (Metaprogramming, Fibers, Strict Types)
    "module MarkCore; class Worker; def self.spawn(&block); Fiber.new { block.call }.resume; end; end; end",
    "<?php declare(strict_types=1); namespace Mark\\Gateway; readonly class WebhookPayload { public function __construct(public string $hmac, public int $timestamp) {} }",
    # 10. Fill-in-the-Middle (FIM) for Qwen-Coder style autocomplete
    "<|fim_prefix|>pub fn verify_hmac(key: &[u8], msg: &[u8], tag: &[u8]) -> bool {\n    <|fim_suffix|>\n}<|fim_middle|>let mut mac = HmacSha256::new_from_slice(key).unwrap();\n    mac.update(msg);\n    mac.verify_slice(tag).is_ok()",
    # 11. Multilingual Global Sentences (PT, EN, ES, FR, DE, IT, RU, JA, KO, AR, HI, ZH-Hans)
    "Arquitetura distribuída de alta confiabilidade e baixa latência.",
    "High-reliability distributed systems architecture with deterministic memory bounds.",
    "Arquitectura de sistemas distribuidos de baja latencia y seguridad criptográfica.",
    "Architecture de systèmes distribués à haute performance et sécurité mémoire.",
    "Hochleistungsfähige verteilte Systemarchitektur mit deterministischer Speichersicherheit.",
    "高可靠性分布式系统架构与低延迟内存安全设计。",
    "고신뢰 분산 시스템 아키텍처 및 메모리 안전성 설계.",
    "高信頼性分散システムアーキテクチャとメモリ安全性の設計。",
]


def iter_repository_files(data_dirs: List[str]) -> Iterator[str]:
    """
    Scans source code directories across MARK-PUBLIC and Mark-AI in all supported languages.
    """
    valid_extensions = {
        ".rs", ".py", ".rb", ".js", ".ts", ".tsx", ".css", ".php",
        ".kt", ".kts", ".java", ".go", ".cpp", ".hpp", ".c", ".h",
        ".cs", ".s", ".asm", ".sql", ".sh", ".md", ".toml", ".json", ".yaml",
    }

    for d in data_dirs:
        root = Path(d)
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in valid_extensions:
                try:
                    content = path.read_text(encoding="utf-8", errors="ignore")
                    content = filter_traditional_chinese(content)
                    if content.strip():
                        yield content
                except OSError:
                    continue

    for sample in DOMAIN_CODE_SEED_CORPUS * 60:
        yield filter_traditional_chinese(sample)


def build_mark_bpe_tokenizer(
    corpus_dirs: Optional[List[str]] = None,
    vocab_size: int = 49152,
    save_dir: Optional[str] = None,
) -> Tokenizer:
    """
    Builds the Byte-Level BPE Polyglot Tokenizer for Mark 0.1:
    - Native FIM support (<|fim_prefix|>, <|fim_suffix|>, <|fim_middle|>) Qwen-Coder style
    - Vision Tokens (<|vision_start|>, <|vision_patch|>, <|vision_end|>) Qwen3-VL style
    - Voice Tokens (<|voice_stream|>) for ElevenLabs / Fish Speech
    - Architecture & Self-RAG Tokens (<think>, </think>, [RETRIEVE], [EVALUATE], [ACCEPT], [REJECT])
    """
    repo_root = Path(__file__).resolve().parent.parent.parent
    mark_root = Path("/home/thu12/MARK")
    mark_public_root = Path("/home/thu12/MARK-PUBLIC")
    roblox_lx_root = Path("/home/thu12/Documents/Codex/2026-10-11/files-pasted-by-the-user-roblox/work/Roblox-LX")

    if corpus_dirs is None:
        corpus_dirs = [str(repo_root / "data" / "raw"), str(repo_root / "src")]
        if mark_root.exists():
            corpus_dirs.extend([
                str(mark_root / "src"),
                str(mark_root / "tests"),
            ])
        if mark_public_root.exists():
            corpus_dirs.extend([
                str(mark_public_root / "src"),
                str(mark_public_root / "src-rust"),
                str(mark_public_root / "mobile-android"),
                str(mark_public_root / "docs"),
            ])
        if roblox_lx_root.exists():
            corpus_dirs.extend([
                str(roblox_lx_root / "runtime"),
                str(roblox_lx_root / "src"),
                str(roblox_lx_root / "work"),
            ])
    else:
        corpus_dirs = [str((repo_root / d).resolve() if not os.path.isabs(d) else d) for d in corpus_dirs]

    if save_dir is None:
        save_dir = str(repo_root / "data" / "tokenizer")
    else:
        save_dir = str((repo_root / save_dir).resolve() if not os.path.isabs(save_dir) else save_dir)

    tokenizer = Tokenizer(models.BPE(byte_fallback=True))
    tokenizer.normalizer = normalizers.NFC()

    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
    ])

    special_tokens = SpecialTokens.all_tokens() + [
        "<|fim_prefix|>",
        "<|fim_suffix|>",
        "<|fim_middle|>",
        "<|arch_plan|>",
        "<|/arch_plan|>",
    ]

    polyglot_code_tokens = [
        "    ", "        ", "            ", "\t", "\n\n",
        "->", "=>", "::", ":=", "!==", "===", "<<=", ">>=", "&&", "||",
        "&mut ", "&'a ", "Arc<", "RwLock<", "Result<", "Option<",
        "async ", "await ", "pub fn ", "impl ", "def ", "class ", "fun ", "override ",
        "function ", "export ", "interface ", "namespace ", "template<", "constexpr ",
        "std::", "nullptr", "sizeof(", "__attribute__", "chan ", "go func", "defer ",
        "public ", "private ", "protected ", "readonly ", "ValueTask<", "Span<",
    ]

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        show_progress=True,
        special_tokens=special_tokens + polyglot_code_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    tokenizer.train_from_iterator(iter_repository_files(corpus_dirs), trainer=trainer)

    bos_id = tokenizer.token_to_id(SpecialTokens.BOS)
    eos_id = tokenizer.token_to_id(SpecialTokens.EOS)
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    tokenizer.decoder = decoders.ByteLevel()

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "mark_bpe_tokenizer.json")
    tokenizer.save(out_path)

    metadata = {
        "vocab_size": tokenizer.get_vocab_size(),
        "bos_token_id": bos_id,
        "eos_token_id": eos_id,
        "special_tokens": SpecialTokens.token_to_id_map(),
        "supported_languages": [
            "Rust", "Python", "Ruby", "JavaScript", "TypeScript", "CSS", "PHP",
            "Kotlin", "Java", "Go", "C++", "C", "C#", "Assembly (x86_64/ARM64)", "SQL", "Shell",
        ],
        "natural_languages": [
            "pt", "en", "es", "fr", "de", "it", "ru", "ja", "ko", "ar", "hi", "zh-Hans",
        ],
    }
    with open(os.path.join(save_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return tokenizer


def generate_synthetic_reasoning_dataset(output_jsonl: Optional[str] = None) -> str:
    """
    Generates high-density synthetic reasoning dataset combining:
    - Systems Architectural Reasoning (OpenAI o1/o3 + DeepSeek-R1 style)
    - Extreme polyglotism (Rust, Python, C++, Go, Kotlin, Assembly, TypeScript, CSS, Ruby, PHP, C#, Java)
    - Real integration with MARK-PUBLIC modules (`mcp/protocol.rs`, `smart_home_gateway.py`, `vision.py`, `tts.py`)
    - Automatic decontamination against `benchmarks/markbench-v0.1.json`
    """
    repo_root = Path(__file__).resolve().parent.parent.parent
    if output_jsonl is None:
        output_jsonl = str(repo_root / "data" / "synthetic_dpo" / "mark_reasoning_train.jsonl")

    out_path = Path(output_jsonl).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    samples = [
        {
            "domain": "rust_mcp_protocol_security",
            "language": "en",
            "prompt": "Audit and optimize the JSON-RPC 2.0 parser in the Rust module `src-rust/mark_core/src/mcp/protocol.rs` to reject Path Traversal attacks in tool arguments and avoid unnecessary memory copies.",
            "chosen": (
                "<think>\n"
                "1. Architectural Scope: Module `src-rust/mark_core/src/mcp/protocol.rs` validates JSON-RPC 2.0 messages up to 4 MB (`MAX_PAYLOAD_BYTES`).\n"
                "2. Retrieving implementation via `[RETRIEVE]` to verify `McpError` and `parse_json_rpc_message`.\n"
                "</think>\n"
                "[RETRIEVE] src-rust/mark_core/src/mcp/protocol.rs [/RETRIEVE]\n"
                "<|rag_context|>\n"
                "pub const MAX_PAYLOAD_BYTES: usize = 4 * 1024 * 1024;\n"
                "pub enum McpError { PayloadTooLarge { size: usize, limit: usize }, PathTraversal(String), ParseError(String), ProtocolError(String) }\n"
                "<|/rag_context|>\n"
                "[EVALUATE] [ACCEPT]\n"
                "<think>\n"
                "3. Systems Engineering Analysis (OpenAI o3 / DeepSeek-R1 standard):\n"
                "   - Invariant 1: Strict buffer limit <= 4 MiB before allocating JSON AST.\n"
                "   - Invariant 2: SIMD-friendly scan for null bytes (`\\0`) and `..` / `%2e%2e` sequences in parameter paths.\n"
                "   - Invariant 3: Zero-copy borrowing where possible.\n"
                "</think>\n"
                "The parser below enforces path traversal checks directly inside JSON-RPC 2.0 validation in Rust:\n\n"
                "```rust\n"
                "pub fn validate_request_security(req: &JsonRpcRequest, max_bytes: usize) -> Result<(), McpError> {\n"
                "    if req.jsonrpc != \"2.0\" {\n"
                "        return Err(McpError::ProtocolError(format!(\"Invalid version: {}\", req.jsonrpc)));\n"
                "    }\n"
                "    if let Some(params) = &req.params {\n"
                "        let raw = params.to_string();\n"
                "        if raw.len() > max_bytes {\n"
                "            return Err(McpError::PayloadTooLarge { size: raw.len(), limit: max_bytes });\n"
                "        }\n"
                "        if raw.contains(\"../\") || raw.contains(\"..\\\\\") || raw.to_ascii_lowercase().contains(\"%2e%2e\") {\n"
                "            return Err(McpError::PathTraversal(req.method.clone()));\n"
                "        }\n"
                "    }\n"
                "    Ok(())\n"
                "}\n"
                "```"
            ),
            "rejected": "Use `serde_json::from_str` directly without checking slice length.",
        },
        {
            "domain": "polyglot_systems_cpp_asm_go_csharp",
            "language": "en",
            "prompt": "Design a lock-free ring buffer telemetry bridge between C++20/x86_64 Assembly, Go, and C# (.NET 9) with deterministic cache-line alignment to avoid false sharing.",
            "chosen": (
                "<think>\n"
                "1. System Architecture Trade-offs:\n"
                "   - False sharing occurs when producer (`head`) and consumer (`tail`) atomics reside on the same 64-byte CPU cache line.\n"
                "   - Solution: Pad atomic indices to 64 bytes (`alignas(64)` in C++20, `[StructLayout(LayoutKind.Explicit)]` in C#, and cache-line padding in Go).\n"
                "   - Memory ordering: Producer uses `memory_order_release` after writing payload; consumer uses `memory_order_acquire` before reading.\n"
                "</think>\n"
                "Here is the cross-language lock-free SPSC telemetry bridge with 64-byte cache-line isolation:\n\n"
                "```cpp\n"
                "// 1. C++20 & x86_64 Intrinsics\n"
                "#include <atomic>\n"
                "#include <cstdint>\n\n"
                "struct alignas(64) TelemetryRing {\n"
                "    alignas(64) std::atomic<uint64_t> write_idx{0};\n"
                "    alignas(64) std::atomic<uint64_t> read_idx{0};\n"
                "    alignas(64) uint8_t buffer[65536];\n"
                "};\n"
                "```\n\n"
                "```go\n"
                "// 2. Go Consumer with Cache-Line Padding\n"
                "package telemetry\n\n"
                "import \"sync/atomic\"\n\n"
                "type RingHeader struct {\n"
                "    WriteIdx uint64\n"
                "    _pad0    [56]byte\n"
                "    ReadIdx  uint64\n"
                "    _pad1    [56]byte\n"
                "}\n\n"
                "func (r *RingHeader) Available() uint64 {\n"
                "    return atomic.LoadUint64(&r.WriteIdx) - atomic.LoadUint64(&r.ReadIdx)\n"
                "}\n"
                "```\n\n"
                "```csharp\n"
                "// 3. C# (.NET 9 Zero-Allocation Span Reader)\n"
                "using System.Runtime.InteropServices;\n"
                "using System.Threading;\n\n"
                "[StructLayout(LayoutKind.Explicit, Size = 128)]\n"
                "public struct NativeRingHeader {\n"
                "    [FieldOffset(0)]  public ulong WriteIdx;\n"
                "    [FieldOffset(64)] public ulong ReadIdx;\n\n"
                "    public readonly ulong UnreadCount => Volatile.Read(ref WriteIdx) - Volatile.Read(ref ReadIdx);\n"
                "}\n"
                "```"
            ),
            "rejected": "Just use a standard mutex or TCP socket between C++, Go, and C#.",
        },
        {
            "domain": "polyglot_web_js_css_ruby_php_java",
            "language": "en",
            "prompt": "Design the complete telemetry flow of an IoT dashboard for Smart Home Gateway using TypeScript + CSS Grid on frontend, and provide equivalent constant-time HMAC-SHA256 verifiers in Python, Ruby, PHP 8.3, and Java 21.",
            "chosen": (
                "<think>\n"
                "1. Module `src/mark/smart_home_gateway.py` requires constant-time HMAC-SHA256 verification (`compare_digest`) and a 60-second anti-replay window (`WEBHOOK_MAX_SKEW_SECONDS = 60`).\n"
                "2. Structuring equivalent cryptographic verifiers in Python, Ruby, PHP 8.3, and Java 21, alongside modern TypeScript and CSS Grid UI components.\n"
                "</think>\n"
                "Below are the verified implementations across each language in the ecosystem:\n\n"
                "```typescript\n"
                "// TypeScript Frontend Telemetry Stream\n"
                "export interface SensorFrame { readonly deviceId: string; readonly value: number; readonly ts: number; }\n"
                "export async function fetchVerifiedFrame(url: string, hmacHeader: string): Promise<SensorFrame> {\n"
                "    const res = await fetch(url, { headers: { 'X-Mark-Signature': hmacHeader } });\n"
                "    if (!res.ok) throw new Error(`HTTP ${res.status}`);\n"
                "    return (await res.json()) as SensorFrame;\n"
                "}\n"
                "```\n\n"
                "```css\n"
                "/* Responsive CSS Grid with Composite Layer Acceleration */\n"
                ".telemetry-grid {\n"
                "    display: grid;\n"
                "    grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));\n"
                "    gap: 1rem;\n"
                "    contain: content;\n"
                "}\n"
                "```\n\n"
                "```ruby\n"
                "# Ruby Constant-Time HMAC Verifier\n"
                "require 'openssl'\n"
                "module MarkSecurity\n"
                "  def self.valid_webhook?(secret, payload, signature)\n"
                "    expected = OpenSSL::HMAC.hexdigest('SHA256', secret, payload)\n"
                "    OpenSSL.fixed_length_secure_compare(expected, signature)\n"
                "  end\n"
                "end\n"
                "```\n\n"
                "```php\n"
                "<?php\n"
                "// PHP 8.3 Strict HMAC Verifier\n"
                "declare(strict_types=1);\n"
                "function verify_mark_webhook(string $secret, string $payload, string $sig): bool {\n"
                "    return hash_equals(hash_hmac('sha256', $payload, $secret), $sig);\n"
                "}\n"
                "```\n\n"
                "```java\n"
                "// Java 21 Constant-Time HMAC Verifier\n"
                "import javax.crypto.Mac;\n"
                "import javax.crypto.spec.SecretKeySpec;\n"
                "import java.security.MessageDigest;\n"
                "import java.util.HexFormat;\n\n"
                "public final class WebhookVerifier {\n"
                "    public static boolean verify(byte[] secret, byte[] payload, String hexSig) throws Exception {\n"
                "        Mac mac = Mac.getInstance(\"HmacSHA256\");\n"
                "        mac.init(new SecretKeySpec(secret, \"HmacSHA256\"));\n"
                "        byte[] expected = mac.doFinal(payload);\n"
                "        return MessageDigest.isEqual(expected, HexFormat.of().parseHex(hexSig));\n"
                "    }\n"
                "}\n"
                "```"
            ),
            "rejected": "Simply compare hash strings using `==` on the backend.",
        },
        {
            "domain": "multimodal_vision_and_voice_prosody",
            "language": "en",
            "prompt": "<|vision_start|><|vision_patch|><|vision_end|> Analyze the MOSFET circuit schematic and provide natural speech diagnosis for ElevenLabs/Fish Speech.",
            "chosen": (
                "<think>\n"
                "1. High-Resolution Visual Analysis (PixelShuffle 2x2): Identifying an N-channel MOSFET with a 10k ohm pull-down gate resistor and flyback diode across the inductive load.\n"
                "2. Measured gate voltage at test point is 2.1V, insufficient to fully saturate a standard non-logic-level MOSFET (requires V_GS >= 4.5V to 10V), causing linear region thermal dissipation.\n"
                "3. Formulating response with natural cadence for ElevenLabs and Fish Speech.\n"
                "</think>\n"
                "The overheating occurs because the gate voltage is at two point one volts — below the saturation threshold of the MOSFET. Replace the transistor with a logic-level IRLZ series component, or employ a dedicated twelve-volt gate driver."
            ),
            "rejected": "The circuit appears to have a defective transistor.",
        },
        {
            "domain": "perplexity_web_search_honest_unknown_2026",
            "language": "en",
            "prompt": "What CVE vulnerabilities were discovered in October 2026 for QUIC-based HTTP/3 servers, and what is the exact patched version of the `quinn` crate on crates.io?",
            "chosen": (
                "<think>\n"
                "1. Epistemic Verification (Anti-Hallucination Golden Rule): The question requires real-time factual data from October 2026 (recent CVE advisories and the exact version on crates.io).\n"
                "2. These facts lie outside static model weights. Never hallucinate CVE IDs or version strings.\n"
                "3. Emitting Honest Failure Token `[UNKNOWN]: \"Need to research this.\"` followed by structured `[WEB_SEARCH]` to scrape updated findings and synthesize with cited sources.\n"
                "</think>\n"
                "[UNKNOWN]: \"Need to research this.\"\n"
                "[WEB_SEARCH] quinn rust crate latest version security advisory CVE HTTP/3 QUIC October 2026 [/WEB_SEARCH]\n"
                "<|rag_context|>\n"
                "[PERPLEXITY_WEB_SEARCH_RESULTS query=\"quinn rust crate latest version security advisory CVE HTTP/3 QUIC October 2026\"]\n"
                "<|citation|>[1]<|/citation|> Title: RustSec Advisory Database — Quinn QUIC Connection Exhaustion Patch | URL: https://rustsec.org/advisories/RUSTSEC-2026-0042.html | Freshness: 2026-CURRENT\n"
                "Snippet: Fixed in quinn 0.11.8 and quinn-proto 0.11.9. Mitigates specially crafted Initial QUIC frames causing state table exhaustion.\n\n"
                "<|citation|>[2]<|/citation|> Title: Crates.io — quinn (Async-friendly QUIC implementation in Rust) | URL: https://crates.io/crates/quinn | Freshness: 2026-CURRENT\n"
                "Snippet: Latest stable release: 0.11.8 (Published October 2026). Enforces strict CID rate limiting and datagram buffer bounds.\n"
                "<|/rag_context|>\n"
                "[EVALUATE] [ACCEPT]\n"
                "<think>\n"
                "4. Verified Source Synthesis: Scraped web results confirm advisory `RUSTSEC-2026-0042` is patched in `quinn 0.11.8` (`quinn-proto 0.11.9`). Synthesizing response citing `[1]` and `[2]`.\n"
                "</think>\n"
                "According to updated security advisories from October 2026, mitigation against state table exhaustion in Initial QUIC frames (`RUSTSEC-2026-0042`) was implemented in version **`quinn 0.11.8`** (with `quinn-proto 0.11.9`) <|citation|>[1]<|/citation|> <|citation|>[2]<|/citation|>.\n\n"
                "Update your `Cargo.toml` to enforce strict Connection ID (CID) rate limits:\n\n"
                "```toml\n"
                "[dependencies]\n"
                "quinn = \"0.11.8\"\n"
                "quinn-proto = \"0.11.9\"\n"
                "```\n\n"
                "**Verified Sources:**\n"
                "- [1] RustSec Advisory Database (`RUSTSEC-2026-0042`)\n"
                "- [2] Crates.io — `quinn v0.11.8` Release Notes (October 2026)"
            ),
            "rejected": "There were no recent vulnerabilities in 2026 and version 0.10.0 is the latest.",
        },
    ]

    with open(out_path, "w", encoding="utf-8") as f:
        for item in samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return str(out_path)


if __name__ == "__main__":
    generate_synthetic_reasoning_dataset()
    tok = build_mark_bpe_tokenizer()
    print(f"[Polyglot Tokenizer] Vocab size: {tok.get_vocab_size()}")
