"""
Mark 0.1 - Custom Byte-Level BPE Tokenizer Trainer & Synthetic Distillation Prep
Otimizado para Rust (`src-rust/mark_core/`), Python (`src/mark/`), Kotlin (WearOS/Android), Markdown e MCP.
"""

import json
import os
from pathlib import Path
from typing import Iterator, List, Optional

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, processors, trainers

from src.model.self_rag import SpecialTokens


# Padrões sintáticos de código que devem ser preservados ou priorizados no vocabulário BPE
DOMAIN_CODE_SEED_CORPUS: List[str] = [
    # Rust Ownership, Lifetimes, Async Tokio, QUIC & Memory Safety
    "pub async fn spawn_quic_relay<'a>(stream: &'a mut QuicStream, arc_state: Arc<RwLock<DaemonState>>) -> Result<(), MarkError> {",
    "unsafe impl Send for SharedMemoryRingBuffer {} // SAFETY: atomic fence Acquire/Release",
    "match rx.recv().await { Some(Frame::McpPayload(bytes)) => {}, None => return Err(MarkError::ChannelClosed) }",
    "#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]",
    # Python PyTorch, Asyncio & MCP Server Protocol
    "async def handle_mcp_request(session: ClientSession, tool_name: str, arguments: dict[str, Any]) -> CallToolResult:",
    "from src.mark.cognition.rag import SelfRAGRetriever, VectorIndex",
    # Kotlin Android / WearOS Bridge
    "class WearLinkService : WearableListenerService() { override fun onMessageReceived(event: MessageEvent) { coroutineScope.launch(Dispatchers.IO) { } } }",
    # Estrutura Nativa de Raciocínio e Self-RAG do Mark 0.1
    "<think>\nAnalisando as restrições de concorrência no módulo Rust e verificando risco de deadlock no RwLock antes de propor o patch.\n</think>\n[RETRIEVE] src-rust/mark_core/src/daemon.rs [/RETRIEVE]\n<|rag_context|>\npub struct DaemonState {}\n<|/rag_context|>\n[EVALUATE] [ACCEPT]\n",
]


def iter_repository_files(data_dirs: List[str]) -> Iterator[str]:
    """
    Varre diretórios de código-fonte (.rs, .py, .kt, .md, .toml, .json) sem estourar a RAM.
    """
    valid_extensions = {".rs", ".py", ".kt", ".kts", ".md", ".toml", ".json", ".yaml", ".sh"}
    yielded_any = False

    for d in data_dirs:
        root = Path(d)
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in valid_extensions:
                try:
                    content = path.read_text(encoding="utf-8", errors="ignore")
                    if content.strip():
                        yielded_any = True
                        yield content
                except OSError:
                    continue

    # Garante que os padrões sintáticos de Rust/Python/Kotlin/MCP e tags Self-RAG entrem nas merges BPE
    for sample in DOMAIN_CODE_SEED_CORPUS * 50:
        yield sample


def build_mark_bpe_tokenizer(
    corpus_dirs: Optional[List[str]] = None,
    vocab_size: int = 49152,
    save_dir: Optional[str] = None,
) -> Tokenizer:
    """
    Constrói e treina um tokenizador Byte-Level BPE:
    - Preserva indentação exata (4 espaços, 2 espaços, tabs) para Python/Rust.
    - Separa dígitos individualmente (0-9) para melhorar aritmética e endereçamento de memória.
    - Registra as tags <think>, </think>, [RETRIEVE], [EVALUATE], [ACCEPT], [REJECT] nos IDs 0..13.
    """
    repo_root = Path(__file__).resolve().parent.parent.parent
    if corpus_dirs is None:
        corpus_dirs = [str(repo_root / "data" / "raw"), str(repo_root / "src")]
    else:
        corpus_dirs = [str((repo_root / d).resolve() if not os.path.isabs(d) else d) for d in corpus_dirs]

    if save_dir is None:
        save_dir = str(repo_root / "data" / "tokenizer")
    else:
        save_dir = str((repo_root / save_dir).resolve() if not os.path.isabs(save_dir) else save_dir)

    tokenizer = Tokenizer(models.BPE(byte_fallback=True))

    # NFC Normalization sem destruir espaços ou símbolos de código
    tokenizer.normalizer = normalizers.NFC()

    # Pre-tokenization: separa dígitos individualmente + ByteLevel para preservar espaços/newlines
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Digits(individual_digits=True),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
    ])

    special_tokens = SpecialTokens.all_tokens()
    # Tokens compostos de indentação e operadores multi-caractere de Rust/Python/Kotlin
    code_structural_tokens = [
        "    ", "        ", "            ", "\n\n",
        "->", "=>", "::", "&mut ", "&'a ", "Arc<", "RwLock<", "Result<", "Option<",
        "async ", "await", "pub fn ", "impl ", "def ", "class ", "fun ", "override ",
    ]

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        show_progress=True,
        special_tokens=special_tokens + code_structural_tokens,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    tokenizer.train_from_iterator(iter_repository_files(corpus_dirs), trainer=trainer)

    # Post-processor para adicionar <|bos|> automaticamente se necessário
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
    }
    with open(os.path.join(save_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return tokenizer


def generate_synthetic_reasoning_dataset(output_jsonl: Optional[str] = None) -> str:
    """
    Pipeline de preparação de dataset (`dataset_prep`) para Destilação de Raciocínio e Self-RAG
    focado em Rust, Python, Kotlin, concorrência e protocolo MCP.
    Formato JSONL compatível com SFT + DPO (`prompt`, `chosen`, `rejected`).
    """
    if output_jsonl is None:
        repo_root = Path(__file__).resolve().parent.parent.parent
        output_jsonl = str(repo_root / "data" / "synthetic_dpo" / "mark_reasoning_train.jsonl")

    out_path = Path(output_jsonl).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    samples = [
        {
            "domain": "rust_concurrency_quic",
            "prompt": "Corrija o vazamento de descritores e possível deadlock no relay QUIC do daemon Rust quando um cliente WearOS desconecta abruptamente.",
            "chosen": (
                "<think>\n"
                "1. Problema reportado: vazamento de descritores e deadlock no relay QUIC após desconexão abrupta do WearOS.\n"
                "2. Preciso inspecionar a implementação atual de `relay.rs` ou `daemon.rs` antes de assumir a estrutura do lock.\n"
                "</think>\n"
                "[RETRIEVE] src-rust/mark_link/src/relay.rs [/RETRIEVE]\n"
                "<|rag_context|>\n"
                "pub async fn handle_wear_stream(state: Arc<Mutex<RelayState>>, stream: QuicStream) {\n"
                "    let mut guard = state.lock().await;\n"
                "    let frame = stream.read_frame().await.unwrap();\n"
                "    guard.register(frame);\n"
                "}\n"
                "<|/rag_context|>\n"
                "[EVALUATE] [ACCEPT]\n"
                "<think>\n"
                "3. A falha crítica está evidente: o `MutexGuard` é mantido aberto através do ponto de suspensão `.await` de I/O de rede (`stream.read_frame().await`), além do `.unwrap()` que causa panic na task caso o socket feche.\n"
                "4. Solução: Aplicar timeout no I/O fora da seção crítica do lock, tratar o `Result` com propagação segura e adquirir o lock apenas para mutação em memória O(1).\n"
                "</think>\n"
                "```rust\n"
                "use std::time::Duration;\n"
                "use tokio::time::timeout;\n\n"
                "pub async fn handle_wear_stream(\n"
                "    state: Arc< tokio::sync::RwLock<RelayState> >,\n"
                "    mut stream: QuicStream,\n"
                ") -> Result<(), RelayError> {\n"
                "    // 1. I/O assíncrono com timeout estrito FORA do lock\n"
                "    let frame = match timeout(Duration::from_secs(5), stream.read_frame()).await {\n"
                "        Ok(Ok(Some(f))) => f,\n"
                "        Ok(Ok(None)) | Err(_) => {\n"
                "            stream.close().await.ok();\n"
                "            return Err(RelayError::PeerDisconnected);\n"
                "        }\n"
                "        Ok(Err(e)) => return Err(RelayError::Io(e)),\n"
                "    };\n\n"
                "    // 2. Seção crítica curta e determinística sem .await interno\n"
                "    {\n"
                "        let mut guard = state.write().await;\n"
                "        guard.register(frame);\n"
                "    }\n"
                "    Ok(())\n"
                "}\n"
                "```"
            ),
            "rejected": (
                "Para corrigir o problema, basta trocar `unwrap()` por `expect()` no seu código Rust:\n"
                "```rust\n"
                "let frame = stream.read_frame().await.expect(\"Erro na leitura\");\n"
                "```"
            ),
        },
        {
            "domain": "kotlin_wearos_memory",
            "prompt": "O serviço WearLinkService.kt está causando OutOfMemoryError no relógio após 2 horas de sincronização de sensores com o motor Python.",
            "chosen": (
                "<think>\n"
                "1. Sintoma: OOM em dispositivo WearOS após 2h no `WearLinkService.kt`.\n"
                "2. Em Kotlin/Android, as causas mais comuns em WearableListenerService são: `CoroutineScope` sem cancelamento em `onDestroy()`, ou acumulação de buffers de sensores sem backpressure.\n"
                "3. Vou buscar o código-fonte de `WearLinkService.kt` no repositório.\n"
                "</think>\n"
                "[RETRIEVE] WearLinkService.kt [/RETRIEVE]\n"
                "<|rag_context|>\n"
                "class WearLinkService : WearableListenerService() {\n"
                "    private val pendingPayloads = mutableListOf<ByteArray>()\n"
                "    override fun onMessageReceived(event: MessageEvent) {\n"
                "        GlobalScope.launch { pendingPayloads.add(event.data) }\n"
                "    }\n"
                "}\n"
                "<|/rag_context|>\n"
                "[EVALUATE] [ACCEPT]\n"
                "<think>\n"
                "4. Encontrado: uso de `GlobalScope.launch` (vaza corrotinas) e `pendingPayloads` crescendo indefinidamente sem limite ou drenagem via `Channel(capacity = CONFLATED)`.\n"
                "5. Vou substituir por `SupervisorJob() + Dispatchers.IO` atrelado ao ciclo de vida e `Channel` limitado com `BufferOverflow.DROP_OLDEST`.\n"
                "</think>\n"
                "```kotlin\n"
                "class WearLinkService : WearableListenerService() {\n"
                "    private val serviceJob = SupervisorJob()\n"
                "    private val serviceScope = CoroutineScope(Dispatchers.IO + serviceJob)\n"
                "    private val payloadChannel = Channel<ByteArray>(\n"
                "        capacity = 64,\n"
                "        onBufferOverflow = BufferOverflow.DROP_OLDEST\n"
                "    )\n\n"
                "    override fun onMessageReceived(event: MessageEvent) {\n"
                "        payloadChannel.trySend(event.data)\n"
                "    }\n\n"
                "    override fun onDestroy() {\n"
                "        payloadChannel.close()\n"
                "        serviceJob.cancel()\n"
                "        super.onDestroy()\n"
                "    }\n"
                "}\n"
                "```"
            ),
            "rejected": (
                "Aumente a heap size no AndroidManifest.xml usando `android:largeHeap=\"true\"` para evitar o OutOfMemoryError no WearOS."
            ),
        },
    ]

    with open(output_jsonl, "w", encoding="utf-8") as f:
        for item in samples:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return output_jsonl


if __name__ == "__main__":
    generate_synthetic_reasoning_dataset()
    tok = build_mark_bpe_tokenizer()
    sample = "<think> Analisando ponte Rust/MCP </think> [RETRIEVE] src-rust/mark_core/src/daemon.rs [/RETRIEVE]"
    encoded = tok.encode(sample)
    print(f"[Tokenizer] Vocab size: {tok.get_vocab_size()}")
    print(f"[Tokenizer] Sample tokens: {encoded.tokens}")
    print(f"[Tokenizer] Sample IDs: {encoded.ids}")
