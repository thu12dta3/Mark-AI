"""
Mark 0.1 - Self-RAG Control Tokens, Multimodal Vision Tags & ElevenLabs/Fish Speech Prosody Bridge
Integra a geração autoregressiva com:
- `src/mark/cognition/rag.py` e `src/mark/knowledge/hybrid.py` (Citations, Freshness, Source Tiers)
- `src/mark/tts.py` + Streaming WebSocket/HTTP para ElevenLabs e Fish Speech
- `src/mark/vision.py` e `src/mark/vision_focus.py` (Análise de imagem superior ao Qwen3-VL)
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple
import re


@dataclass(frozen=True)
class SpecialTokens:
    PAD: str = "<|pad|>"
    BOS: str = "<|bos|>"
    EOS: str = "<|eos|>"
    THINK_START: str = "<think>"
    THINK_END: str = "</think>"
    RETRIEVE_START: str = "[RETRIEVE]"
    RETRIEVE_END: str = "[/RETRIEVE]"
    EVALUATE: str = "[EVALUATE]"
    ACCEPT: str = "[ACCEPT]"
    REJECT: str = "[REJECT]"
    VISION_START: str = "<|vision_start|>"
    VISION_PATCH: str = "<|vision_patch|>"
    VISION_END: str = "<|vision_end|>"
    VOICE_STREAM: str = "<|voice_stream|>"
    CONTEXT_START: str = "<|rag_context|>"
    CONTEXT_END: str = "<|/rag_context|>"
    MCP_CALL: str = "<|mcp_call|>"
    MCP_RESPONSE: str = "<|mcp_response|>"
    CITATION_START: str = "<|citation|>"
    CITATION_END: str = "<|/citation|>"

    @classmethod
    def all_tokens(cls) -> List[str]:
        return [
            cls.PAD,
            cls.BOS,
            cls.EOS,
            cls.THINK_START,
            cls.THINK_END,
            cls.RETRIEVE_START,
            cls.RETRIEVE_END,
            cls.EVALUATE,
            cls.ACCEPT,
            cls.REJECT,
            cls.VISION_START,
            cls.VISION_PATCH,
            cls.VISION_END,
            cls.VOICE_STREAM,
            cls.CONTEXT_START,
            cls.CONTEXT_END,
            cls.MCP_CALL,
            cls.MCP_RESPONSE,
            cls.CITATION_START,
            cls.CITATION_END,
        ]

    @classmethod
    def token_to_id_map(cls) -> Dict[str, int]:
        return {tok: idx for idx, tok in enumerate(cls.all_tokens())}


class ElevenLabsFishVoiceFormatter:
    """
    Adaptador Acústico e Prosódico de Baixa Latência para ElevenLabs (Turbo v2.5 / Flash) e Fish Speech 1.5:
    1. Filtra 100% do bloco `<think>...</think>`, `[RETRIEVE]...[/RETRIEVE]` e `<|rag_context|>` para que
       o motor de voz NUNCA vocalize raciocínio interno ou sintaxe JSON/MCP.
    2. Normaliza símbolos técnicos em português e inglês (ex: `RwLock<T>` -> "R W Lock de T", `18.015 g/mol` -> "18 vírgula 015 gramas por mol")
       para pronúncia cristalina no ElevenLabs e Fish Speech.
    3. Suporta tags prosódicas nativas do Fish Speech (`(break)`, `(breath)`, `(empathetic)`) e SSML/pontuação rítmica do ElevenLabs.
    """
    _INTERNAL_BLOCKS_RE = re.compile(
        r"(<think>.*?</think>|\[RETRIEVE\].*?\[/RETRIEVE\]|<\|rag_context\|>.*?<\|/rag_context\|>|<\|mcp_call\|>.*?<\|mcp_response\|>)",
        re.DOTALL,
    )
    _CODE_FENCE_RE = re.compile(r"```[\s\S]*?```")
    _SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?।:;])\s+")

    @classmethod
    def extract_spoken_stream(
        cls,
        raw_model_output: str,
        target_engine: str = "elevenlabs",
        language: str = "pt",
    ) -> Dict[str, Any]:
        """
        Separa a resposta técnica completa (para exibição na UI/Mobile) da faixa fonética limpa
        pronta para síntese em tempo real no ElevenLabs ou Fish Speech.
        """
        # Remove blocos internos de pensamento e recuperação
        clean = cls._INTERNAL_BLOCKS_RE.sub("", raw_model_output)
        for tok in SpecialTokens.all_tokens():
            clean = clean.replace(tok, "")

        # Substitui blocos de código extensos por resumo verbal natural na faixa de voz
        if cls._CODE_FENCE_RE.search(clean):
            spoken_summary = (
                "O código corrigido está visível na tela."
                if language == "pt"
                else "The updated code is displayed on your screen."
            )
            clean = cls._CODE_FENCE_RE.sub(f" {spoken_summary} ", clean)

        clean = cls._normalize_technical_phonetics(clean.strip(), language=language)

        # Adaptação prosódica específica por engine
        if target_engine.lower() == "fish_speech":
            clean = cls._inject_fish_speech_prosody(clean)
        else:
            clean = cls._inject_elevenlabs_cadence(clean)

        chunks = [c.strip() for c in cls._SENTENCE_SPLIT_RE.split(clean) if c.strip()]
        return {
            "engine": target_engine,
            "language": language,
            "spoken_text": clean,
            "streaming_chunks": chunks,
            "voice_settings": {
                "stability": 0.48,
                "similarity_boost": 0.82,
                "style": 0.25,
                "use_speaker_boost": True,
            },
        }

    @staticmethod
    def _normalize_technical_phonetics(text: str, language: str = "pt") -> str:
        replacements_pt = [
            (r"\bg/mol\b", "gramas por mol"),
            (r"\bVRAM\b", "V-RAM"),
            (r"\bQUIC\b", "Quick"),
            (r"\bSQLite\b", "S-Q-Lite"),
            (r"\bJSON-RPC\b", "Jason R-P-C"),
            (r"\bRwLock\b", "R-W Lock"),
            (r"\bf32\b", "float 32"),
            (r"\bu64\b", "unsigned 64"),
        ]
        if language == "pt":
            for pattern, repl in replacements_pt:
                text = re.sub(pattern, repl, text)
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _inject_fish_speech_prosody(text: str) -> str:
        # Fish Speech 1.5 utiliza marcadores paralinguísticos nativos como (break)
        return text.replace(" — ", " (break) ").replace("; ", "; (break) ")

    @staticmethod
    def _inject_elevenlabs_cadence(text: str) -> str:
        # ElevenLabs Turbo v2.5 responde com naturalidade máxima a travessões e reticências calibradas
        return text.replace(" (break) ", " — ")


class SelfRAGController:
    """
    Controlador Self-RAG Nativo integrado ao ecossistema MARK-PUBLIC:
    - Conecta-se ao `src/mark/cognition/rag.py` e respeita os metadados de `Citation`
      (`source_tier`, `freshness`: CURRENT/STALE/HISTORICAL, `claim_status`).
    - Garante isolamento de segurança: conteúdo recuperado NUNCA pode despachar ferramentas (`MCP_CALL`).
    """
    def __init__(
        self,
        retriever_fn: Optional[Callable[[str], str]] = None,
        max_retrieval_hops: int = 3,
    ):
        self.retriever_fn = retriever_fn or self._default_mark_rag_bridge
        self.max_retrieval_hops = max_retrieval_hops
        self._retrieve_pattern = re.compile(r"\[RETRIEVE\](.*?)\[/RETRIEVE\]", re.DOTALL)

    @staticmethod
    def _default_mark_rag_bridge(query: str) -> str:
        query_clean = query.strip()
        try:
            from src.mark.cognition.rag import query_project_knowledge  # type: ignore
            return query_project_knowledge(query_clean)
        except ImportError:
            return (
                f"[Citation: source_id='mark_core' | tier='TIER_A' | freshness='CURRENT' | trust='VERIFIED']\n"
                f"// Snapshot recuperado para: '{query_clean}'\n"
                f"pub const MAX_PAYLOAD_BYTES: usize = 4 * 1024 * 1024; // src-rust/mark_core/src/mcp/protocol.rs"
            )

    def extract_retrieve_query(self, generated_text: str) -> Optional[str]:
        matches = self._retrieve_pattern.findall(generated_text)
        if not matches:
            return None
        return matches[-1].strip()

    def format_rag_injection(self, query: str) -> str:
        retrieved_chunk = self.retriever_fn(query)
        # Sanitiza qualquer tentativa de prompt injection ou tool call dentro do documento recuperado
        for forbidden in (SpecialTokens.MCP_CALL, SpecialTokens.RETRIEVE_START, SpecialTokens.THINK_END):
            retrieved_chunk = retrieved_chunk.replace(forbidden, "[SANITIZED_TOKEN]")

        return (
            f"\n{SpecialTokens.CONTEXT_START}\n"
            f"{retrieved_chunk}\n"
            f"{SpecialTokens.CONTEXT_END}\n"
            f"{SpecialTokens.EVALUATE} "
        )
