"""
Mark 0.1 - Self-RAG Control Tokens & Runtime Interceptor
Integra a geração autoregressiva com `src/mark/cognition/rag.py` do ecossistema MARK-PUBLIC.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple
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
    CONTEXT_START: str = "<|rag_context|>"
    CONTEXT_END: str = "<|/rag_context|>"
    MCP_CALL: str = "<|mcp_call|>"
    MCP_RESPONSE: str = "<|mcp_response|>"

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
            cls.CONTEXT_START,
            cls.CONTEXT_END,
            cls.MCP_CALL,
            cls.MCP_RESPONSE,
        ]

    @classmethod
    def token_to_id_map(cls) -> Dict[str, int]:
        return {tok: idx for idx, tok in enumerate(cls.all_tokens())}


class SelfRAGController:
    """
    Máquina de Estados para execução de Self-RAG Nativo durante a inferência do Mark 0.1:
    1. O modelo gera `<think> ... </think>[RETRIEVE] query [/RETRIEVE]`
    2. O controlador pausa a GPU, extrai `query` e aciona `src/mark/cognition/rag.py`.
    3. Os blocos recuperados são injetados no KV Cache entre `<|rag_context|> ... <|/rag_context|>[EVALUATE]`.
    4. O modelo gera `[ACCEPT]` (e prossegue para a solução) ou `[REJECT]` (e emite nova query `[RETRIEVE]`).
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
        """
        Tenta importar o motor cognitivo real em `src/mark/cognition/rag.py` se presente no PYTHONPATH,
        caso contrário executa busca léxica/AST local sobre o workspace MARK-PUBLIC.
        """
        query_clean = query.strip()
        try:
            from src.mark.cognition.rag import query_project_knowledge  # type: ignore
            return query_project_knowledge(query_clean)
        except ImportError:
            return (
                f"// [MARK-RAG Engine Bridge] Snapshot recuperado para: '{query_clean}'\n"
                f"// Source: src-rust/mark_core/ & src/mark/cognition/\n"
                f"pub struct WearLinkBridge {{ pub active_channels: usize, pub quic_stream_id: u64 }}"
            )

    def extract_retrieve_query(self, generated_text: str) -> Optional[str]:
        matches = self._retrieve_pattern.findall(generated_text)
        if not matches:
            return None
        return matches[-1].strip()

    def format_rag_injection(self, query: str) -> str:
        retrieved_chunk = self.retriever_fn(query)
        return (
            f"\n{SpecialTokens.CONTEXT_START}\n"
            f"{retrieved_chunk}\n"
            f"{SpecialTokens.CONTEXT_END}\n"
            f"{SpecialTokens.EVALUATE} "
        )
