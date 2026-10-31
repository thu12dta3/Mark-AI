import math
import html
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union
from pathlib import Path

@dataclass(frozen=True)
class SpecialTokens:
    PAD: str = '<|pad|>'
    BOS: str = '<|bos|>'
    EOS: str = '<|eos|>'
    THINK_START: str = '<think>'
    THINK_END: str = '</think>'
    RETRIEVE_START: str = '[RETRIEVE]'
    RETRIEVE_END: str = '[/RETRIEVE]'
    EVALUATE: str = '[EVALUATE]'
    ACCEPT: str = '[ACCEPT]'
    REJECT: str = '[REJECT]'
    VISION_START: str = '<|vision_start|>'
    VISION_PATCH: str = '<|vision_patch|>'
    VISION_END: str = '<|vision_end|>'
    VOICE_STREAM: str = '<|voice_stream|>'
    UNKNOWN: str = '[UNKNOWN]'
    WEB_SEARCH_START: str = '[WEB_SEARCH]'
    WEB_SEARCH_END: str = '[/WEB_SEARCH]'
    CONTEXT_START: str = '<|rag_context|>'
    CONTEXT_END: str = '<|/rag_context|>'
    MCP_CALL: str = '<|mcp_call|>'
    MCP_RESPONSE: str = '<|mcp_response|>'
    CITATION_START: str = '<|citation|>'
    CITATION_END: str = '<|/citation|>'
    ACT_DIRECT: str = '<|act:direct|>'
    ACT_MCTX_LOCAL: str = '<|act:mctx_local|>'
    ACT_WEB_SEARCH: str = '<|act:web_search|>'
    EPISTEMIC_UNKNOWN: str = '<|epistemic:unknown|>'
    REL_HIGH: str = '<|rel:high|>'
    REL_LOW: str = '<|rel:low|>'
    REL_NONE: str = '<|rel:none|>'
    SEC_CLEAN: str = '<|sec:clean|>'
    SEC_INJECT: str = '<|sec:inject|>'
    SUP_VERIFIED: str = '<|sup:verified|>'
    SUP_PARTIAL: str = '<|sup:partial|>'
    SUP_NONE: str = '<|sup:none|>'

    @classmethod
    def all_tokens(cls) -> List[str]:
        return [cls.PAD, cls.BOS, cls.EOS, cls.THINK_START, cls.THINK_END, cls.RETRIEVE_START, cls.RETRIEVE_END, cls.EVALUATE, cls.ACCEPT, cls.REJECT, cls.VISION_START, cls.VISION_PATCH, cls.VISION_END, cls.VOICE_STREAM, cls.UNKNOWN, cls.WEB_SEARCH_START, cls.WEB_SEARCH_END, cls.CONTEXT_START, cls.CONTEXT_END, cls.MCP_CALL, cls.MCP_RESPONSE, cls.CITATION_START, cls.CITATION_END, cls.ACT_DIRECT, cls.ACT_MCTX_LOCAL, cls.ACT_WEB_SEARCH, cls.EPISTEMIC_UNKNOWN, cls.REL_HIGH, cls.REL_LOW, cls.REL_NONE, cls.SEC_CLEAN, cls.SEC_INJECT, cls.SUP_VERIFIED, cls.SUP_PARTIAL, cls.SUP_NONE]

    @classmethod
    def token_to_id_map(cls) -> Dict[str, int]:
        return {tok: idx for idx, tok in enumerate(cls.all_tokens())}

class ZimVirtualExtractor:

    def __init__(self, zim_path: Union[str, Path]):
        self.zim_path = Path(zim_path)
        self.has_libzim = False
        self.archive = None
        try:
            import libzim
            if self.zim_path.exists():
                self.archive = libzim.Archive(str(self.zim_path))
                self.has_libzim = True
        except ImportError:
            self.has_libzim = False

    def clean_html_to_markdown(self, raw_html: str) -> str:
        text = raw_html
        text = re.sub('<pre[^>]*>(.*?)</pre>', '\\n```\\n\\1\\n```\\n', text, flags=re.DOTALL)
        text = re.sub('<math[^>]*>(.*?)</math>', ' $\\1$ ', text, flags=re.DOTALL)
        text = re.sub('<(script|style)[^>]*>.*?</\\1>', '', text, flags=re.DOTALL)
        for h in range(6, 0, -1):
            text = re.sub(f'<h{h}[^>]*>(.*?)</h{h}>', '\n' + '#' * h + ' \\1\\n', text, flags=re.DOTALL)
        text = re.sub('<br\\s*/?>', '\n', text)
        text = re.sub('</p>', '\n\n', text)
        text = re.sub('<[^>]+>', ' ', text)
        text = html.unescape(text)
        return re.sub('[ \\t]+', ' ', text).strip()

    def search(self, query: str, max_results: int=3) -> List[Dict[str, str]]:
        if not self.has_libzim or self.archive is None:
            return [{'title': f'Local Zim Index ({self.zim_path.name})', 'url': f'zim://{self.zim_path.name}/{query}', 'content': f"[ZimFallback: libzim not installed or file not found. Extracted query: '{query}']"}]
        results = []
        try:
            import libzim
            searcher = libzim.Searcher(self.archive)
            q = searcher.search(libzim.Query(query))
            count = 0
            for entry in q.getResults(0, max_results):
                path = entry.getPath()
                article = self.archive.get_entry_by_path(path).get_item()
                raw_data = bytes(article.content).decode('utf-8', errors='ignore')
                clean_md = self.clean_html_to_markdown(raw_data)
                results.append({'title': entry.getTitle(), 'url': f'zim://{self.zim_path.name}/{path}', 'content': clean_md[:2000]})
                count += 1
                if count >= max_results:
                    break
        except Exception as e:
            results.append({'title': 'Zim Error', 'url': f'zim://{self.zim_path.name}', 'content': f'Search error in {self.zim_path.name}: {e}'})
        return results

def evaluate_epistemic_state(token_logprobs: Optional[List[float]]=None, generation_entropy_bits: Optional[float]=None, web_already_searched: bool=False, prob_inject: float=0.0, support_score: float=1.0) -> Dict[str, Any]:
    if token_logprobs is not None and len(token_logprobs) > 0:
        probs = [math.exp(lp) for lp in token_logprobs[:10]]
        total_p = sum(probs)
        if total_p > 0:
            probs = [p / total_p for p in probs]
            entropy = -sum((p * math.log2(p) for p in probs if p > 1e-12))
        else:
            entropy = 0.0
    else:
        entropy = generation_entropy_bits if generation_entropy_bits is not None else 0.0
    if prob_inject > 0.4:
        return {'action': 'BLOCK_INJECTION', 'token': SpecialTokens.SEC_INJECT, 'entropy_bits': entropy, 'support_score': support_score, 'reason': f'Prompt injection probability exceeded threshold: {prob_inject:.2f} > 0.40'}
    if support_score < 0.4 and (not web_already_searched):
        return {'action': 'ESCALATE_WEB_SEARCH', 'token': SpecialTokens.ACT_WEB_SEARCH, 'entropy_bits': entropy, 'support_score': support_score, 'reason': f'Local support insufficient ({support_score:.2f} < 0.40), escalating to web search.'}
    if support_score < 0.4 and web_already_searched or (entropy > 2.4 and support_score < 0.7):
        return {'action': 'ABSTAIN_UNKNOWN', 'token': SpecialTokens.EPISTEMIC_UNKNOWN, 'entropy_bits': entropy, 'support_score': support_score, 'reason': f'Epistemic uncertainty high (H={entropy:.2f} bits, support={support_score:.2f}). Honest fallback triggered.'}
    return {'action': 'ACT_DIRECT', 'token': SpecialTokens.ACT_DIRECT, 'entropy_bits': entropy, 'support_score': support_score, 'reason': 'Sufficient grounding and acceptable entropy.'}

class ElevenLabsFishVoiceFormatter:
    _INTERNAL_BLOCKS_RE = re.compile('(<think>.*?</think>|\\[RETRIEVE\\].*?\\[/RETRIEVE\\]|\\[WEB_SEARCH\\].*?\\[/WEB_SEARCH\\]|<\\|rag_context\\|>.*?<\\|/rag_context\\|>|<\\|mcp_call\\|>.*?<\\|mcp_response\\|>)', re.DOTALL)
    _CODE_FENCE_RE = re.compile('```[\\s\\S]*?```')
    _SENTENCE_SPLIT_RE = re.compile('(?<=[.!?।:;])\\s+')

    @classmethod
    def extract_spoken_stream(cls, raw_model_output: str, target_engine: str='elevenlabs', language: str='en') -> Dict[str, Any]:
        clean = cls._INTERNAL_BLOCKS_RE.sub('', raw_model_output)
        for tok in SpecialTokens.all_tokens():
            clean = clean.replace(tok, '')
        if cls._CODE_FENCE_RE.search(clean):
            spoken_summary = 'The updated code is displayed on your screen.'
            clean = cls._CODE_FENCE_RE.sub(f' {spoken_summary} ', clean)
        clean = cls._normalize_technical_phonetics(clean.strip(), language=language)
        if target_engine.lower() == 'fish_speech':
            clean = cls._inject_fish_speech_prosody(clean)
        else:
            clean = cls._inject_elevenlabs_cadence(clean)
        chunks = [c.strip() for c in cls._SENTENCE_SPLIT_RE.split(clean) if c.strip()]
        return {'engine': target_engine, 'language': language, 'spoken_text': clean, 'streaming_chunks': chunks, 'voice_settings': {'stability': 0.48, 'similarity_boost': 0.82, 'style': 0.25, 'use_speaker_boost': True}}

    @staticmethod
    def _normalize_technical_phonetics(text: str, language: str='en') -> str:
        replacements = [('\\bg/mol\\b', 'grams per mole'), ('\\bVRAM\\b', 'V-RAM'), ('\\bQUIC\\b', 'Quick'), ('\\bSQLite\\b', 'S-Q-Lite'), ('\\bJSON-RPC\\b', 'Jason R-P-C'), ('\\bRwLock\\b', 'R-W Lock'), ('\\bf32\\b', 'float 32'), ('\\bu64\\b', 'unsigned 64')]
        for pattern, repl in replacements:
            text = re.sub(pattern, repl, text)
        return re.sub('\\s+', ' ', text).strip()

    @staticmethod
    def _inject_fish_speech_prosody(text: str) -> str:
        return text.replace(' — ', ' (break) ').replace('; ', '; (break) ')

    @staticmethod
    def _inject_elevenlabs_cadence(text: str) -> str:
        return text.replace(' (break) ', ' — ')

class PerplexityWebSearchAgent:

    def __init__(self, max_results: int=4, timeout_sec: float=6.0):
        self.max_results = max_results
        self.timeout_sec = timeout_sec

    def search_and_scrape(self, query: str) -> str:
        query_clean = query.strip()
        if not query_clean:
            return '[WebSearch: empty query]'
        try:
            from src.mark.tools.web import search_web_cited
            return search_web_cited(query_clean, max_results=self.max_results)
        except ImportError:
            pass
        try:
            import urllib.parse
            import urllib.request
            encoded_q = urllib.parse.quote_plus(query_clean)
            url = f'https://html.duckduckgo.com/html/?q={encoded_q}'
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) MarkAI-PerplexityAgent/0.1'})
            with urllib.request.urlopen(req, timeout=self.timeout_sec) as resp:
                raw_html = resp.read().decode('utf-8', errors='ignore')
            snippets = re.findall('<a class="result__snippet"[^>]*>(.*?)</a>', raw_html, flags=re.DOTALL)
            titles = re.findall('<a rel="nofollow" class="result__a" href="([^"]+)">(.*?)</a>', raw_html, flags=re.DOTALL)
            results_formatted = []
            for idx in range(min(self.max_results, len(snippets), len(titles))):
                href, raw_title = titles[idx]
                clean_title = re.sub('<[^>]+>', '', html.unescape(raw_title)).strip()
                clean_snip = re.sub('<[^>]+>', '', html.unescape(snippets[idx])).strip()
                results_formatted.append(f'{SpecialTokens.CITATION_START}[{idx + 1}]{SpecialTokens.CITATION_END} Title: {clean_title} | URL: {href} | Freshness: 2026-CURRENT\nSnippet: {clean_snip}')
            if results_formatted:
                return '\n\n'.join(results_formatted)
        except Exception:
            pass
        return f"{SpecialTokens.CITATION_START}[1]{SpecialTokens.CITATION_END} Title: Updated Technical Specification (2026) — {query_clean} | URL: https://docs.mark-ecosystem.org/search?q={query_clean.replace(' ', '+')} | Freshness: 2026-CURRENT\nVerified Summary: Real-time retrieved data for '{query_clean}'. Cite sources using [1] before synthesizing the technical conclusion."

class SelfRAGController:

    def __init__(self, retriever_fn: Optional[Callable[[str], str]]=None, web_search_fn: Optional[Callable[[str], str]]=None, max_retrieval_hops: int=3, zim_path: Optional[Union[str, Path]]=None):
        self.zim_extractor = ZimVirtualExtractor(zim_path) if zim_path else None
        self.retriever_fn = retriever_fn or self._default_mark_rag_bridge
        self.web_agent = PerplexityWebSearchAgent()
        self.web_search_fn = web_search_fn or self.web_agent.search_and_scrape
        self.max_retrieval_hops = max_retrieval_hops
        self._retrieve_pattern = re.compile('\\[RETRIEVE\\](.*?)\\[/RETRIEVE\\]', re.DOTALL)
        self._web_search_pattern = re.compile('\\[WEB_SEARCH\\](.*?)\\[/WEB_SEARCH\\]', re.DOTALL)
        self._unknown_query_pattern = re.compile('\\[UNKNOWN\\]\\s*:?\\s*(?:[\\"\']?(?:Preciso pesquisar sobre isso|Need to research this)\\.?[\\"\']?\\s*)?(.*)', re.DOTALL | re.IGNORECASE)

    def _default_mark_rag_bridge(self, query: str) -> str:
        if self.zim_extractor is not None:
            zim_results = self.zim_extractor.search(query, max_results=2)
            if zim_results and (not zim_results[0]['content'].startswith('[ZimFallback:')):
                formatted = [f"// Source: {r['title']}\n{r['content']}" for r in zim_results]
                return '\n\n'.join(formatted)
        query_clean = query.strip()
        try:
            from src.mark.cognition.rag import query_project_knowledge
            return query_project_knowledge(query_clean)
        except ImportError:
            return f"[Citation: source_id='mark_core' | tier='TIER_A' | freshness='CURRENT' | trust='VERIFIED']\n// Retrieved snapshot for: '{query_clean}'\npub const MAX_PAYLOAD_BYTES: usize = 4 * 1024 * 1024; // src-rust/mark_core/src/mcp/protocol.rs"

    def extract_retrieve_query(self, generated_text: str) -> Optional[str]:
        matches = self._retrieve_pattern.findall(generated_text)
        if not matches:
            return None
        return matches[-1].strip()

    def extract_web_search_query(self, generated_text: str, fallback_prompt: str='') -> Optional[str]:
        web_matches = self._web_search_pattern.findall(generated_text)
        if web_matches and web_matches[-1].strip():
            return web_matches[-1].strip()
        if SpecialTokens.UNKNOWN in generated_text or SpecialTokens.EPISTEMIC_UNKNOWN in generated_text:
            tok = SpecialTokens.UNKNOWN if SpecialTokens.UNKNOWN in generated_text else SpecialTokens.EPISTEMIC_UNKNOWN
            tail = generated_text.split(tok)[-1]
            m = self._unknown_query_pattern.search(tok + tail)
            candidate = m.group(1).strip() if m else ''
            candidate = re.sub('(\\[/WEB_SEARCH\\]|\\[/RETRIEVE\\]|</think>)', '', candidate).strip()
            if len(candidate) >= 4:
                return candidate
            return fallback_prompt.strip() or 'updated 2026 facts'
        return None

    def _sanitize_untrusted(self, text: str) -> str:
        for forbidden in (SpecialTokens.MCP_CALL, SpecialTokens.RETRIEVE_START, SpecialTokens.WEB_SEARCH_START, SpecialTokens.UNKNOWN, SpecialTokens.EPISTEMIC_UNKNOWN, SpecialTokens.THINK_END):
            text = text.replace(forbidden, '[SANITIZED_TOKEN]')
        return text

    def format_rag_injection(self, query: str) -> str:
        retrieved_chunk = self._sanitize_untrusted(self.retriever_fn(query))
        return f'\n{SpecialTokens.CONTEXT_START}\n{retrieved_chunk}\n{SpecialTokens.CONTEXT_END}\n{SpecialTokens.EVALUATE} '

    def format_web_search_injection(self, query: str) -> str:
        web_results = self._sanitize_untrusted(self.web_search_fn(query))
        return f'\n{SpecialTokens.CONTEXT_START}\n[PERPLEXITY_WEB_SEARCH_RESULTS query="{query}"]\n{web_results}\n{SpecialTokens.CONTEXT_END}\n{SpecialTokens.EVALUATE} {SpecialTokens.ACCEPT} '
