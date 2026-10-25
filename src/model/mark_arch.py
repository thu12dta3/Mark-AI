"""
Mark 0.1 - Core Neural Architecture (PyTorch Native)
Target: ~3.0B Dense Parameters | GQA (4:1) | SwiGLU | RoPE (8k-32k Dynamic Scaling) | FlashAttention-2
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MarkConfig:
    vocab_size: int = 49152          # Vocabulário BPE otimizado (múltiplo de 256 p/ Tensor Cores A100)
    dim: int = 2560                  # Hidden dimension (d_model)
    n_layers: int = 32               # Profundidade do Transformer
    n_heads: int = 32                # Cabeças de Query (head_dim = 80)
    n_kv_heads: int = 8              # Cabeças Key/Value (GQA ratio 4:1 -> reduz KV cache em 75%)
    intermediate_size: int = 6912    # Dimensão oculta SwiGLU (~8/3 * dim, alinhado em 256)
    max_seq_len: int = 8192          # Contexto base de treino (expansível até 32768 via NTK-RoPE)
    rope_theta: float = 500000.0     # Base de frequência alta para estabilidade em contextos longos (8k-32k)
    rope_scaling_factor: float = 1.0 # Fator de interpolação dinâmica (1.0 = 8k, 4.0 = 32k)
    rms_norm_eps: float = 1e-5       # Epsilon de estabilidade para RMSNorm em BF16/FP16
    dropout: float = 0.0             # Zero dropout em pré-treino moderno
    tie_word_embeddings: bool = True # Compartilhamento de pesos Embed <-> LM Head (economiza ~125M params)

    # IDs de Tokens Especiais de Controle (Alinhados com tokenizer_setup.py)
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    think_start_id: int = 3          # <think>
    think_end_id: int = 4            # </think>
    retrieve_id: int = 5             # [RETRIEVE]
    retrieve_end_id: int = 6         # [/RETRIEVE]
    evaluate_id: int = 7             # [EVALUATE]
    accept_id: int = 8               # [ACCEPT]
    reject_id: int = 9               # [REJECT]

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    def count_parameters(self) -> int:
        embed = self.vocab_size * self.dim
        attn_per_layer = self.dim * (
            self.n_heads * self.head_dim +
            2 * self.n_kv_heads * self.head_dim +
            self.n_heads * self.head_dim
        )
        ffn_per_layer = 3 * self.dim * self.intermediate_size
        norm_per_layer = 2 * self.dim
        total = embed + self.n_layers * (attn_per_layer + ffn_per_layer + norm_per_layer) + self.dim
        if not self.tie_word_embeddings:
            total += embed
        return total


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization:
    RMSNorm(x) = (x / sqrt(mean(x^2) + eps)) * weight
    Cálculo forçado em FP32 para evitar underflow/overflow na A100 (BF16) e RTX 8GB (FP16).
    """
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(
    head_dim: int,
    max_seq_len: int,
    theta: float = 500000.0,
    scaling_factor: float = 1.0,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Pré-calcula matrizes de rotação cos(m*theta_i) e sin(m*theta_i) para RoPE.
    Suporta NTK-aware Dynamic Scaling para extrapolação de 8k -> 32k tokens.
    """
    if scaling_factor > 1.0:
        theta = theta * (
            (scaling_factor * max_seq_len / max_seq_len) - (scaling_factor - 1.0)
        ) ** (head_dim / (head_dim - 2))

    freqs = 1.0 / (
        theta ** (torch.arange(0, head_dim, 2, device=device)[: (head_dim // 2)].float() / head_dim)
    )
    t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, freqs)  # (max_seq_len, head_dim // 2)
    cos = torch.cos(freqs)
    sin = torch.sin(freqs)
    return cos, sin


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Aplica a rotação complexa em pares de dimensões adjacentes de Query e Key:
    f_q(x_m, m) = (W_q x_m) e^{i m theta}
    """
    dtype = xq.dtype
    xq_f = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_f = xk.float().reshape(*xk.shape[:-1], -1, 2)

    # Broadcast para (1, seqlen, 1, head_dim // 2)
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)

    xq_r, xq_i = xq_f[..., 0], xq_f[..., 1]
    xk_r, xk_i = xk_f[..., 0], xk_f[..., 1]

    xq_out_r = xq_r * cos - xq_i * sin
    xq_out_i = xq_r * sin + xq_i * cos
    xk_out_r = xk_r * cos - xk_i * sin
    xk_out_i = xk_r * sin + xk_i * cos

    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.to(dtype), xk_out.to(dtype)


class PagedKVCache:
    """
    KV Cache pré-alocado com gerenciamento estrito de VRAM para o limite de 8GB.
    Evita chamadas repetidas de cudaMalloc que causam fragmentação e OOM em contextos longos.
    """
    def __init__(
        self,
        n_layers: int,
        max_batch_size: int,
        max_seq_len: int,
        n_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype = torch.float16,
        device: torch.device = torch.device("cuda"),
    ):
        self.max_seq_len = max_seq_len
        self.k_cache = torch.zeros(
            (n_layers, max_batch_size, n_kv_heads, max_seq_len, head_dim),
            dtype=dtype,
            device=device,
        )
        self.v_cache = torch.zeros(
            (n_layers, max_batch_size, n_kv_heads, max_seq_len, head_dim),
            dtype=dtype,
            device=device,
        )

    def update(
        self,
        layer_idx: int,
        xk: torch.Tensor,
        xv: torch.Tensor,
        start_pos: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz, _, seqlen, _ = xk.shape
        end_pos = start_pos + seqlen
        if end_pos > self.max_seq_len:
            # Rolling window eviction mantendo os primeros 128 tokens (Attention Sinks) + contexto recente
            sink_tokens = 128
            shift = end_pos - self.max_seq_len
            keep_len = self.max_seq_len - sink_tokens - seqlen
            self.k_cache[layer_idx, :bsz, :, sink_tokens : sink_tokens + keep_len, :] = (
                self.k_cache[layer_idx, :bsz, :, sink_tokens + shift : sink_tokens + shift + keep_len, :].clone()
            )
            self.v_cache[layer_idx, :bsz, :, sink_tokens : sink_tokens + keep_len, :] = (
                self.v_cache[layer_idx, :bsz, :, sink_tokens + shift : sink_tokens + shift + keep_len, :].clone()
            )
            start_pos = self.max_seq_len - seqlen
            end_pos = self.max_seq_len

        self.k_cache[layer_idx, :bsz, :, start_pos:end_pos, :] = xk
        self.v_cache[layer_idx, :bsz, :, start_pos:end_pos, :] = xv
        return (
            self.k_cache[layer_idx, :bsz, :, :end_pos, :],
            self.v_cache[layer_idx, :bsz, :, :end_pos, :],
        )

    def reset(self):
        self.k_cache.zero_()
        self.v_cache.zero_()

    def memory_allocated_mb(self) -> float:
        total_bytes = self.k_cache.numel() * self.k_cache.element_size() * 2
        return total_bytes / (1024 ** 2)


class MarkSelfAttention(nn.Module):
    """
    Grouped-Query Attention (GQA) com FlashAttention-2 nativo via F.scaled_dot_product_attention.
    Compatível com a especificação da Architecture Bible (Seção 7).
    """
    def __init__(self, config: MarkConfig, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads
        self.dropout = config.dropout

        self.wq = nn.Linear(config.dim, config.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, config.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(config.n_heads * self.head_dim, config.dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[PagedKVCache] = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        bsz, seqlen, _ = x.shape

        xq = self.wq(x).view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = self.wk(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = self.wv(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        # Aplicação do Rotary Positional Embedding (RoPE)
        xq, xk = apply_rotary_emb(xq, xk, cos, sin)

        xq = xq.transpose(1, 2)  # (bsz, n_heads, seqlen, head_dim)
        xk = xk.transpose(1, 2)  # (bsz, n_kv_heads, seqlen, head_dim)
        xv = xv.transpose(1, 2)

        if kv_cache is not None:
            xk, xv = kv_cache.update(self.layer_idx, xk, xv, start_pos)

        # GQA zero-copy view expansion (evita duplicação física de memória de repeat_interleave)
        if self.n_rep > 1:
            xk = (
                xk[:, :, None, :, :]
                .expand(bsz, self.n_kv_heads, self.n_rep, -1, self.head_dim)
                .reshape(bsz, self.n_heads, -1, self.head_dim)
            )
            xv = (
                xv[:, :, None, :, :]
                .expand(bsz, self.n_kv_heads, self.n_rep, -1, self.head_dim)
                .reshape(bsz, self.n_heads, -1, self.head_dim)
            )

        # FlashAttention-2 Nativo: Evita carregar matriz QxK completa na VRAM
        is_causal = (mask is None) and (kv_cache is None or seqlen > 1)
        output = F.scaled_dot_product_attention(
            xq,
            xk,
            xv,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)


class MarkSwiGLUFFN(nn.Module):
    """
    SwiGLU(x) = Swish(x W_gate) (x) (x W_up) W_down
    Evita congelamento de neurônios em contextos longos de código Rust/Python.
    """
    def __init__(self, config: MarkConfig):
        super().__init__()
        self.w_gate = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w_up = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w_down = nn.Linear(config.intermediate_size, config.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class MarkTransformerBlock(nn.Module):
    def __init__(self, config: MarkConfig, layer_idx: int):
        super().__init__()
        self.attn_norm = RMSNorm(config.dim, eps=config.rms_norm_eps)
        self.attn = MarkSelfAttention(config, layer_idx=layer_idx)
        self.ffn_norm = RMSNorm(config.dim, eps=config.rms_norm_eps)
        self.ffn = MarkSwiGLUFFN(config)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[PagedKVCache] = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        h = x + self.attn(self.attn_norm(x), cos, sin, mask=mask, kv_cache=kv_cache, start_pos=start_pos)
        out = h + self.ffn(self.ffn_norm(h))
        return out


class MarkModel(nn.Module):
    """
    Rede Neural Autoregressiva Completa Mark 0.1 (~2.95B parâmetros densos).
    """
    def __init__(self, config: MarkConfig):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList([
            MarkTransformerBlock(config, layer_idx=i) for i in range(config.n_layers)
        ])
        self.norm = RMSNorm(config.dim, eps=config.rms_norm_eps)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.tok_embeddings.weight

        cos, sin = precompute_freqs_cis(
            config.head_dim,
            config.max_seq_len * int(max(1.0, config.rope_scaling_factor)),
            theta=config.rope_theta,
            scaling_factor=config.rope_scaling_factor,
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        self.gradient_checkpointing = False
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        std = 0.02
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

        for name, p in module.named_parameters():
            if name.endswith("wo.weight") or name.endswith("w_down.weight"):
                nn.init.normal_(p, mean=0.0, std=std / math.sqrt(2 * self.config.n_layers))

    def enable_gradient_checkpointing(self):
        self.gradient_checkpointing = True

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[PagedKVCache] = None,
        start_pos: int = 0,
    ) -> torch.Tensor:
        bsz, seqlen = input_ids.shape
        h = self.tok_embeddings(input_ids)

        if start_pos + seqlen > self.rope_cos.shape[0]:
            new_max = start_pos + seqlen + 1024
            cos, sin = precompute_freqs_cis(
                self.config.head_dim,
                new_max,
                theta=self.config.rope_theta,
                scaling_factor=max(1.0, new_max / self.config.max_seq_len),
                device=input_ids.device,
            )
            self.rope_cos = cos
            self.rope_sin = sin

        cos = self.rope_cos[start_pos : start_pos + seqlen].to(h.device)
        sin = self.rope_sin[start_pos : start_pos + seqlen].to(h.device)

        causal_mask = None
        if attention_mask is not None and kv_cache is None:
            causal_mask = torch.ones((bsz, 1, seqlen, seqlen), dtype=torch.bool, device=input_ids.device).tril(diagonal=0)
            padding_mask = attention_mask[:, None, None, :].bool()
            causal_mask = causal_mask & padding_mask

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                h = torch.utils.checkpoint.checkpoint(
                    layer,
                    h,
                    cos,
                    sin,
                    causal_mask,
                    None,
                    0,
                    use_reentrant=False,
                )
            else:
                h = layer(h, cos, sin, mask=causal_mask, kv_cache=kv_cache, start_pos=start_pos)

        h = self.norm(h)
        logits = self.lm_head(h)
        return logits


MarkForCausalLM = MarkModel
