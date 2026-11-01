#!/usr/bin/env python3
"""
Mark Engineer 1.5B - Interactive Inference Application (Hugging Face Space)
Hardware Target: cpu-basic (2 vCPU, 16 GB RAM, Free Tier)
Architecture: Native DeepSeek2 MLA + 3D MoE (IQ4_NL Quantization)
Total Parameters: 1,496,121,824 | Active Parameters: 364,151,264 (24.34%)
"""

import os
import time
import math
from pathlib import Path
from typing import Generator, Tuple, List, Optional
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
import gradio as gr
from huggingface_hub import hf_hub_download

# Lookup table for IQ4_NL non-linear quantization values
IQ4_NL_LOOKUP_VALUES = [
    -1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453,
    -0.28444138169288635, -0.18477343022823334, -0.09105003625154495, 0.0,
    0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
    0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0
]

LUT_VALUES = torch.tensor(IQ4_NL_LOOKUP_VALUES, dtype=torch.float32)
LUT_256 = torch.zeros(256, 2, dtype=torch.float32)
for b in range(256):
    LUT_256[b, 0] = LUT_VALUES[b & 15]
    LUT_256[b, 1] = LUT_VALUES[(b >> 4) & 15]

# Precompute RoPE freqs
def precompute_freqs_cis(head_dim: int, max_seq_len: int = 4096, theta: float = 500000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    return torch.cos(freqs), torch.sin(freqs)

ROPE_COS, ROPE_SIN = precompute_freqs_cis(32, 4096)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    dtype = xq.dtype
    xq_f = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_f = xk.float().reshape(*xk.shape[:-1], -1, 2)
    cos = cos.unsqueeze(0).unsqueeze(2)
    sin = sin.unsqueeze(0).unsqueeze(2)
    xq_r, xq_i = (xq_f[..., 0], xq_f[..., 1])
    xk_r, xk_i = (xk_f[..., 0], xk_f[..., 1])
    xq_out = torch.stack([xq_r * cos - xq_i * sin, xq_r * sin + xq_i * cos], dim=-1).flatten(3)
    xk_out = torch.stack([xk_r * cos - xk_i * sin, xk_r * sin + xk_i * cos], dim=-1).flatten(3)
    return xq_out.to(dtype), xk_out.to(dtype)

def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)) * w.float()

def dequant_weight(sd: dict, prefix: str, in_features: int, out_features: int) -> torch.Tensor:
    qk = f"{prefix}.qweight"
    sk = f"{prefix}.scales"
    if qk in sd and sk in sd:
        qw = sd[qk]
        sc = sd[sk].float()
        w_norm = LUT_256[qw.long()].view(out_features, qw.shape[1], 32)
        return (w_norm * sc).view(out_features, -1)[:, :in_features]
    elif f"{prefix}.weight" in sd:
        return sd[f"{prefix}.weight"].float()
    elif prefix in sd:
        return sd[prefix].float()
    raise KeyError(f"Missing tensor: {prefix}")


class MarkEngineerEngine:
    def __init__(self):
        self.device = torch.device("cpu")
        self.checkpoint = None
        self.tokenizer = None
        self.load_artifacts()

    def load_artifacts(self):
        print("[Mark Engine] Loading tokenizer...")
        local_tok = Path("tokenizer.json")
        if not local_tok.exists():
            tok_path = hf_hub_download(
                repo_id="miceworks/Mark-Engineer-1.5B",
                filename="tokenizer.json",
            )
        else:
            tok_path = str(local_tok)
        self.tokenizer = Tokenizer.from_file(tok_path)
        print(f"[Mark Engine] Tokenizer loaded (vocab: {self.tokenizer.get_vocab_size()})")

        print("[Mark Engine] Loading trained IQ4_NL checkpoint...")
        local_ckpt = Path("mark_engineer_1_5b_iq4_nl.pt")
        candidates = [
            local_ckpt,
            Path("/mnt/ssd_1/mark ai/checkpoints/mark_engineer_1_5b_iq4_nl.pt"),
            Path("/run/media/thu12/hd 1tb/mark_checkpoints/mark_engineer_1_5b_iq4_nl.pt"),
        ]
        found = next((p for p in candidates if p.exists()), None)
        if found:
            ckpt_file = str(found)
        else:
            ckpt_file = hf_hub_download(
                repo_id="miceworks/Mark-Engineer-1.5B",
                filename="mark_engineer_1_5b_iq4_nl.pt",
            )
        self.checkpoint = torch.load(ckpt_file, map_location="cpu", weights_only=False)
        print(f"[Mark Engine] Checkpoint loaded ({len(self.checkpoint)} tensors)")

    def forward_layer(self, h: torch.Tensor, i: int, seq_len: int) -> torch.Tensor:
        sd = self.checkpoint
        # 1. Attention Norm
        h_norm = rms_norm(h, sd[f"layers.{i}.attn_norm.weight"])

        # 2. MLA Projections
        wq_a = dequant_weight(sd, f"layers.{i}.attn.wq_a", 1280, 640)
        q_a = h_norm @ wq_a.T
        q_a_norm = rms_norm(q_a, sd[f"layers.{i}.attn.q_norm.weight"])
        wq_b = dequant_weight(sd, f"layers.{i}.attn.wq_b", 640, 16 * 96)
        q = (q_a_norm @ wq_b.T).view(seq_len, 16, 96)

        wkv_a = dequant_weight(sd, f"layers.{i}.attn.wkv_a", 1280, 320 + 32)
        kv_a = h_norm @ wkv_a.T
        kv_norm = rms_norm(kv_a[:, :320], sd[f"layers.{i}.attn.kv_norm.weight"])

        wkv_b = dequant_weight(sd, f"layers.{i}.attn.wkv_b", 320, 16 * 160)
        kv_b = (kv_norm @ wkv_b.T).view(seq_len, 16, 160)

        k_nope, v = torch.split(kv_b, [64, 96], dim=-1)
        q_nope, q_rope = torch.split(q, [64, 32], dim=-1)
        k_rope = kv_a[:, 320:].view(seq_len, 1, 32).expand(-1, 16, 32)

        cos = ROPE_COS[:seq_len]
        sin = ROPE_SIN[:seq_len]
        q_rope, k_rope = apply_rotary_emb(q_rope.unsqueeze(0), k_rope.unsqueeze(0), cos, sin)
        q_rope = q_rope.squeeze(0)
        k_rope = k_rope.squeeze(0)

        q_full = torch.cat([q_nope, q_rope], dim=-1)
        k_full = torch.cat([k_nope, k_rope], dim=-1)

        scores = torch.einsum("t h d, s h d -> h t s", q_full, k_full) / (96.0 ** 0.5)
        causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
        scores = scores.masked_fill(~causal.unsqueeze(0), -1e9)
        attn = F.softmax(scores, dim=-1)
        out_attn = torch.einsum("h t s, s h d -> t h d", attn, v).reshape(seq_len, -1)

        wo = dequant_weight(sd, f"layers.{i}.attn.wo", 16 * 96, 1280)
        h = h + out_attn @ wo.T

        # 3. FFN / MoE
        h_fnorm = rms_norm(h, sd[f"layers.{i}.ffn_norm.weight"])

        if i < 4:
            # Dense layers (0, 1, 2, 3)
            w_gate = dequant_weight(sd, f"layers.{i}.ffn.w_gate", 1280, 3840)
            w_up = dequant_weight(sd, f"layers.{i}.ffn.w_up", 1280, 3840)
            w_down = dequant_weight(sd, f"layers.{i}.ffn.w_down", 3840, 1280)
            ffn_out = (F.silu(h_fnorm @ w_gate.T) * (h_fnorm @ w_up.T)) @ w_down.T
        else:
            # MoE layers (4..15)
            router = sd[f"layers.{i}.ffn.router.weight"].float()
            router_logits = h_fnorm @ router.T
            top2_weights, top2_indices = torch.topk(F.softmax(router_logits, dim=-1), k=2, dim=-1)

            # Shared Expert (1S)
            w_sgate = dequant_weight(sd, f"layers.{i}.ffn.shared_experts.w_gate", 1280, 1504)
            w_sup = dequant_weight(sd, f"layers.{i}.ffn.shared_experts.w_up", 1280, 1504)
            w_sdown = dequant_weight(sd, f"layers.{i}.ffn.shared_experts.w_down", 1504, 1280)
            shared_out = (F.silu(h_fnorm @ w_sgate.T) * (h_fnorm @ w_sup.T)) @ w_sdown.T

            # Routed Experts (2 of 16 active per token)
            routed_out = torch.zeros_like(shared_out)
            last_idx = seq_len - 1
            for k_pos in range(2):
                exp_id = int(top2_indices[last_idx, k_pos].item())
                w_val = top2_weights[last_idx, k_pos]
                w_egate = dequant_weight(sd, f"layers.{i}.ffn.routed_experts.{exp_id}.w_gate", 1280, 1504)
                w_eup = dequant_weight(sd, f"layers.{i}.ffn.routed_experts.{exp_id}.w_up", 1280, 1504)
                w_edown = dequant_weight(sd, f"layers.{i}.ffn.routed_experts.{exp_id}.w_down", 1504, 1280)
                expert_act = (F.silu(h_fnorm[last_idx:last_idx+1] @ w_egate.T) * (h_fnorm[last_idx:last_idx+1] @ w_eup.T)) @ w_edown.T
                routed_out[last_idx:last_idx+1] += w_val * expert_act

            ffn_out = shared_out + routed_out

        h = h + ffn_out
        return h

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 48,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Generator[Tuple[str, str], None, None]:
        if not prompt.strip():
            yield "", "Status: Enter a prompt to generate."
            return

        tokens = self.tokenizer.encode(prompt).ids
        if len(tokens) == 0:
            tokens = [1]  # BOS

        output_tokens = []
        start_time = time.time()
        token_times = []

        sd = self.checkpoint
        w_emb = sd["tok_embeddings.weight"].float()
        w_norm = sd["norm.weight"].float()

        for step in range(max_new_tokens):
            step_start = time.time()
            seq_len = len(tokens)
            if seq_len > 2048:
                break

            input_tensor = torch.tensor(tokens, dtype=torch.long)
            h = w_emb[input_tensor]

            for layer_idx in range(16):
                h = self.forward_layer(h, layer_idx, seq_len)

            h_last = rms_norm(h[-1:], w_norm)
            logits = (h_last @ w_emb.T).squeeze(0)

            if temperature <= 1e-4:
                next_token = int(torch.argmax(logits).item())
            else:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cum_probs = torch.cumsum(sorted_probs, dim=-1)
                mask = cum_probs - sorted_probs > top_p
                sorted_probs[mask] = 0.0
                sorted_probs /= sorted_probs.sum()
                sampled_idx = torch.multinomial(sorted_probs, 1).item()
                next_token = int(sorted_indices[sampled_idx].item())

            step_end = time.time()
            token_times.append((step_end - step_start) * 1000)

            if next_token == 2:  # EOS
                break

            tokens.append(next_token)
            output_tokens.append(next_token)

            current_text = self.tokenizer.decode(output_tokens)
            avg_latency = sum(token_times) / len(token_times)
            tokens_per_sec = 1000.0 / avg_latency if avg_latency > 0 else 0.0

            telemetry = (
                f"Generated Tokens: {len(output_tokens)} | "
                f"Latency: {avg_latency:.1f} ms/token ({tokens_per_sec:.2f} tok/s) | "
                f"Active Parameters: 364.15M / 1,496,121,824 (24.34% active) | "
                f"MLA c_KV: 320 | MoE: 16R+1S (Top-2) | Hardware: cpu-basic (2 vCPU)"
            )
            yield current_text, telemetry

        total_time = time.time() - start_time
        final_text = self.tokenizer.decode(output_tokens)
        final_latency = (total_time / len(output_tokens) * 1000) if output_tokens else 0
        final_telemetry = (
            f"Completed: {len(output_tokens)} tokens in {total_time:.2f}s | "
            f"Average Latency: {final_latency:.1f} ms/token | "
            f"Active: 364.15M / 1.50B | MLA c_KV=320 | MoE 16R+1S"
        )
        yield final_text, final_telemetry


# Global Engine Instance
print("[Mark Space] Initializing MarkEngineerEngine...")
ENGINE = MarkEngineerEngine()
print("[Mark Space] Engine ready.")

def chat_interface(prompt: str, max_tokens: int, temperature: float, top_p: float):
    for text, telemetry in ENGINE.generate(prompt, max_new_tokens=int(max_tokens), temperature=temperature, top_p=top_p):
        yield text, telemetry

CUSTOM_CSS = """
.container { max-width: 900px; margin: auto; }
.header-box { background: #0f172a; border: 1px solid #334155; padding: 1.5rem; border-radius: 8px; margin-bottom: 1rem; color: #f8fafc; }
.header-box h1 { color: #38bdf8; margin: 0 0 0.5rem 0; font-size: 1.75rem; }
.badge { display: inline-block; background: #1e293b; color: #94a3b8; border: 1px solid #475569; padding: 2px 8px; border-radius: 9999px; font-size: 0.8rem; margin-right: 0.5rem; }
.telemetry-box { background: #090d16; border: 1px solid #1e293b; color: #38bdf8; font-family: monospace; font-size: 0.85rem; padding: 0.75rem; border-radius: 6px; }
"""

with gr.Blocks(title="Mark Engineer 1.5B - Interactive Inference", theme=gr.themes.Base()) as demo:
    gr.HTML("""
    <div class="header-box">
        <h1>Mark-Engineer-1.5B (DeepSeek2 MLA + 3D MoE)</h1>
        <div>
            <span class="badge">Architecture: DeepSeek2 MoE</span>
            <span class="badge">Quantization: IQ4_NL (4-bit Non-Linear)</span>
            <span class="badge">MLA c_KV: 320</span>
            <span class="badge">Active: 364.15M / 1.50B</span>
            <span class="badge">Hardware: cpu-basic (2 vCPU)</span>
        </div>
        <p style="margin-top: 0.75rem; color: #94a3b8; font-size: 0.9rem;">
            Real-time interactive CPU inference powered by native DeepSeek2 Multi-Head Latent Attention and Mixture-of-Experts.
            Trained on systems engineering, low-level OS kernels, concurrency primitives, and systems code.
        </p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=4):
            prompt_input = gr.Textbox(
                label="Prompt",
                placeholder="Enter a systems engineering instruction or code stub...",
                lines=4,
                value="void spinlock_acquire(volatile int *lock) {"
            )
            generate_btn = gr.Button("Generate Code", variant="primary")

            output_text = gr.Code(
                label="Generated Systems Code",
                language="c",
                lines=10,
                interactive=False
            )
            telemetry_text = gr.Textbox(
                label="Real-Time Telemetry",
                interactive=False,
                elem_classes=["telemetry-box"]
            )

        with gr.Column(scale=2):
            gr.Markdown("### Generation Parameters")
            max_tokens_slider = gr.Slider(minimum=8, maximum=128, value=32, step=8, label="Max New Tokens")
            temperature_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.6, step=0.05, label="Temperature")
            top_p_slider = gr.Slider(minimum=0.1, maximum=1.0, value=0.9, step=0.05, label="Top-p Sampling")

            gr.Markdown("### Technical Specifications")
            gr.Markdown("""
            - **Model**: [miceworks/Mark-Engineer-1.5B](https://huggingface.co/miceworks/Mark-Engineer-1.5B)
            - **Physical Parameters**: 1,496,121,824
            - **Active Parameters**: 364,151,264 (24.34%)
            - **Attention**: MLA (c_KV=320, q_lora=640)
            - **Topology**: 16 Layers (4 Dense + 12 MoE)
            - **MoE Experts**: 16 Routed + 1 Shared (Top-2)
            - **Vocab Size**: 4,311 BPE Tokens
            - **Binary Weight**: IQ4_NL Quantized
            """)

            gr.Examples(
                examples=[
                    ["void spinlock_acquire(volatile int *lock) {", 32, 0.6, 0.9],
                    ["struct ring_buffer {\n    size_t head;\n    size_t tail;", 32, 0.6, 0.9],
                    ["// Explain how MLA compresses the KV cache:\n//", 32, 0.6, 0.9],
                ],
                inputs=[prompt_input, max_tokens_slider, temperature_slider, top_p_slider],
            )

    generate_btn.click(
        fn=chat_interface,
        inputs=[prompt_input, max_tokens_slider, temperature_slider, top_p_slider],
        outputs=[output_text, telemetry_text],
    )

if __name__ == "__main__":
    demo.queue().launch()
