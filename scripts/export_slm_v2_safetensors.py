import sys
from pathlib import Path
import torch
import safetensors.torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "models" / "mark-slm-v2-302m"))

from configuration_mark_slm_v2 import MarkSLMV2Config
from modeling_mark_slm_v2 import MarkSLMV2ForCausalLM

CKPT_PATH = Path("/mnt/ssd_1/mark ai/checkpoints/colab_302m/mark_slm_v2_302m_step_10000.pt")
OUTPUT_SAFETENSORS = REPO_ROOT / "models" / "mark-slm-v2-302m" / "model.safetensors"

def project_weight_svd(w: torch.Tensor, in_features: int, out_features: int) -> torch.Tensor:
    if w.shape == (out_features, in_features):
        return w
    cur_out, cur_in = w.shape
    if cur_out != out_features and cur_in == in_features:
        U, S, Vh = torch.linalg.svd(w.float(), full_matrices=False)
        k = min(out_features, cur_out)
        w_proj = torch.diag(S[:k]) @ Vh[:k, :]
        if k < out_features:
            pad = torch.zeros((out_features - k, in_features), dtype=torch.float32, device=w.device)
            w_proj = torch.cat([w_proj, pad], dim=0)
        return w_proj
    elif cur_out == out_features and cur_in != in_features:
        U, S, Vh = torch.linalg.svd(w.float(), full_matrices=False)
        k = min(in_features, cur_in)
        w_proj = U[:, :k] @ torch.diag(S[:k])
        if k < in_features:
            pad = torch.zeros((out_features, in_features - k), dtype=torch.float32, device=w.device)
            w_proj = torch.cat([w_proj, pad], dim=1)
        return w_proj
    else:
        return w[:out_features, :in_features]

def main():
    print(f"Loading checkpoint from {CKPT_PATH}...")
    pt_data = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    sd_pt = pt_data.get("model_state_dict", pt_data)
    print(f"Loaded {len(sd_pt)} tensors. Step: {pt_data.get('step')}, Loss: {pt_data.get('loss')}")

    mapped_sd = {}
    for k, v in sd_pt.items():
        if not isinstance(v, torch.Tensor):
            continue
        new_k = k
        if new_k.startswith("tok_embeddings."):
            new_k = new_k.replace("tok_embeddings.", "model.embed_tokens.")
        elif not new_k.startswith("model.") and not new_k.startswith("lm_head.") and not new_k.startswith("mtp_"):
            new_k = "model." + new_k
        new_k = new_k.replace(".ffn.router.", ".ffn.router_gate.")
        new_k = new_k.replace(".ffn.shared_experts.w_", ".ffn.shared_experts.0.w_")

        # Project 400 -> 384
        if "w_gate.weight" in new_k or "w_up.weight" in new_k:
            if v.shape == (400, 768):
                v = project_weight_svd(v, 768, 384)
        elif "w_down.weight" in new_k:
            if v.shape == (768, 400):
                v = project_weight_svd(v, 384, 768)

        mapped_sd[new_k] = v.contiguous().to(torch.float32)

    config = MarkSLMV2Config.from_pretrained(REPO_ROOT / "models" / "mark-slm-v2-302m")
    model = MarkSLMV2ForCausalLM(config)
    res = model.load_state_dict(mapped_sd, strict=True)
    print("State dict loaded with strict=True successfully!")
    print(f"Missing keys: {len(res.missing_keys)}, Unexpected keys: {len(res.unexpected_keys)}")

    # Ensure tied weights have separate buffers for safetensors serialization
    if "model.embed_tokens.weight" in mapped_sd and "lm_head.weight" in mapped_sd:
        mapped_sd["lm_head.weight"] = mapped_sd["model.embed_tokens.weight"].clone()

    print(f"Saving safetensors to {OUTPUT_SAFETENSORS}...")
    safetensors.torch.save_file(mapped_sd, str(OUTPUT_SAFETENSORS), metadata={"format": "pt"})
    print(f"Saved {OUTPUT_SAFETENSORS} successfully ({OUTPUT_SAFETENSORS.stat().st_size:,} bytes).")

if __name__ == "__main__":
    main()
