import tempfile
from pathlib import Path
import pytest
import torch
import numpy as np
import gguf

from src.model.mark_vl_v3_4b_arch import (
    MarkVLV3Config,
    MarkVLV3ForConditionalGeneration,
    Spatial2DRoPE,
    MarkV3VisionTower,
    MarkV3PixelShuffleProjector,
)
from export.convert_to_deepseek2_gguf import (
    export_to_deepseek2_gguf,
    verify_gguf_integrity,
    verify_mmproj_integrity,
)


def test_mark_vl_v3_4b_parameter_count():
    cfg = MarkVLV3Config()
    stats = cfg.count_parameters()
    total_params = stats["total_physical_parameters"]

    # Source of truth brackets: 3.9B to 4.2B total physical parameters
    assert 3_900_000_000 <= total_params <= 4_200_000_000, f"Total parameters out of range: {total_params}"
    assert total_params == 4_134_688_769, f"Exact parameter mismatch: {total_params}"

    # Verify active parameters per token
    active_backbone = stats["active_parameters_per_token_backbone"]
    assert 800_000_000 <= active_backbone <= 900_000_000, f"Active backbone parameters out of range: {active_backbone}"
    assert active_backbone == 824_483_584

    # Meta-device allocation confirms integer parity between analytical math and physical graph
    with torch.device("meta"):
        meta_model = MarkVLV3ForConditionalGeneration(cfg)
        meta_total = sum(p.numel() for p in meta_model.parameters())

    assert meta_total == total_params, f"Meta-device total ({meta_total}) != Analytical total ({total_params})"


def test_mark_vl_v3_4b_moe_block32_divisibility():
    cfg = MarkVLV3Config()

    # Intermediate sizes must be divisible by 32 for direct IQ4_NL super-block quantization without padding
    assert cfg.moe_intermediate_size % 32 == 0, f"MoE intermediate size {cfg.moe_intermediate_size} not divisible by 32"
    assert cfg.moe_intermediate_size == 832  # 26 blocks of 32

    # Vision intermediate and hidden dimensions
    assert cfg.vision_intermediate_size % 32 == 0, f"Vision intermediate size {cfg.vision_intermediate_size} not divisible by 32"
    assert cfg.vision_hidden_size % 32 == 0, f"Vision hidden size {cfg.vision_hidden_size} not divisible by 32"
    assert cfg.hidden_size % 32 == 0, f"Language hidden size {cfg.hidden_size} not divisible by 32"


def test_mark_vl_v3_4b_forward_multimodal_and_hadamard():
    mini_cfg = MarkVLV3Config(
        num_hidden_layers=2,
        vision_layers=2,
        vocab_size=32032,
        hidden_size=256,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=64,
        kv_lora_rank=32,
        qk_nope_head_dim=32,
        qk_rope_head_dim=32,
        v_head_dim=32,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=1,
        moe_intermediate_size=64,
        vision_hidden_size=128,
        vision_heads=4,
        vision_intermediate_size=256,
    )
    model = MarkVLV3ForConditionalGeneration(mini_cfg)
    model.eval()

    # Synthetic image 28x28: patch_size=14 -> 2x2 = 4 patches.
    # 2x2 pixel shuffle merges 4 patches into 1 visual token with feature dim 256.
    synthetic_image = torch.sin(torch.linspace(0.1, 10.0, 1 * 3 * 28 * 28)).reshape(1, 3, 28, 28)
    visual_embeds = model.encode_images(synthetic_image)
    assert visual_embeds.shape == (1, 1, mini_cfg.hidden_size), f"Unexpected visual embeds shape: {visual_embeds.shape}"

    # Multimodal sequence containing image placeholder token (32016)
    input_ids = torch.tensor([[1, mini_cfg.image_token_id, 25, 42, 2]], dtype=torch.long)

    with torch.no_grad():
        logits_pre = model(input_ids, images=synthetic_image)
        # Execute offline Walsh-Hadamard weight absorption across all backbone layers
        model.fuse_hadamard_weights_()
        logits_post = model(input_ids, images=synthetic_image)

    assert logits_pre.shape == (1, 5, mini_cfg.vocab_size)
    assert logits_post.shape == (1, 5, mini_cfg.vocab_size)

    # Numerical equivalence before and after Hadamard absorption
    max_diff = (logits_pre - logits_post).abs().max().item()
    assert torch.allclose(logits_pre, logits_post, atol=1e-4), f"Hadamard fusion divergence: {max_diff}"


def test_mark_vl_v3_4b_gguf_and_mmproj_export():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        mini_cfg = MarkVLV3Config(
            num_hidden_layers=1,
            vision_layers=1,
            vocab_size=256,
        )
        model = MarkVLV3ForConditionalGeneration(mini_cfg)
        model.eval()
        sd = model.state_dict()

        pt_file = tmp_path / "mini_vl.pt"
        gguf_file = tmp_path / "mark_vl_v3_4b_iq4_nl.gguf"
        mmproj_file = tmp_path / "mmproj-mark_vl_v3_4b.gguf"

        torch.save(sd, pt_file)

        export_to_deepseek2_gguf(
            "vl-v3-4b",
            ckpt_path=pt_file,
            output_gguf=gguf_file,
            model_name="Mark-VL-V3-4B",
            quant_type="iq4_nl",
            mmproj_path=mmproj_file,
            mmproj_quant="q8_0",
        )

        verify_gguf_integrity(gguf_file)
        verify_mmproj_integrity(mmproj_file)

        reader_lm = gguf.GGUFReader(str(gguf_file))
        lm_tensors = {t.name: t for t in reader_lm.tensors}
        assert "token_embd.weight" in lm_tensors
        assert "blk.0.attn_kv_a_mqa.weight" in lm_tensors
        assert "blk.0.ffn_gate_exps.weight" in lm_tensors

        reader_mm = gguf.GGUFReader(str(mmproj_file))
        mm_tensors = {t.name: t for t in reader_mm.tensors}
        assert "v.patch_embed.weight" in mm_tensors
        assert "mm.w_gate.weight" in mm_tensors
        assert "mm.alpha_vis" in mm_tensors
