import pytest
import torch
import gguf
import tempfile
from pathlib import Path

from src.model.mark_arch import MarkConfig, MarkModel
from src.model.mark_v3_arch import MarkV3Config, MarkV3ForCausalLM
from src.model.mark_vl_2_2b_arch import MarkVL2_2BConfig, MarkVLForCausalLM
from src.model.mark_general_2_2b_arch import MarkGeneral2_2BConfig, MarkGeneral2_2BForConditionalGeneration
from src.model.self_rag import SpecialTokens, ZimVirtualExtractor, evaluate_epistemic_state
from export.convert_to_deepseek2_gguf import export_to_deepseek2_gguf, verify_gguf_integrity


def test_mark_v3_forward_and_rope():
    cfg = MarkV3Config(num_hidden_layers=2)
    model = MarkV3ForCausalLM(cfg)
    model.eval()

    toks_a = torch.tensor([[10, 20, 30]])
    toks_b = torch.tensor([[999, 10, 20, 30]])
    with torch.no_grad():
        out_a = model(toks_a)
        out_b = model(toks_b)

    assert out_a.shape == (1, 3, 32032)
    diff = (out_a[0, 1] - out_b[0, 2]).abs().max().item()
    assert diff > 1e-4


def test_mark_vl_forward_and_2d_rope():
    cfg = MarkVL2_2BConfig(n_layers=2)
    model = MarkVLForCausalLM(cfg)
    model.eval()

    img = torch.randn(1, 3, 224, 224)
    dummy_toks = torch.tensor([[1, 10, 11, 12]])
    with torch.no_grad():
        out_vl = model(dummy_toks, images=img)
        img_shifted = torch.roll(img, shifts=14, dims=2)
        out_vl_shifted = model(dummy_toks, images=img_shifted)

    assert out_vl.shape == (1, 4, 32000)
    diff = (out_vl - out_vl_shifted).abs().max().item()
    assert diff > 1e-4


def test_mark_general_forward():
    mg = MarkGeneral2_2BForConditionalGeneration(MarkGeneral2_2BConfig(n_layers=2))
    img = torch.randn(1, 3, 56, 56)
    ids = torch.tensor([[1, 10, 11, 11, 12, 50, 60]])
    res = mg(ids, images=img)
    assert "logits" in res
    assert res["logits"].shape[-1] == 32000


def test_gguf_deepseek2_export_and_integrity():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        cfg = MarkConfig.create_engineer_300m()
        cfg.n_layers = 1
        cfg.n_dense_layers = 0
        cfg.vocab_size = 128
        model = MarkModel(cfg)
        sd = model.state_dict()
        pt_file = tmp_path / "test.pt"
        gguf_file = tmp_path / "test.gguf"
        torch.save(sd, pt_file)

        export_to_deepseek2_gguf("302m", pt_file, gguf_file, "Test-302m")
        verify_gguf_integrity(gguf_file)


def test_epistemic_evaluation():
    # Prompt injection
    s_inj = evaluate_epistemic_state(prob_inject=0.6)
    assert s_inj["action"] == "BLOCK_INJECTION"

    # Low support -> Web
    s_web = evaluate_epistemic_state(support_score=0.2, web_already_searched=False)
    assert s_web["action"] == "ESCALATE_WEB_SEARCH"

    # Web searched but support still low -> Abstain
    s_abs = evaluate_epistemic_state(support_score=0.2, web_already_searched=True)
    assert s_abs["action"] == "ABSTAIN_UNKNOWN"

    # High entropy -> Abstain
    uniform_lp = [-2.0794] * 8
    s_ent = evaluate_epistemic_state(token_logprobs=uniform_lp, support_score=0.5)
    assert s_ent["action"] == "ABSTAIN_UNKNOWN"
