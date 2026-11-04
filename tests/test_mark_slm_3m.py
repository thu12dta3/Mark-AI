import pytest
import torch
from src.model.mark_slm_3m_arch import Mark3MConfig, Mark3MForCausalLM


def test_mark_slm_3m_parameter_ceiling():
    cfg = Mark3MConfig()
    model = Mark3MForCausalLM(cfg)
    total_params = sum(p.numel() for p in model.parameters())

    assert total_params < 3_000_000, f"Parameter ceiling violated: {total_params} >= 3,000,000"
    assert total_params == 2_885_376, f"Expected exact parameter count of 2,885,376, got {total_params}"


def test_mark_slm_3m_moe_intermediate_size_divisibility():
    cfg = Mark3MConfig()
    assert cfg.moe_intermediate_size == 128
    assert cfg.moe_intermediate_size % 32 == 0, f"moe_intermediate_size ({cfg.moe_intermediate_size}) must be divisible by block size 32"


def test_mark_slm_3m_forward_pass_clean():
    cfg = Mark3MConfig()
    model = Mark3MForCausalLM(cfg)
    model.eval()

    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), dtype=torch.long)

    with torch.no_grad():
        out = model(input_ids)
        logits = out.logits

    assert logits.shape == (batch_size, seq_len, cfg.vocab_size), f"Unexpected logits shape: {logits.shape}"
    assert not torch.isnan(logits).any(), "NaN detected in logits"
    assert not torch.isinf(logits).any(), "Inf detected in logits"


def test_mark_slm_3m_loss_calculation():
    cfg = Mark3MConfig()
    model = Mark3MForCausalLM(cfg)
    model.eval()

    batch_size = 2
    seq_len = 16
    input_ids = torch.randint(0, cfg.vocab_size, (batch_size, seq_len), dtype=torch.long)
    labels = input_ids.clone()

    out = model(input_ids, labels=labels)
    loss = out.loss

    assert loss is not None, "Loss should not be None when labels are provided"
    assert not torch.isnan(loss).any(), "NaN detected in loss"
    assert not torch.isinf(loss).any(), "Inf detected in loss"
    assert loss.item() > 0.0, f"Expected positive cross-entropy loss, got {loss.item()}"
