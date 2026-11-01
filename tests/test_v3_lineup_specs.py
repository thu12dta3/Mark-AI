import pytest
import torch
from src.model.mark_v3_arch import MarkV3Config, MarkV3ForCausalLM


def test_slm_v3_150m_parameter_ceiling():
    cfg = MarkV3Config.create_slm_v3_150m()
    model = MarkV3ForCausalLM(cfg)
    total = sum(p.numel() for p in model.parameters())
    assert total < 150_000_000, f"Violacao de teto: {total} >= 150000000"
    assert total >= 140_000_000, f"Subdimensionado: {total} < 140000000"


def test_v3_lineup_dimensioning():
    cfg_gen = MarkV3Config.create_general_v3_1_5b()
    cfg_eng = MarkV3Config.create_engineer_v3_1_5b()
    cfg_exp = MarkV3Config.create_experimental_v3_8b()

    with torch.device("meta"):
        m_gen = MarkV3ForCausalLM(cfg_gen)
        tot_gen = sum(p.numel() for p in m_gen.parameters())

        m_eng = MarkV3ForCausalLM(cfg_eng)
        tot_eng = sum(p.numel() for p in m_eng.parameters())

        m_exp = MarkV3ForCausalLM(cfg_exp)
        tot_exp = sum(p.numel() for p in m_exp.parameters())

    assert 1_475_000_000 <= tot_gen <= 1_550_000_000, f"General-V3-1.5B fora da faixa: {tot_gen}"
    assert 1_475_000_000 <= tot_eng <= 1_550_000_000, f"Engineer-V3-1.5B fora da faixa: {tot_eng}"
    assert tot_gen == tot_eng, "General-V3 e Engineer-V3 devem possuir contagem de parametros identica"
    assert 7_900_000_000 <= tot_exp <= 8_200_000_000, f"Experimental-V3-8B fora da faixa: {tot_exp}"


def test_v3_lineup_initial_forward():
    cfgs = [
        MarkV3Config.create_slm_v3_150m(),
        MarkV3Config.create_general_v3_1_5b(),
        MarkV3Config.create_engineer_v3_1_5b(),
        MarkV3Config.create_experimental_v3_8b(),
    ]

    batch_size = 1
    seq_len = 16
    eval_tokens = torch.randint(0, 1000, (batch_size, seq_len), dtype=torch.long)

    for cfg in cfgs:
        forward_cfg = MarkV3Config(
            model_name=cfg.model_name,
            vocab_size=cfg.vocab_size,
            hidden_size=cfg.hidden_size,
            num_hidden_layers=1,
            num_attention_heads=cfg.num_attention_heads,
            num_key_value_heads=cfg.num_key_value_heads,
            intermediate_size=cfg.intermediate_size,
            q_lora_rank=cfg.q_lora_rank,
            kv_lora_rank=cfg.kv_lora_rank,
            qk_nope_head_dim=cfg.qk_nope_head_dim,
            qk_rope_head_dim=cfg.qk_rope_head_dim,
            v_head_dim=cfg.v_head_dim,
            n_routed_experts=cfg.n_routed_experts,
            num_experts_per_tok=cfg.num_experts_per_tok,
            n_shared_experts=cfg.n_shared_experts,
            moe_intermediate_size=cfg.moe_intermediate_size,
            tie_word_embeddings=cfg.tie_word_embeddings,
            max_position_embeddings=cfg.max_position_embeddings,
            original_max_position_embeddings=cfg.original_max_position_embeddings,
            rope_theta=cfg.rope_theta,
            rope_scaling_factor=cfg.rope_scaling_factor,
        )
        model = MarkV3ForCausalLM(forward_cfg)
        model.eval()
        with torch.no_grad():
            logits = model(eval_tokens)

        assert logits.shape == (batch_size, seq_len, cfg.vocab_size), (
            f"Shape incorreto para {cfg.model_name}: {logits.shape}"
        )
        assert not torch.isnan(logits).any(), f"NaN detectado nos logits de {cfg.model_name}"
        assert not torch.isinf(logits).any(), f"Inf detectado nos logits de {cfg.model_name}"
