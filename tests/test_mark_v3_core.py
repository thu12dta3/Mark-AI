import tempfile
from pathlib import Path
import pytest
import torch
import numpy as np
import gguf
from src.model.mark_v3_arch import MarkV3Config, MarkV3MLA, MarkV3MoE, MarkV3Model, MarkV3ForCausalLM, get_walsh_hadamard_matrix
from export.convert_to_deepseek2_gguf import export_to_deepseek2_gguf, verify_gguf_integrity

def test_mark_v3_config():
    cfg = MarkV3Config()
    assert cfg.model_type == 'deepseek2'
    assert cfg.vocab_size == 32032
    assert cfg.hidden_size == 1152
    assert cfg.num_hidden_layers == 18
    assert cfg.num_attention_heads == 18
    assert cfg.q_lora_rank == 384
    assert cfg.kv_lora_rank == 256
    assert cfg.qk_nope_head_dim == 64
    assert cfg.qk_rope_head_dim == 64
    assert cfg.v_head_dim == 64
    assert cfg.n_routed_experts == 32
    assert cfg.num_experts_per_tok == 4
    assert cfg.n_shared_experts == 2
    assert cfg.moe_intermediate_size == 512
    assert cfg.tie_word_embeddings is True
    assert cfg.rms_norm_eps == 1e-06
    assert cfg.max_position_embeddings == 32768
    assert cfg.original_max_position_embeddings == 4096
    assert cfg.rope_theta == 10000.0
    assert cfg.rope_scaling_factor == 8.0

def test_mark_v3_mla_hadamard_offline_fusion():
    cfg = MarkV3Config(num_hidden_layers=2)
    mla = MarkV3MLA(cfg)
    mla.eval()
    h = get_walsh_hadamard_matrix(256)
    assert torch.allclose(h @ h.T, torch.eye(256), atol=1e-06)
    assert torch.allclose(h, h.T, atol=1e-06)
    bsz, seqlen = (2, 8)
    x = torch.randn(bsz, seqlen, cfg.hidden_size)
    cos = torch.randn(seqlen, cfg.qk_rope_head_dim // 2)
    sin = torch.randn(seqlen, cfg.qk_rope_head_dim // 2)
    with torch.no_grad():
        out_before = mla(x, cos, sin)
        mla.fuse_hadamard_weights_()
        out_after = mla(x, cos, sin)
    assert torch.allclose(out_before, out_after, atol=1e-05)
    assert torch.allclose(mla.kv_norm.weight, torch.ones_like(mla.kv_norm.weight))

def test_mark_v3_pb_moe_clans_and_soft_pinning():
    cfg = MarkV3Config(num_hidden_layers=2)
    moe = MarkV3MoE(cfg)
    moe.eval()
    bsz, seqlen = (2, 4)
    x = torch.randn(bsz, seqlen, cfg.hidden_size)
    with torch.no_grad():
        out_global = moe(x, clan_id=None)
        assert out_global.shape == (bsz, seqlen, cfg.hidden_size)
        for clan_id in (1, 2, 3, 4):
            out_clan = moe(x, clan_id=clan_id)
            assert out_clan.shape == (bsz, seqlen, cfg.hidden_size)
    moe.train()
    init_bias = moe.exp_probs_b.clone()
    _ = moe(x, clan_id=None)
    assert not torch.equal(init_bias, moe.exp_probs_b)

def test_mark_v3_lrd_recursive_depth():
    cfg = MarkV3Config(num_hidden_layers=18)
    model = MarkV3Model(cfg)
    model.eval()
    input_ids = torch.randint(0, cfg.vocab_size, (1, 6))
    with torch.no_grad():
        out_loop1 = model(input_ids, core_loops=1)
        out_loop2 = model(input_ids, core_loops=2)
    assert out_loop1.shape == (1, 6, cfg.hidden_size)
    assert out_loop2.shape == (1, 6, cfg.hidden_size)
    assert not torch.allclose(out_loop1, out_loop2, atol=0.0001)

def test_mark_v3_causal_lm_and_loss():
    cfg = MarkV3Config(num_hidden_layers=2)
    lm = MarkV3ForCausalLM(cfg)
    lm.eval()
    assert lm.lm_head.weight is lm.model.embed_tokens.weight
    input_ids = torch.tensor([[1, 10, 20, 30]])
    with torch.no_grad():
        logits = lm(input_ids, clan_id=1, core_loops=1)
    assert logits.shape == (1, 4, cfg.vocab_size)
    res = lm(input_ids, labels=input_ids, clan_id=2, core_loops=1)
    assert 'loss' in res
    assert res['loss'].item() > 0.0

def test_mark_v3_gguf_export():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        cfg = MarkV3Config(num_hidden_layers=1, vocab_size=256)
        model = MarkV3ForCausalLM(cfg)
        sd = model.state_dict()
        pt_file = tmp_path / 'v3.pt'
        gguf_file = tmp_path / 'v3.gguf'
        torch.save(sd, pt_file)
        export_to_deepseek2_gguf('v3-1.1b', pt_file, gguf_file, 'Mark-v3-Core-1.1B')
        verify_gguf_integrity(gguf_file)
        reader = gguf.GGUFReader(str(gguf_file))
        tensor_dict = {t.name: t for t in reader.tensors}
        assert 'token_embd.weight' in tensor_dict
        assert 'output_norm.weight' in tensor_dict
        assert 'output.weight' in tensor_dict
        assert 'blk.0.attn_norm.weight' in tensor_dict
        assert 'blk.0.attn_q_a.weight' in tensor_dict
        assert 'blk.0.attn_q_a_norm.weight' in tensor_dict
        assert 'blk.0.attn_q_b.weight' in tensor_dict
        assert 'blk.0.attn_kv_a_mqa.weight' in tensor_dict
        assert 'blk.0.attn_kv_a_norm.weight' in tensor_dict
        assert 'blk.0.attn_k_b.weight' in tensor_dict
        assert 'blk.0.attn_v_b.weight' in tensor_dict
        assert 'blk.0.attn_output.weight' in tensor_dict
        assert 'blk.0.ffn_norm.weight' in tensor_dict
        assert 'blk.0.ffn_gate_inp.weight' in tensor_dict
        assert 'blk.0.ffn_gate_shexp.weight' in tensor_dict
        assert 'blk.0.ffn_up_shexp.weight' in tensor_dict
        assert 'blk.0.ffn_down_shexp.weight' in tensor_dict
        assert 'blk.0.ffn_gate_exps.weight' in tensor_dict
        assert 'blk.0.ffn_up_exps.weight' in tensor_dict
        assert 'blk.0.ffn_down_exps.weight' in tensor_dict
        assert 'blk.0.exp_probs_b.bias' in tensor_dict
        assert np.allclose(tensor_dict['blk.0.attn_kv_a_norm.weight'].data, 1.0)
        assert tensor_dict['blk.0.attn_norm.weight'].data.dtype == np.float32
        assert tensor_dict['blk.0.ffn_norm.weight'].data.dtype == np.float32
        assert tensor_dict['output_norm.weight'].data.dtype == np.float32
        assert tensor_dict['blk.0.exp_probs_b.bias'].data.dtype == np.float32
        assert tensor_dict['blk.0.ffn_gate_exps.weight'].data.shape == (32, 512, 1152)
