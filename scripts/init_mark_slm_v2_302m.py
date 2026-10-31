import json
import os
import sys
from pathlib import Path
import torch
from safetensors.torch import save_file
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model.mark_slm_v2_302m_arch import MarkSLMV2Config, MarkSLMV2Model
from src.model.configuration_mark_slm_v2 import MarkSLMV2Config as HFMarkSLMV2Config
from src.model.modeling_mark_slm_v2 import MarkSLMV2ForCausalLM

def main():
    print('[1/5] Initializing Mark-SLM-V2-302M native architecture...')
    config = MarkSLMV2Config()
    model = MarkSLMV2Model(config)
    model.eval()
    total_params = sum((p.numel() for p in model.parameters()))
    print(f'      Total physical parameters: {total_params:,} ({total_params / 1000000.0:.2f}M)')
    assert total_params == 302260224, f'Expected 302,260,224 params, got {total_params}'
    print('[2/5] Verifying forward pass & MTP head with dummy batch...')
    x = torch.randint(0, config.vocab_size, (2, 64))
    with torch.no_grad():
        logits, mtp_logits = model(x, return_mtp=True)
    print(f'      Logits shape: {logits.shape}')
    print(f'      MTP Logits shape: {mtp_logits.shape}')
    assert logits.shape == (2, 64, config.vocab_size)
    assert mtp_logits.shape == (2, 63, config.vocab_size)
    ssd_ckpt_dir = Path('/mnt/ssd_1/mark ai/checkpoints')
    ssd_ckpt_dir.mkdir(parents=True, exist_ok=True)
    pt_path = ssd_ckpt_dir / 'mark_slm_v2_302m_init.pt'
    print(f'[3/5] Saving initial untrained PyTorch checkpoint to: {pt_path}...')
    torch.save({'model_state_dict': model.state_dict(), 'config': config.__dict__, 'total_params': total_params, 'architecture': 'Mark-SLM-V2-302M', 'status': 'untrained_init'}, pt_path)
    print('      PyTorch checkpoint successfully saved.')
    print('[4/5] Testing Hugging Face Transformers CausalLM compatibility...')
    hf_config = HFMarkSLMV2Config(vocab_size=config.vocab_size, hidden_size=config.dim, num_hidden_layers=config.n_layers, num_attention_heads=config.n_heads, num_key_value_heads=config.n_kv_heads, q_lora_rank=config.q_lora_rank, kv_lora_rank=config.kv_lora_rank, qk_nope_head_dim=config.qk_nope_head_dim, qk_rope_head_dim=config.qk_rope_head_dim, v_head_dim=config.v_head_dim, intermediate_size=config.intermediate_size, n_dense_layers=config.n_dense_layers, n_shared_experts=config.n_shared_experts, n_routed_experts=config.n_routed_experts, num_experts_per_tok=config.num_experts_per_tok, moe_intermediate_size=config.moe_intermediate_size, router_bias_update_rate=config.router_bias_update_rate, mtp_depth=config.mtp_depth, mtp_loss_weight=config.mtp_loss_weight, max_position_embeddings=config.max_seq_len, rope_theta=config.rope_theta, rms_norm_eps=config.rms_norm_eps, tie_word_embeddings=config.tie_word_embeddings)
    hf_model = MarkSLMV2ForCausalLM(hf_config)
    hf_params = sum((p.numel() for p in hf_model.parameters()))
    print(f'      HF CausalLM parameters: {hf_params:,} ({hf_params / 1000000.0:.2f}M)')
    labels = x.clone()
    hf_out = hf_model(input_ids=x, labels=labels)
    print(f'      Initial Cross-Entropy Loss: {hf_out.loss.item():.4f} (expected ~{torch.log(torch.tensor(49152.0)).item():.2f})')
    hf_dir = Path(__file__).resolve().parent.parent / 'models' / 'mark-slm-v2-302m'
    hf_dir.mkdir(parents=True, exist_ok=True)
    print(f'[5/5] Exporting Hugging Face weights and config to {hf_dir}...')
    hf_model.save_pretrained(hf_dir, safe_serialization=True)
    config_path = hf_dir / 'config.json'
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg_data = json.load(f)
    cfg_data['auto_map'] = {'AutoConfig': 'configuration_mark_slm_v2.MarkSLMV2Config', 'AutoModelForCausalLM': 'modeling_mark_slm_v2.MarkSLMV2ForCausalLM'}
    cfg_data['architectures'] = ['MarkSLMV2ForCausalLM']
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(cfg_data, f, indent=2)
    print('      Model safetensors and config.json exported successfully.')
    print('\nInitialization Complete:')
    print(f'- Physical Parameters: 302,260,224 (302.26M)')
    print(f'- Active Compute: ~121.61M per token (MoE 16+1, Top-2)')
    print(f'- Native Checkpoint: {pt_path}')
    print(f'- Hugging Face Repo: {hf_dir}')
if __name__ == '__main__':
    main()
