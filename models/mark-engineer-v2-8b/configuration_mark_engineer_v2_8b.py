"""
Hugging Face Transformers Configuration for Mark-Engineer-V2-8B
"""

from transformers.configuration_utils import PretrainedConfig


class MarkEngineerV2_8BConfig(PretrainedConfig):
    model_type = "mark_engineer_v2_8b"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 49152,
        hidden_size: int = 2560,
        num_hidden_layers: int = 32,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        q_lora_rank: int = 1024,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 64,
        qk_rope_head_dim: int = 32,
        v_head_dim: int = 80,
        intermediate_size: int = 7680,
        n_dense_layers: int = 2,
        n_shared_experts: int = 1,
        n_routed_experts: int = 24,
        num_experts_per_tok: int = 2,
        moe_intermediate_size: int = 1280,
        router_bias_update_rate: float = 1e-3,
        mtp_depth: int = 1,
        mtp_loss_weight: float = 0.3,
        max_position_embeddings: int = 8192,
        rope_theta: float = 500000.0,
        rms_norm_eps: float = 1e-5,
        tie_word_embeddings: bool = True,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        pad_token_id: int = 0,
        **kwargs,
    ):
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.intermediate_size = intermediate_size
        self.n_dense_layers = n_dense_layers
        self.n_shared_experts = n_shared_experts
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.moe_intermediate_size = moe_intermediate_size
        self.router_bias_update_rate = router_bias_update_rate
        self.mtp_depth = mtp_depth
        self.mtp_loss_weight = mtp_loss_weight
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
