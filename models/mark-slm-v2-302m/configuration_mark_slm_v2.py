"""
Hugging Face Transformers Configuration for Mark-SLM-V2-302M
"""

from transformers.configuration_utils import PretrainedConfig


class MarkSLMV2Config(PretrainedConfig):
    model_type = "mark_slm_v2"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 49152,
        hidden_size: int = 768,
        num_hidden_layers: int = 16,
        num_attention_heads: int = 12,
        num_key_value_heads: int = 4,
        q_lora_rank: int = 384,
        kv_lora_rank: int = 192,
        qk_nope_head_dim: int = 64,
        qk_rope_head_dim: int = 32,
        v_head_dim: int = 64,
        intermediate_size: int = 3072,
        n_dense_layers: int = 2,
        n_shared_experts: int = 1,
        n_routed_experts: int = 16,
        num_experts_per_tok: int = 2,
        moe_intermediate_size: int = 400,
        router_bias_update_rate: float = 1e-3,
        mtp_depth: int = 1,
        mtp_loss_weight: float = 0.3,
        max_position_embeddings: int = 4096,
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
