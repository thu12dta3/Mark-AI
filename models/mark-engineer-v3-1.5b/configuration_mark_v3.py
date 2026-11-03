"""
Hugging Face Transformers Configuration for Mark-Engineer-V3-1.5B
Sparse Mixture-of-Experts (MoE) Architecture with Multi-Head Latent Attention (MLA).
Total parameters: 1,520,496,960 (405,729,600 active per token).
"""

from transformers.configuration_utils import PretrainedConfig


class MarkV3Config(PretrainedConfig):
    model_type = "deepseek2"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 32032,
        hidden_size: int = 1280,
        num_hidden_layers: int = 18,
        num_attention_heads: int = 20,
        num_key_value_heads: int = 20,
        intermediate_size: int = 3456,
        q_lora_rank: int = 512,
        kv_lora_rank: int = 256,
        qk_nope_head_dim: int = 64,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 64,
        n_routed_experts: int = 32,
        num_experts_per_tok: int = 4,
        n_shared_experts: int = 2,
        moe_intermediate_size: int = 576,
        max_position_embeddings: int = 32768,
        original_max_position_embeddings: int = 4096,
        rope_theta: float = 10000.0,
        rope_scaling_factor: float = 8.0,
        rms_norm_eps: float = 1e-6,
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
        self.intermediate_size = intermediate_size
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.n_routed_experts = n_routed_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.n_shared_experts = n_shared_experts
        self.moe_intermediate_size = moe_intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.original_max_position_embeddings = original_max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling_factor = rope_scaling_factor
        self.rms_norm_eps = rms_norm_eps
