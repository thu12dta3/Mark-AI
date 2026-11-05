"""
Hugging Face Transformers Configuration for Mark-SLM-3M (< 3M Parameter Ceiling)
"""

from transformers.configuration_utils import PretrainedConfig


class Mark3MConfig(PretrainedConfig):
    model_type = "deepseek2"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 4096,
        hidden_size: int = 192,
        num_hidden_layers: int = 8,
        num_attention_heads: int = 6,
        num_key_value_heads: int = 6,
        intermediate_size: int = 160,
        q_lora_rank: int = 96,
        kv_lora_rank: int = 64,
        qk_nope_head_dim: int = 16,
        qk_rope_head_dim: int = 16,
        v_head_dim: int = 32,
        n_routed_experts: int = 4,
        num_experts_per_tok: int = 2,
        n_shared_experts: int = 1,
        moe_intermediate_size: int = 64,
        core_loops: int = 2,
        dense_layer_indices: list = None,
        max_position_embeddings: int = 4096,
        original_max_position_embeddings: int = 2048,
        rope_theta: float = 10000.0,
        rope_scaling_factor: float = 2.0,
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
        self.core_loops = core_loops
        self.dense_layer_indices = dense_layer_indices if dense_layer_indices is not None else [0, 7]
        self.max_position_embeddings = max_position_embeddings
        self.original_max_position_embeddings = original_max_position_embeddings
        self.rope_theta = rope_theta
        self.rope_scaling_factor = rope_scaling_factor
        self.rms_norm_eps = rms_norm_eps

    @property
    def dim(self) -> int:
        return self.hidden_size

    @property
    def n_layers(self) -> int:
        return self.num_hidden_layers

    @property
    def n_heads(self) -> int:
        return self.num_attention_heads

    @property
    def qk_head_dim(self) -> int:
        return self.qk_nope_head_dim + self.qk_rope_head_dim
