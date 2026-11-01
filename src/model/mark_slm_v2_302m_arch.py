from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from .mark_arch import DeepSeekMLAAttention, DeepSeekMoELayer, DeepSeekMTPModule, MarkConfig, MarkModel, MarkSwiGLUFFN, MarkTransformerBlock, PagedKVCache, RMSNorm, precompute_freqs_cis

class ReasoningLevel(str, Enum):
    LEVEL_0_DIRECT = 'L0_DIRECT'
    LEVEL_1_ANALYTICAL = 'L1_ANALYTICAL'
    LEVEL_2_DEEP = 'L2_DEEP'

@dataclass
class MarkSLMV2Config(MarkConfig):
    vocab_size: int = 49152
    dim: int = 768
    n_layers: int = 16
    n_heads: int = 12
    n_kv_heads: int = 4
    use_mla: bool = True
    q_lora_rank: int = 384
    kv_lora_rank: int = 192
    qk_nope_head_dim: int = 64
    qk_rope_head_dim: int = 32
    v_head_dim: int = 64
    kv_cache_quant: str = 'IQ4_NL'
    n_dense_layers: int = 2
    moe_layer_freq: int = 1
    recursive_depth_loops: int = 1
    intermediate_size: int = 3072
    n_shared_experts: int = 1
    n_routed_experts: int = 16
    num_experts_per_tok: int = 2
    moe_intermediate_size: int = 384
    router_bias_update_rate: float = 0.001
    mtp_depth: int = 1
    mtp_loss_weight: float = 0.3
    vision_enabled: bool = False
    voice_prosody_enabled: bool = False
    max_seq_len: int = 4096
    rope_theta: float = 500000.0
    rope_scaling_factor: float = 1.0
    rms_norm_eps: float = 1e-05
    dropout: float = 0.0
    tie_word_embeddings: bool = True
    default_reasoning_level: ReasoningLevel = ReasoningLevel.LEVEL_1_ANALYTICAL
SLM_EXPERT_DOMAIN_REGISTRY: Dict[int, str] = {0: 'arithmetic_integer_word_problems', 1: 'arithmetic_fractions_percentages', 2: 'symbolic_logic_transitive_relations', 3: 'mathematical_geometry_measurement', 4: 'physical_commonsense_mechanics', 5: 'physical_commonsense_materials', 6: 'elementary_science_astronomy_earth', 7: 'elementary_science_biology_ecosystems', 8: 'middle_school_physics_forces', 9: 'middle_school_chemistry_states', 10: 'commonsense_narrative_continuations', 11: 'commonsense_procedural_instructions', 12: 'algorithmic_code_python_logic', 13: 'formal_reasoning_syllogisms', 14: 'linguistic_syntax_orthography', 15: 'epistemic_uncertainty_verification'}

class MarkSLMV2Model(MarkModel):

    def __init__(self, config: Optional[MarkSLMV2Config]=None):
        if config is None:
            config = MarkSLMV2Config()
        super().__init__(config)
        self.config: MarkSLMV2Config = config

    def get_expert_domain(self, expert_id: int) -> str:
        return SLM_EXPERT_DOMAIN_REGISTRY.get(expert_id, 'general_cognition')

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor]=None, kv_cache: Optional[PagedKVCache]=None, start_pos: int=0, return_mtp: bool=False) -> Union[torch.Tensor, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        return super().forward(input_ids=input_ids, attention_mask=attention_mask, kv_cache=kv_cache, start_pos=start_pos, return_mtp=return_mtp)

    def forward_with_reasoning_gate(self, input_ids: torch.Tensor, reasoning_level: ReasoningLevel=ReasoningLevel.LEVEL_1_ANALYTICAL, start_pos: int=0, kv_cache: Optional[PagedKVCache]=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, any]]:
        logits, mtp_logits = self.forward(input_ids, start_pos=start_pos, kv_cache=kv_cache, return_mtp=True)
        telemetry = {'reasoning_level': reasoning_level.value, 'total_params': self.config.count_parameters(), 'n_layers': self.config.n_layers, 'active_experts_per_tok': self.config.num_experts_per_tok, 'shared_experts': self.config.n_shared_experts, 'active_params_per_tok': 121610000}
        return (logits, mtp_logits, telemetry)
