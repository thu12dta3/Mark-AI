from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from .mark_arch import MarkConfig, MarkModel, PagedKVCache, RMSNorm

class ReasoningLevel(str, Enum):
    LEVEL_0_DIRECT = 'L0_DIRECT'
    LEVEL_1_ANALYTICAL = 'L1_ANALYTICAL'
    LEVEL_2_DEEP = 'L2_DEEP'

@dataclass
class MarkEngineerV2Config(MarkConfig):
    vocab_size: int = 49152
    dim: int = 2560
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int = 8
    use_mla: bool = True
    q_lora_rank: int = 1024
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 64
    qk_rope_head_dim: int = 32
    v_head_dim: int = 80
    n_dense_layers: int = 2
    moe_layer_freq: int = 1
    recursive_depth_loops: int = 2
    intermediate_size: int = 7680
    n_shared_experts: int = 1
    n_routed_experts: int = 24
    num_experts_per_tok: int = 2
    moe_intermediate_size: int = 1280
    router_bias_update_rate: float = 0.001
    mtp_depth: int = 1
    mtp_loss_weight: float = 0.3
    vision_enabled: bool = False
    voice_prosody_enabled: bool = False
    max_seq_len: int = 8192
    default_reasoning_level: ReasoningLevel = ReasoningLevel.LEVEL_1_ANALYTICAL
EXPERT_DOMAIN_REGISTRY: Dict[int, str] = {0: 'linux_kernel_core_mm', 1: 'embedded_systems_firmware_rtos', 2: 'electrical_engineering_circuits_pcb', 3: 'mechanical_engineering_physics_cad', 4: 'robotics_kinematics_dynamics_ros2', 5: 'hardware_fpga_verilog_riscv', 6: 'c_cpp_systems_memory_models', 7: 'rust_systems_concurrency', 8: 'go_distributed_backends', 9: 'asm_binary_reverse_engineering', 10: 'cybersec_red_team_pwn', 11: 'cybersec_appsec_crypto', 12: 'software_architecture_full_system', 13: 'distributed_systems_consensus', 14: 'database_internals_storage', 15: 'networking_protocols_kernel', 16: 'compilers_interpreters_jit', 17: 'enterprise_polyglot_tier1', 18: 'cloud_native_devops_infra', 19: 'vague_prompt_intent_disambiguation', 20: 'epistemic_calibration_research', 21: 'top20_polyglot_linguistics', 22: 'high_eq_senior_tech_lead_persona', 23: 'formal_verification_debugging'}

class MarkEngineerV2Model(MarkModel):

    def __init__(self, config: Optional[MarkEngineerV2Config]=None):
        if config is None:
            config = MarkEngineerV2Config()
        super().__init__(config)
        self.config: MarkEngineerV2Config = config

    def get_expert_domain(self, expert_id: int) -> str:
        return EXPERT_DOMAIN_REGISTRY.get(expert_id, 'general_systems_computation')

    def forward_with_reasoning_gate(self, input_ids: torch.Tensor, reasoning_level: ReasoningLevel=ReasoningLevel.LEVEL_1_ANALYTICAL, start_pos: int=0, kv_cache: Optional[PagedKVCache]=None) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, any]]:
        logits, mtp_logits = self.forward(input_ids, start_pos=start_pos, kv_cache=kv_cache, return_mtp=True)
        telemetry = {'reasoning_level': reasoning_level.value, 'total_params': self.config.count_parameters(), 'n_layers': self.config.n_layers, 'active_experts_per_tok': self.config.num_experts_per_tok, 'shared_experts': self.config.n_shared_experts}
        return (logits, mtp_logits, telemetry)
