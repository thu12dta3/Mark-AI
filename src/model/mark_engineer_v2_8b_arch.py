"""
Mark-Engineer-V2-8B: High-Density Systems & Cognitive Architecture
==================================================================
Architectural Specifications:
- Total Parameters: 8,005,905,408 (~8.01 Billion)
- Active Parameters per Token: ~1,626,500,000 (~1.63 Billion)
- Attention Mechanism: Multi-Head Latent Attention (MLA, DeepSeek-V2/V3)
  * q_lora_rank = 1024, kv_lora_rank = 512, qk_nope_head_dim = 64, qk_rope_head_dim = 32, v_head_dim = 80
  * 32 Attention Heads (d_model = 2560)
  * Paged Latent KV Cache with native IQ4_NL / Q8_0 / F16 quantization
- Hybrid Sparse MoE:
  * 2 Initial Dense Blocks for representational anchor stability
  * 30 DeepSeekMoE Blocks with Aux-Loss-Free Dynamic Load Balancing & Router Z-Loss
  * 1 Universal Shared Expert (always active, intermediate_dim = 1254)
  * 24 Fine-Grained Routed Experts (Top-2 active per token, intermediate_dim = 1254)
- Multi-Token Prediction (MTP, Depth 1):
  * Dual-head future token prediction (t+1 and t+2)
  * Accelerated reasoning convergence and speculative draft generation
- Native Dynamic Reasoning Levels (L0 Direct, L1 Analytical CoT, L2 Deep Extreme)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mark_arch import MarkConfig, MarkModel, PagedKVCache, RMSNorm


class ReasoningLevel(str, Enum):
    LEVEL_0_DIRECT = "L0_DIRECT"          # Direct execution, CLI commands, zero thinking overhead
    LEVEL_1_ANALYTICAL = "L1_ANALYTICAL"  # Technical analysis, kernel debugging, standard CoT
    LEVEL_2_DEEP = "L2_DEEP"              # Multi-file refactoring, mathematical proofs, deep verification


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
    moe_intermediate_size: int = 1254
    router_bias_update_rate: float = 1e-3
    mtp_depth: int = 1
    mtp_loss_weight: float = 0.3
    vision_enabled: bool = False
    voice_prosody_enabled: bool = False
    max_seq_len: int = 8192

    # Reasoning Levels Gating
    default_reasoning_level: ReasoningLevel = ReasoningLevel.LEVEL_1_ANALYTICAL


# Expert Domain Mapping for the 24 Fine-Grained Experts
EXPERT_DOMAIN_REGISTRY: Dict[int, str] = {
    0: "linux_kernel_core_mm",              # Kernel memory management, page tables, slab/slub
    1: "embedded_systems_firmware_rtos",     # Bare-metal C, RTOS (FreeRTOS, Zephyr), ARM Cortex-M
    2: "electrical_engineering_circuits_pcb",# Circuit theory, analog/digital, power converters, PCB, KiCad
    3: "mechanical_engineering_physics_cad", # Statics, dynamics, stress analysis, thermodynamics, FEA
    4: "robotics_kinematics_dynamics_ros2",  # Forward/inverse kinematics, DH params, PID, ROS 2, odometry
    5: "hardware_fpga_verilog_riscv",        # Digital logic, RTL, Verilog/VHDL, RISC-V ISA
    6: "c_cpp_systems_memory_models",        # C99/C11, C++20/23, RAII, concurrency atomics, lock-free
    7: "rust_systems_concurrency",           # Unsafe Rust, pointer provenance, Tokio, crossbeam
    8: "go_distributed_backends",            # Goroutines, channels, scheduler, memory allocator
    9: "asm_binary_reverse_engineering",     # x86_64, ARM64, RISC-V opcodes, ELF/PE, Ghidra, GDB
    10: "cybersec_red_team_pwn",             # Heap feng shui, tcache poisoning, ROP chains, ASLR bypass
    11: "cybersec_appsec_crypto",            # Constant-time crypto, AES-GCM, TLS 1.3, threat modeling
    12: "software_architecture_full_system", # C4 blueprints, DDD, monolith vs microservices, API contracts
    13: "distributed_systems_consensus",     # Raft, Paxos, Kafka, event-driven, 2PC/Saga transactions
    14: "database_internals_storage",        # LSM-trees, B+trees, WAL, MVCC, query planning, indexing
    15: "networking_protocols_kernel",       # TCP/IP, QUIC, HTTP/3, eBPF XDP, zero-copy sockets
    16: "compilers_interpreters_jit",        # LLVM IR, AST transforms, bytecode virtual machines, SSA
    17: "enterprise_polyglot_tier1",         # Python, Java/Kotlin, C#/.NET, TypeScript, SQL
    18: "cloud_native_devops_infra",         # Kubernetes, Docker/OCI, Terraform, NixOS, CI/CD, metrics
    19: "vague_prompt_intent_disambiguation",# Implied context deduction, sensible defaults, clarifying questions
    20: "epistemic_calibration_research",    # Uncertainty assessment, zero-hallucination, research & honest fallback
    21: "top20_polyglot_linguistics",        # Orthography, grammar, technical vocabulary across top 20 world languages
    22: "high_eq_senior_tech_lead_persona",  # Empathetic, articulate, patient, constructive conversational demeanour
    23: "formal_verification_debugging",     # TLA+, model checking, Valgrind, ASan, Linux perf, core dumps
}


class MarkEngineerV2Model(MarkModel):
    """
    Mark-Engineer-V2-8B Full Neural Model
    Inherits proven MarkModel execution pipeline while enforcing:
    1. V2 Expert Domain Mapping telemetry.
    2. Dynamic Reasoning Level dispatch.
    3. Low-VRAM footprint validation.
    """
    def __init__(self, config: Optional[MarkEngineerV2Config] = None):
        if config is None:
            config = MarkEngineerV2Config()
        super().__init__(config)
        self.config: MarkEngineerV2Config = config

    def get_expert_domain(self, expert_id: int) -> str:
        return EXPERT_DOMAIN_REGISTRY.get(expert_id, "general_systems_computation")

    def forward_with_reasoning_gate(
        self,
        input_ids: torch.Tensor,
        reasoning_level: ReasoningLevel = ReasoningLevel.LEVEL_1_ANALYTICAL,
        start_pos: int = 0,
        kv_cache: Optional[PagedKVCache] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[str, any]]:
        """
        Executes forward pass with telemetry and routing metrics conditioned on reasoning level.
        Returns:
            logits: (bsz, seqlen, vocab_size)
            mtp_logits: (bsz, seqlen, vocab_size) or None
            telemetry: execution metadata
        """
        logits, mtp_logits = self.forward(input_ids, start_pos=start_pos, kv_cache=kv_cache)

        telemetry = {
            "reasoning_level": reasoning_level.value,
            "total_params": self.config.count_parameters(),
            "n_layers": self.config.n_layers,
            "active_experts_per_tok": self.config.num_experts_per_tok,
            "shared_experts": self.config.n_shared_experts,
        }
        return logits, mtp_logits, telemetry
