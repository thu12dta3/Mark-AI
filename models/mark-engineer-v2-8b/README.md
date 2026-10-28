---
language:
- en
- pt
license: agpl-3.0
tags:
- mark-ai
- systems-engineering
- linux-kernel
- rust
- cpp
- c
- reverse-engineering
- robotics
- electrical-engineering
- mla
- moe
- mtp
- gguf
- ollama
pipeline_tag: text-generation
inference: false
---

# Mark-Engineer-V2-8B: High-Density Cognitive Systems Architecture

Mark-Engineer-V2-8B is a specialized 8.01-billion-parameter Mixture-of-Experts (MoE) foundation model designed for low-level systems engineering, operating system kernels, distributed consensus, firmware, compiler toolchains, cybersecurity (AppSec P0/Red Team), and physical computing.

Engineered with Multi-Head Latent Attention (MLA), 24 routed experts plus 1 shared expert (Top-2 routing, ~1.63B active compute per token), and Multi-Token Prediction (MTP, Depth 1).

> **IMPORTANT DISCLOSURE: TRAINED BUT UNTESTED**  
> Mark-Engineer-V2-8B has completed pre-training with mathematical convergence verified (Final Loss: **1.9047** | NTP: 1.2952, MTP: 0.1905). However, this checkpoint has **not yet undergone live human qualitative evaluation, automated red-teaming, or standardized bench testing**. All stated capabilities represent architectural design specifications and curriculum objectives. Users are advised to run preliminary evaluations in sandboxed environments.

---

## Architectural Specifications

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Total Physical Parameters** | **8,005,905,408 (~8.01B)** | Total neural weights across dense and expert layers |
| **Active Parameters per Token** | **~1,626,500,000 (~1.63B)** | Compute footprint per token during forward pass |
| **Layers** | **32** | 2 Initial Dense Blocks + 30 DeepSeekMoE Blocks |
| **Hidden Dimension ($d_{model}$)** | **2,560** | Base model embedding dimension |
| **Attention Mechanism** | **MLA (Multi-Head Latent Attention)** | Decoupled RoPE with low-rank KV & Q compression |
| **Query Compression ($q_{lora}$)** | **1,024** | Compresses query states into latent representation |
| **Key-Value Compression ($kv_{lora}$)** | **512** | Compresses KV states into 512-dim latent vector |
| **RoPE Decoupled Dimension** | **32** | Positional embedding applied only to decoupled key/query |
| **MoE Routing Topology** | **1 Shared + 24 Routed Experts** | Top-2 routed experts + 1 universal shared expert |
| **Expert Dimension ($d_{expert}$)** | **1,254** | Intermediate dimension per expert block |
| **Multi-Token Prediction (MTP)** | **Depth 1 ($t+1, t+2$)** | Parallel dual-head prediction for fast speculative decoding |
| **Vocabulary Size** | **49,152 tokens** | Byte-Pair Encoding (BPE) tuned for code, math, and technical prose |
| **Context Window** | **8,192 tokens** | Native rotary positional context span |

---

## Expert Domain Specialization Matrix

The 24 routed experts are partitioned across specialized computational domains:

| Expert ID | Domain Descriptor | Focus Area |
| :---: | :--- | :--- |
| **E00** | `linux_kernel_core_mm` | Linux kernel memory management, page tables, slab/slub allocators, eBPF |
| **E01** | `embedded_systems_firmware_rtos` | Bare-metal C, FreeRTOS, Zephyr, ARM Cortex-M/R, memory-mapped I/O |
| **E02** | `electrical_engineering_circuits_pcb` | Circuit theory, power converters, analog/digital filter design, KiCad schematics |
| **E03** | `mechanical_engineering_physics_cad` | Statics, dynamics, stress analysis, thermodynamics, mechanical tolerances |
| **E04** | `robotics_kinematics_dynamics_ros2` | Forward/inverse kinematics, Denavit-Hartenberg, PID controllers, ROS 2, odometry |
| **E05** | `hardware_fpga_verilog_riscv` | Digital logic, RTL design, Verilog/VHDL, RISC-V ISA extensions |
| **E06** | `c_cpp_systems_memory_models` | C99/C11, C++20/23, RAII, concurrency atomics, lock-free rings, memory orderings |
| **E07** | `rust_systems_concurrency` | Unsafe Rust, pointer provenance, Tokio runtime, crossbeam channels, borrow checker |
| **E08** | `go_distributed_backends` | Goroutines, Go scheduler internals, sync pools, low-latency microservices |
| **E09** | `asm_binary_reverse_engineering` | x86_64, ARM64, RISC-V disassembly, ELF/PE parsing, Ghidra decompilation |
| **E10** | `cybersec_red_team_pwn` | Heap exploitation, tcache poisoning, ROP/JOP chains, ASLR/DEP mitigations |
| **E11** | `cybersec_appsec_crypto` | Constant-time cryptography, AES-GCM, TLS 1.3 protocol verification, threat models |
| **E12** | `software_architecture_full_system` | C4 blueprints, Domain-Driven Design (DDD), clean architecture, event sourcing |
| **E13** | `distributed_systems_consensus` | Raft, Multi-Paxos, Kafka partitions, 2PC/Saga distributed transactions |
| **E14** | `database_internals_storage` | LSM-trees, B+Trees, Write-Ahead Logs (WAL), MVCC engines, query planners |
| **E15** | `networking_protocols_kernel` | TCP/IP stack tuning, QUIC, HTTP/3, AF_XDP, zero-copy packet processing |
| **E16** | `compilers_interpreters_jit` | LLVM IR passes, AST transformations, register allocation, SSA form |
| **E17** | `enterprise_polyglot_tier1` | Modern Python, Java/Kotlin, C#/.NET, TypeScript, PostgreSQL/MySQL |
| **E18** | `cloud_native_devops_infra` | Kubernetes operators, Docker/OCI runtimes, Terraform, NixOS, Prometheus |
| **E19** | `vague_prompt_intent_disambiguation` | Implied context deduction, sensible engineering defaults, interactive clarification |
| **E20** | `epistemic_calibration_research` | Uncertainty assessment, zero-hallucination verification, fact-checking citations |
| **E21** | `top20_polyglot_linguistics` | Technical vocabulary and syntax across top 20 global languages (EN, PT, ES, etc.) |
| **E22** | `high_eq_senior_tech_lead_persona` | Constructive, articulate, and patient technical communication persona |
| **E23** | `formal_verification_debugging` | TLA+ specifications, Model Checking, Valgrind, AddressSanitizer, Linux perf profiling |

---

## Training Convergence

The model was trained on the consolidated Mark Ultra-Master Curriculum with multi-objective optimization:

$$\mathcal{L}_{\text{Total}} = \mathcal{L}_{\text{NTP}} + 0.3 \cdot \mathcal{L}_{\text{MTP}} + 10^{-4} \cdot \mathcal{L}_{\text{Router\_Z}}$$

- **Initial Loss:** 11.4500
- **Final Pre-training Loss:** **1.9047**
  - Next-Token Prediction (NTP) Loss: **1.2952**
  - Knowledge Distillation (KD) Loss: **0.4190**
  - Multi-Token Prediction (MTP) Loss: **0.1905**
  - Router Z-Loss: Stable ($< 1.2 \times 10^{-4}$)
- **Learning Rate Schedule:** Cosine decay with linear warmup ($1.2 \times 10^{-4} \rightarrow 1.0 \times 10^{-5}$)

---

## Quickstart: Plug and Play Deployment

### 1. Ollama (Recommended)

1. Download the quantized model file `mark_engineer_v2_8b_iq4_nl.gguf` from the Releases section.
2. Create a `Modelfile`:

```dockerfile
FROM ./mark_engineer_v2_8b_iq4_nl.gguf

TEMPLATE """<|im_start|>system
{{ .System }}<|im_end|>
<|im_start|>user
{{ .Prompt }}<|im_end|>
<|im_start|>assistant
<think>
"""

PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"
PARAMETER temperature 0.2
PARAMETER top_p 0.95

SYSTEM """Voce e o Mark Engineer V2 8B, um modelo de sistemas e engenharia de software de altissima capacidade."""
```

3. Register and run the model:

```bash
ollama create mark-engineer-v2-8b -f Modelfile
ollama run mark-engineer-v2-8b
```

---

### 2. LM Studio / llama.cpp

Load `mark_engineer_v2_8b_iq4_nl.gguf` directly into LM Studio or llama.cpp:

```bash
./llama-cli -m mark_engineer_v2_8b_iq4_nl.gguf -p "Implement a lock-free ring buffer in Rust using atomic pointers." -n 1024 --temp 0.2
```

---

### 3. PyTorch / Transformers

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "miceworks/Mark-Engineer-V2-8B",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("miceworks/Mark-Engineer-V2-8B")

inputs = tokenizer("<|im_start|>user\nExplain the slab allocator in the Linux kernel.<|im_end|>\n<|im_start|>assistant\n", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.2)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## License

GNU Affero General Public License v3.0 (AGPL-3.0). Free for open-source research and community deployment.
