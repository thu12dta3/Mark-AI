# Mark General 1.5B - Preserved Baseline (10 Cognition Domains & STEM)
==============================================================================
Model Name: Mark General 1.5B
Architecture: Multi-Head Latent Attention (MLA) + DeepSeekMoE (16+1) + Epistemic Self-RAG
Parameters: 1,501,680,656 (1.502B) Physical | ~315,000,000 Active per Token
Status: Preserved Foundational Baseline (Permanently Archived)
Convergence Loss: 2.8743 on NVIDIA A100-SXM4-80GB

Strict Constraint: Zero emojis across all documentation and logs.

---

## 1. Executive Summary & Preservation Rationale

**Mark General 1.5B** is the preserved foundational generalist checkpoint of the Mark AI project. While `Mark Engineer 1.5B` was subsequently fine-tuned for kernel programming and systems engineering, `Mark General 1.5B` preserves the pure, unperturbed STEM, epistemic, and scientific reasoning foundation.

### Preservation Guarantee
This checkpoint is permanently archived in the repository. Its weights, parameters, and tokenizer definitions will never be overwritten, modified, or degraded by downstream engineering specializations.

---

## 2. The 10 Cognitive & Scientific Specialization Domains

Mark General 1.5B was trained across a master multi-domain corpus spanning rigorous academic, mathematical, and epistemic disciplines:

### 2.1 Quantum Mechanics & Condensed Matter Physics
- Hamiltonian mechanics, Schrödinger equation boundary problems, and Dirac bra-ket notation.
- Semiconductor bandgap physics, Fermi-Dirac distributions, and phonon dispersion relations.

### 2.2 Fluid Dynamics & Ballistic Aerodynamics
- Incompressible and compressible Navier-Stokes approximations.
- Supersonic shock wave formation, boundary layer separation, and drag coefficients under G1 vs G7 ballistic projectile standards.

### 2.3 Physical, Organic & Computational Chemistry
- Chemical reaction kinetics, Eyring-Polanyi transition-state rate theory, and Arrhenius activation energy.
- Organic synthesis retrosynthesis pathways, stereocenter inversions, and molecular orbital resonance.

### 2.4 Clinical Pharmacology & Renal Hemodynamics
- Pharmacokinetic/Pharmacodynamic (PK/PD) modeling and two-compartment ADME clearance calculations.
- Tubuloglomerular feedback mechanisms, sodium-glucose cotransporter-2 (SGLT2) inhibitor hemodynamics, and ACE/ARB renal protection.

### 2.5 Pure Mathematics & Formal Logic
- Abstract algebra (groups, rings, fields), topology, and differential geometry on Riemannian manifolds.
- First-order predicate calculus, Gödelian incompleteness consequences, and lambda calculus reduction.

### 2.6 Distributed Systems & Consensus Theory
- Formal proofs of Paxos, Raft, and Byzantine Fault Tolerant (BFT) consensus algorithms.
- CAP theorem trade-offs, vector clocks, linearizability vs sequential consistency, and network partition recovery.

### 2.7 Epistemology & Scientific Methodology
- Bayesian belief updating, probabilistic risk estimation, and prior sensitivity analysis.
- Popperian demarcation criteria, falsifiability metrics, and epistemic self-calibration.

### 2.8 Electrical Engineering & RF Signal Processing
- Transfer function derivations for active Butterworth/Chebyshev filter topologies.
- Discrete Fourier Transform (DFT), transmission line characteristic impedance ($Z_0$), and Smith chart matching.

### 2.9 Macroeconomics & Quantitative Finance
- Dynamic Stochastic General Equilibrium (DSGE) modeling, central bank Taylor rules, and IS-LM framework.
- Black-Scholes-Merton option pricing partial differential equations and volatility surface modeling.

### 2.10 Statutory Interpretation & Legal Engineering
- Statutory construction cannons, contractual consideration doctrine, and fiduciary standard analysis.
- Intellectual property licensing invariants (GPL, Apache, MIT, AGPL) and patent claim scope analysis.

---

## 3. Epistemic Self-RAG Protocol & Hallucination Suppression

Mark General 1.5B features built-in epistemic tokens embedded directly in its vocabulary:
- `[UNKNOWN]` (Token ID 14): Emitted when a question asks for non-existent, unverifiable, or contradictory factual claims.
- `[WEB_SEARCH]` (Token ID 15): Emitted when answering requires live external real-time data or knowledge beyond its training cut-off.

Rather than hallucinating plausible-sounding falsehoods or fabricating citations, the model dynamically routes uncertain tokens to its epistemic bounds.

---

## 4. Training Convergence Telemetry

- **Compute Cluster:** NVIDIA A100-SXM4-80GB (Ampere Architecture, 80GB HBM2e at 2.0 TB/s).
- **Optimization:** AdamW ($\beta_1 = 0.9$, $\beta_2 = 0.95$, weight decay $0.1$, cosine learning rate schedule).
- **Initial Loss:** `11.0542`
- **Converged Final Loss:** `2.8743` (Asymptotic convergence across all 10 domains).
- **Gradient Steps:** 600 steps under mixed precision BF16.

---

## 5. Technical Specifications

| Property | Value |
| :--- | :--- |
| **Physical Parameter Count** | 1,501,680,656 (1.502B) |
| **Active FLOPs per Token** | ~315,000,000 (20.9% of Total) |
| **Hidden Dimension ($d$)** | 1,280 |
| **Transformer Layers** | 14 Layers |
| **Multi-Head Latent Attention** | $c_{KV} = 320$, $q_{LoRA} = 640$, $d_{RoPE} = 32$, $d_{NoPE} = 64$ |
| **DeepSeekMoE Grid** | 16 Routed Experts + 1 Universal Shared Expert |
| **Active Experts** | Top-2 Routed Experts + Shared Expert 00 per token |
| **Multi-Token Prediction (MTP)** | Depth $k=2$ |
| **Quantization** | Native IQ4_NL (4-bit non-linear lookup) |
| **On-Disk Checkpoint Size** | 823.84 MB GGUF |
| **Runtime VRAM Footprint** | ~1.25 GB on NVIDIA GeForce RTX 2060 Super (Leaving > 6.7 GB Free) |
| **Throughput** | 120.0 to 135.0 tokens/second |

---

## 6. Plug-and-Play Quickstart

### 6.1 Ollama
```bash
cd models/mark-general-1.5b
ollama create mark-general-1.5b -f Modelfile
ollama run mark-general-1.5b "Deduce the Eyring-Polanyi transition-state equation from statistical thermodynamics."
```

### 6.2 LM Studio
```bash
# 1. Download compiled GGUF:
mkdir -p ~/.lmstudio/models/Mark-AI/mark-general-1.5b
curl -L -o ~/.lmstudio/models/Mark-AI/mark-general-1.5b/mark_general_1_5b.gguf \
  https://github.com/thu12dta3/Mark-AI/releases/download/v1.0.0-general-1.5b/mark_general_1_5b_iq4_nl.gguf

# 2. Select Mark-AI / mark-general-1.5b in LM Studio.
```

### 6.3 llama.cpp CLI
```bash
./llama-cli \
  -m ~/.lmstudio/models/Mark-AI/mark-general-1.5b/mark_general_1_5b.gguf \
  -p "User: Explain the aerodynamic drag difference between G1 and G7 ballistic standards.\nAssistant: <think>" \
  -n 512
```

### 6.4 Automated Python Downloader
```bash
python scripts/download_model.py --model general-1.5b
```
