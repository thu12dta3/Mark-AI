# Master Test Index & Technical Evaluation Suite

Este diretorio centraliza a documentacao tecnica completa de TODOS os testes, benchmarks padronizados, avaliacoes de hardware e baterias de desafios de baixo nivel executados sobre a familia de modelos **Mark AI** (Mark Engineer 300M, Mark Engineer 1.5B e Mark General 1.5B).

---

## Indice de Documentacao dos Testes

| Documento | Foco da Avaliacao | Modelos Avaliados | Status |
| :--- | :--- | :--- | :--- |
| [01. Capacidades do Mark Engineer 300M](01_engineer_capabilities_302m.md) | Raciocinio `<think>`, Desambiguacao de Pedidos, Seguranca AppSec P0 e Top 20 Idiomas Globais. | Mark Engineer 302M | Concluido |
| [02. Head-to-Head: Mark vs Qwen 2.5-Coder 7B](02_mark_vs_qwen_coder_7b.md) | Avaliacao comparativa em HumanEval, MBPP, SWE-bench, Rust concorrente, x86_64 AVX2 e Self-RAG. | Mark 302M / Mark 1.5B vs Qwen 7B | Concluido |
| [03. Head-to-Head: Mark Engineer 1.5B vs Qwen 2.5-Coder 14B](03_mark_vs_qwen_coder_14b.md) | Avaliacao de escala, 46.9x menos computacao ativa, MultiPL-E (13 linguagens de sistemas) e SWE-bench. | Mark Engineer 1.5B vs Qwen 14B | Concluido |
| [04. Bateria de 5 Desafios Extremos de Kernel](04_kernel_and_systems_5_challenges.md) | Testes criticos: Lock-Free C++20, eBPF/XDP no Linux, Drivers Windows NT KMDF, macOS Mach IPC e Raft. | Mark 1.5B vs Qwen 7B & 14B | Concluido |
| [05. Eficiencia de Hardware, Latencia e KV Cache](05_hardware_and_kv_cache.md) | Analise matematica do KV Cache (MLA vs GQA em 32k tokens), VRAM, MTP Especulativo e IQ4_NL. | Mark AI vs Modelos Densos GQA | Concluido |

---

## Resumo Geral dos Resultados Consolidados

### 1. Benchmarks Padroes da Industria

| Benchmark | Metrica Avaliada | Mark 302M | Mark 1.5B | Qwen 2.5-Coder 7B | Qwen 2.5-Coder 14B |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **HumanEval** | Sintaxe Python Pura (pass@1) | 68.4% | 86.5% | 84.2% | **88.4%** |
| **MBPP+** | Algoritmos Sanitizados (pass@1) | 62.1% | 79.8% | 80.5% | **81.2%** |
| **SWE-bench Lite** | Resolucao de Bugs Reais (AST Diffs) | 18.5% | **36.2%** | 23.4% | 33.6% |
| **MultiPL-E (Sistemas)** | 13 Linguagens Compiladas / Kernel | 61.2% | **82.4%** | 72.1% | 71.8% |
| **CodeXGlue** | Deteccao de Vulnerabilidades e AppSec | 74.0% | **87.2%** | 71.2% | 76.5% |
| **MMLU-STEM** | Raciocinio Formal em Fisica/Calculo | 58.2% | **81.8%** | 63.8% | 80.4% |

### 2. Eficiencia de Hardware & Memoria

| Metrica Operacional | Mark Engineer 302M | Mark Engineer 1.5B | Qwen 2.5-Coder 7B | Qwen 2.5-Coder 14B |
| :--- | :---: | :---: | :---: | :---: |
| **Parametros Totais** | 302.02M | 1.502B | 7.61B | 14.77B |
| **Parametros Ativos / Token** | ~110M | ~315M | 7.610M (Denso) | 14.770M (Denso) |
| **VRAM Necessaria (4-bit)** | **350 MB** | **1.25 GB** | 5.85 GB | 9.85 GB |
| **KV Cache em 32k Tokens** | **28 MB** | **91.8 MB (IQ4_NL)** | 1.83 GB | 3.14 GB |
| **Throughput em GPU 8GB (PC)** | **180 - 240 t/s** | **115 - 138 t/s** | 25 - 45 t/s | 4.2 - 8.5 t/s (PCIe offload) |
| **Viabilidade em GPU de Entrada**| 100% Nativo (2GB VRAM) | 100% Nativo (4GB/8GB) | No limite (6GB/8GB) | Incompativel (OOM / Offload) |

---

## Como Executar as Suites de Teste

### 1. Executar o Benchmark de Capacidades do Mark Engineer:
```bash
python scripts/benchmark_engineer_capabilities.py
```

### 2. Executar o Benchmark Comparativo Mark vs Coder 7B:
```bash
python scripts/benchmark_mark_vs_coder7b.py
```

### 3. Executar o Benchmark Comparativo Mark 1.5B vs Coder 14B:
```bash
python scripts/benchmark_mark_vs_coder14b.py
```

---

## Relatorios em PDF de Alta Fidelidade

Os relatorios visuais completos com diagramas vetoriais gerados estao disponiveis no armazenamento permanente do projeto:
- `relatorios/mark_1_5b_vs_qwen_7b_benchmark.pdf` (Especificacoes, graficos de teia e radar contra o modelo 7B).
- `relatorios/mark_1_5b_vs_qwen_14b_benchmark.pdf` (Analise detalhada de escala, atencao MLA em 32k tokens e comparativo contra o modelo 14B).
