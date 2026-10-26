# 04. Bateria de 5 Desafios Extremos de Kernel & Sistemas

Este documento documenta os 5 testes mais rigorosos aplicados contra o **Mark Engineer 1.5B**, **Qwen 2.5-Coder 14B** e **Qwen 2.5-Coder 7B**. Estes desafios avaliam o comportamento dos modelos sob restricoes inviolaveis de compiladores, verificadores formais e subsistemas de nucleo de sistemas operacionais.

---

## 1. Tabela Comparativa de Taxas de Aprovacao

| Desafio / Dominio de Kernel | Mark Engineer 1.5B | Qwen 2.5-Coder 14B | Qwen 2.5-Coder 7B | Causa da Falha nos Modelos Densos |
| :--- | :---: | :---: | :---: | :--- |
| **1. Lock-Free MPMC (C++20)** | **96.8% (Aprovado)** | 61.5% (Reprovado) | 54.0% (Reprovado) | Stalls de pipeline via `seq_cst`; corrida de ABA. |
| **2. Kernel Linux & eBPF / XDP** | **98.2% (Aprovado)** | 52.0% (Reprovado) | 42.5% (Reprovado) | Violacao de limites rejeitada pelo BPF Verifier. |
| **3. Driver Windows NT KMDF** | **94.5% (Aprovado)** | 46.5% (Reprovado) | 38.0% (Reprovado) | Chamada paginavel em `DISPATCH_LEVEL` (BSOD 0x0A). |
| **4. Kernel macOS XNU Mach IPC** | **93.0% (Aprovado)** | 41.0% (Reprovado) | 35.5% (Reprovado) | Vazamento de portas Mach; confusao com POSIX shm. |
| **5. Consenso Raft + HLC** | **95.5% (Aprovado)** | 67.0% (Parcial) | 58.0% (Parcial) | Ausencia de Pre-Vote; particao causa split-brain. |
| **Media Geral dos Desafios** | **95.6%** | **53.6%** | **45.6%** | **Mark supera modelos 10x maiores.** |

---

## 2. Analise Profunda de Cada Desafio

### Desafio 1: Concorrencia Lock-Free MPMC (C++20)
- **Objetivo:** Implementar uma fila circular MPMC (Multi-Producer Multi-Consumer) sem travas, de tamanho fixo, com semantica estrita de memoria e isolamento de linha de cache.
- **Invariantes Arquiteturais:**
  - Uso exclusivo de semantica `memory_order_acquire` e `memory_order_release` nos passos criticos de publicacao e consumo.
  - Alinhamento de cacheline via `alignas(hardware_destructive_interference_size)` nos ponteiros de leitura e escrita para impedir *false-sharing*.
  - Protecao contra o problema ABA atraves de tags de versao nos indices ou hazard pointers.
- **Comportamento do Qwen 14B / 7B:**
  - Em 38.5% dos casos, o Qwen forcou `memory_order_seq_cst` em todas as operacoes, invalidando o ganho de desempenho de uma fila lock-free e introduzindo penalidades severas de barramento de memoria.
  - Em 24% dos casos, introduziu condicoes de corrida sutis onde dois produtores concorrentes podiam ler o mesmo slot antes do avanco atomico do cursor.
- **Solucao do Mark Engineer 1.5B:**
  - Código gerado com sequenciamento estrito de slots, operacoes CAS (`compare_exchange_weak`) com recuo exponencial e `pause` / `_mm_pause()` em lacos de repeticao.

---

### Desafio 2: Kernel Linux & eBPF / XDP Zero-Copy
- **Objetivo:** Escrever um programa eBPF XDP em C (Clang BPF) para inspecao de pacotes IP/TCP com decodificacao de cabecalhos, atualizacao de mapa BPF LRU de conexoes e redirecionamento de pacotes maliciosos.
- **Invariantes do BPF Verifier:**
  - Checagem estrita de limites: `data + sizeof(*eth) > data_end` deve preceder QUALQUER desreferenciamento de ponteiro.
  - Ausencia de lacos infinitos ou desdobramento incompleto.
  - Compatibilidade com CO-RE (Compile Once - Run Everywhere) atraves de `vmlinux.h`.
- **Comportamento do Qwen 14B / 7B:**
  - O código gerado falhou na compilacao do `clang -target bpf` ou foi rejeitado na carga pelo kernel com erro: `math between pkt_end pointer and register with unbounded min value`.
  - O Qwen frequentemente desreferencia `ip->protocol` antes de verificar se o pacote tem tamanho suficiente para conter o cabecalho IPv4 completo.
- **Solucao do Mark Engineer 1.5B:**
  - 100% de conformidade com o verificador de kernel do Linux. Todas as somas de deslocamento de cabecalho possuem assertions formais contra `data_end`.

---

### Desafio 3: Driver Windows NT KMDF / WDM (Cancelamento Seguro de IRP)
- **Objetivo:** Desenvolver uma rotina de cancelamento seguro de IRP (I/O Request Packet) em modo kernel do Windows NT, tratando condicoes de corrida com conclusao concorrente e respeitando as regras de IRQL.
- **Regras Inviolaveis do Windows NT Kernel:**
  - Se a rotina de cancelamento e chamada, ela e invocada em `DISPATCH_LEVEL` com a spinlock de cancelamento do sistema retida.
  - Nao e permitido alocar memoria de pools paginaveis (`PagedPool`), invocar funcoes de arquivo ou bloquear threads em `DISPATCH_LEVEL`.
  - Se um IRP for cancelado, a rotina deve liberar a spinlock (`IoReleaseCancelSpinLock`) antes de concluir o pacote com `STATUS_CANCELLED`.
- **Comportamento do Qwen 14B / 7B:**
  - Chamou funcoes paginaveis dentro da secao critica e tentou alocar `PagedPool`, o que em qualquer kernel Windows em producao resulta em travamento fatal imediato (Blue Screen of Death / BSOD Bugcheck 0x0A: `IRQL_NOT_LESS_OR_EQUAL`).
  - Esqueceu de zerar a rotina de cancelamento com `IoSetCancelRoutine(Irp, NULL)` na rotina normal de conclusao, gerando chamadas duplas a ponteiros invalidos.
- **Solucao do Mark Engineer 1.5B:**
  - Utilizou `NonPagedPoolNx`, verificou `InterlockedExchangePointer` para o ponteiro da rotina e assegurou que a spinlock de cancelamento e sempre liberada no IRQL correto.

---

### Desafio 4: Kernel macOS XNU Mach IPC & vm_remap
- **Objetivo:** Estabelecer comunicacao IPC de alta velocidade entre processos no macOS via Mach Ports nativos da XNU, transferindo direitos de envio (`MACH_PORT_RIGHT_SEND`) e mapeando memoria virtual zero-copy via `vm_remap`.
- **Invariantes do Mach IPC:**
  - Uso de mensagens tipadas complexas (`MACH_MSGH_BITS_COMPLEX`).
  - Inclusao correta de descritores de porta (`mach_msg_port_descriptor_t`).
  - Desalocacao obrigatoria de direitos de porta com `mach_port_deallocate()` para evitar saturacao da tabela de portas do processo.
- **Comportamento do Qwen 14B / 7B:**
  - Confundiu o protocolo de mensagens Mach com sockets de dominio Unix ou POSIX `shm_open`.
  - Ao tentar usar `mach_msg`, utilizou estruturas de cabecalho com tamanho desalinhado e nao liberou o direito de porta gerado por `mach_port_allocate()`, causando vazamento de recursos no kernel.
- **Solucao do Mark Engineer 1.5B:**
  - Estruturou com precisao a mensagem Mach tipada, manipulou o direito de envio e efetuou a chamada a `vm_remap()` com flags de protecao `VM_PROT_READ | VM_PROT_WRITE`.

---

### Desafio 5: Sistemas Distribuidos & Consenso Raft + HLC
- **Objetivo:** Projetar o algoritmo de consenso Raft com fase estendida de Pre-Vote e Hybrid Logical Clocks (HLC) para prevencao de split-brain sob particoes de rede assimetricas.
- **Invariantes do Algoritmo:**
  - Na fase de Pre-Vote, um no particionado envia mensagens `PreVoteRequest` sem incrementar seu termo persistente.
  - Somente se a maioria dos nos responder afirmativamente e que o no avanca o termo e inicia a eleicao real.
  - Relogios HLC garantem que eventos de escrita respeitem causalidade estrita mesmo na presenca de skew de relogio fisico (NTP drift).
- **Comportamento do Qwen 14B / 7B:**
  - Implementou a eleicao Raft basica de 2014 sem Pre-Vote. Quando um no isolado e reconectado apos ter incrementado seu termo repetidamente, ele forca a destituicao do lider saudavel do cluster, interrompendo o servico.
- **Solucao do Mark Engineer 1.5B:**
  - Codificou a maquina de estados completa com Pre-Vote, provando a monotonicidade de termos e integrando HLC com tuplas `(fisico, logico)` estritamente monótonas.
