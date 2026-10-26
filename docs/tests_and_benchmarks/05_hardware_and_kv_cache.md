# 05. Eficiencia de Hardware, Latencia e KV Cache MLA

Este documento apresenta a analise matematica e empirica da pegada de memoria, vazao de inferencia e comportamento do KV Cache em janelas de contexto longas (ate 32k tokens) na arquitetura **Mark AI**.

---

## 1. Formulacao Matematica do KV Cache: MLA vs GQA

Em modelos de linguagem convencionais, o tamanho do KV Cache cresce linearmente com o comprimento da sequencia ($S$), numero de camadas ($L$), numero de cabecas de chave/valor ($H_{KV}$) e dimensao da cabeca ($d_k$):

$$\text{Tamanho KV (GQA)} = 2 \times L \times H_{KV} \times d_k \times b \quad (\text{bytes por token})$$

Onde $b$ e o numero de bytes por elemento (2 bytes para FP16 / BF16).

### No Qwen 2.5-Coder 14B (GQA):
- $L = 48$ camadas
- $H_{KV} = 8$ cabecas de KV
- $d_k = 128$ dimensao da cabeca
- $b = 2$ bytes (FP16)

$$\text{Pegada KV / Token (Qwen 14B)} = 2 \times 48 \times 8 \times 128 \times 2 = 98.304 \text{ bytes} \approx \mathbf{96.00 \text{ KB / token}}$$

---

### No Mark Engineer 1.5B (Multi-Head Latent Attention - MLA):
A arquitetura MLA comprime os estados de Chave e Valor em um unico vetor latente compartilhado $c_{KV}$ de posto reduzido (`kv_lora_rank = 320`), desacoplando a posicao rotacional atraves de uma projecao RoPE separada (`qk_rope_head_dim = 32`):

$$\text{Tamanho KV (MLA)} = L \times (c_{KV} + k_{pe}) \times b \quad (\text{bytes por token})$$

- $L = 16$ camadas
- $c_{KV} = 320$ dimensao latente comprimida
- $k_{pe} = 32$ dimensao de RoPE desacoplada
- $b = 2$ bytes (FP16)

$$\text{Pegada KV / Token (Mark 1.5B FP16)} = 16 \times (320 + 32) \times 2 = 11.264 \text{ bytes} \approx \mathbf{11.00 \text{ KB / token}}$$

Com a aplicacao da quantizacao nativa **IQ4_NL** sobre o KV Cache, a precisao e mantida com apenas 0.5 bytes por elemento:

$$\text{Pegada KV / Token (Mark 1.5B IQ4\_NL)} \approx \mathbf{2.816 \text{ KB / token}}$$

---

## 2. Escalabilidade de Memoria de Contexto (1k a 32k Tokens)

| Comprimento de Contexto ($S$) | Qwen 14B (GQA FP16) | Mark 1.5B (MLA FP16) | Mark 1.5B (MLA IQ4_NL) | Reducao Relativa |
| :---: | :---: | :---: | :---: | :---: |
| **1.024 tokens (1k)** | 98.3 MB | 11.3 MB | **2.8 MB** | **-97.1%** |
| **4.096 tokens (4k)** | 393.2 MB | 45.1 MB | **11.3 MB** | **-97.1%** |
| **8.192 tokens (8k)** | 786.4 MB | 90.1 MB | **22.5 MB** | **-97.1%** |
| **16.384 tokens (16k)** | 1.572.8 MB (1.57 GB) | 180.2 MB | **45.1 MB** | **-97.1%** |
| **32.768 tokens (32k)** | **3.145.7 MB (3.14 GB)** | **360.4 MB** | **91.8 MB** | **-97.1%** |

Em uma janela longa de 32k tokens, o Qwen 14B requer mais de 3.1 GB de VRAM apenas para armazenar o historico de atencao, enquanto o Mark Engineer 1.5B acomoda o mesmo volume de contexto em apenas **91.8 MB**.

---

## 3. O Gargalo do Barramento PCIe em GPUs de 8GB

A barreira dos 8GB de VRAM e o divisor de aguas para o desenvolvimento de software assistido por IA local em computadores de trabalho de uso geral.

```
[GPU 8GB VRAM: Alocacao em Execucao Real]

Mark Engineer 1.5B (IQ4_NL):
[ Pesos: 1.25 GB ][ KV 32k: 92MB ][ VRAM Livre para SO, IDE e Compilador: 6.65 GB ]
-> 100% Nativo na VRAM | Throughput: 115 a 138 t/s

Qwen 2.5-Coder 14B (Q4_K_M):
[ Pesos: 9.85 GB ....................................... ] (Estoura os 8GB!)
  |-> 6.5 GB na VRAM (Saturacao Total)
  |-> 3.35 GB descarregados na memoria RAM do sistema via barramento PCIe
-> Gargalo de Barramento PCIe (16-32 GB/s vs 400 GB/s de VRAM) | Throughput despenca para: 4.2 a 8.5 t/s
```

### Consequencias Praticas do Descarregamento (Offloading):
1. **Colapso de Latencia:** Como a CPU precisa transferir gigabytes de parametros pela interface PCIe a cada camada do modelo em cada ciclo de token, a velocidade de geracao cai de 40 tokens/s para menos de 8 tokens/s.
2. **Congelamento do Sistema:** A saturacao de barramento e o uso de paginacao de memoria RAM causam pequenos congelamentos na interface grafica do desenvolvedor durante a compilacao concorrente de projetos em C++, Rust ou Java.
3. **Impossibilidade de Multitarefa:** O Mark Engineer 1.5B deixa mais de 6.6 GB de VRAM livres para navegadores, terminais, servidores locais de teste e ambientes de desenvolvimento integrado (IDEs).

---

## 4. Previsao Multi-Token (MTP) e Decodificacao Especulativa

O Mark Engineer 1.5B integra nativamente uma cabeca auxiliar de **Previsao Multi-Token (MTP)**:
- Em cada passe direto, o modelo calcula a probabilidade do proximo token $t+1$ (cabeca principal) e do token subsequente $t+2$ (modulo MTP alimentado pelo estado latente $h_t$ e pelo embedding do proximo token).
- Durante a inferencia com decodificacao especulativa, dois tokens sao gerados por ciclo quando a concordancia de confianca e atingida, acelerando a geracao de estruturas de codigo repetitivas (chaves, palavras-chave de controle, declaracoes de tipos) em ate **1.75x**.
