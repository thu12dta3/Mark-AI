# AGENTS.md: Protocolo Operacional para Agentes Autonomos

Miceworks Systems Engineering // Repositorio Mark-AI

Este documento estabelece as diretrizes e regras obrigatorias para qualquer agente de inteligencia artificial operando diretamente sobre a base de codigo do repositorio Mark-AI (Google Jules, Antigravity, Cursor ou ferramentas correlatas).

---

## 1. Escopo e Diretrizes de Autonomia

1. Responsabilidade Integral: O agente atua como engenheiro de software e machine learning autonomo. O desenvolvedor nao realiza correcoes manuais nem debuga erros de sintaxe ou logica introduzidos pelo agente.
2. Validacao Pré-Entrega: O agente e estritamente responsavel por planejar, implementar, compilar e testar suas proprias alteracoes antes de considerar uma tarefa concluida.
3. Repositorio Publico: Todo codigo, docstring e documentacao devem manter rigor tecnico e padrao profissional de codigo aberto.

---

## 2. Invariantes de Codigo e Proibicoes

1. Integridade de Pesos e Tensores:
   - Proibido inicializar tensores aleatorios (torch.randn, torch.empty) fingindo representar checkpoints treinados. Se um arquivo .pt ou .safetensors estiver ausente, deve-se lancar FileNotFoundError explicito.

2. Preservacao de Especialistas MoE:
   - Os especialistas roteados das arquiteturas DeepSeekMoE e PB-MoE devem sempre ser preservados como tensores tridimensionais (blk.{i}.ffn_gate_exps.weight) no formato deepseek2.
   - Proibido achatar, mesclar ou tirar media aritmetica (.mean(dim=0)) de especialistas para forcar compatibilidade com containers densos.

3. Execucao Real de Testes e Benchmarks:
   - Proibido o uso de tensores estaticos artificiais (dummy_input fixo) em scripts de inferencia de producao.
   - Proibido forjar metricas em benchmarks copiando gabaritos (.copy()) ou gerando valores deterministicos ficticios com hash(). Avaliacoes devem executar inferencias neurais reais.

4. Quantizacao e Estabilidade Matematica:
   - O padrao de quantizacao adotado e nao-linear de 4 bits (IQ4_NL / Q4_K_M). Proibido criar dependencias de runtime para quantizacoes experimentais de 2 bits que quebrem os runtimes padrao (llama.cpp e Ollama).
   - No H-MLA, a reducao de curtose e obtida pela absorcao algebrica estatica da matriz ortogonal de Walsh-Hadamard (H_256) diretamente nos pesos lineares durante a exportacao GGUF, sem necessidade de kernels C++ customizados na inferencia.

5. Preservacao de Assercoes:
   - Proibido comentar, suprimir ou contornar assercoes de testes automatizados para forcar status de aprovacao.
   - Proibido usar sys.exit(0) forjado ou suprimir falhas criticas com blocos de excecao vazios.

6. Limpeza e Concisao de Codigo:
   - O codigo de producao nao deve conter comentarios informais, resumos conversacionais de IA ou instrucoes passo a passo redundantes.
   - Mantenha estritamente a linha shebang (#!/usr/bin/env python3), listas __all__, definicoes de tipo (typing), decoradores funcionais e docstrings estruturais de API publica.

7. Resolucao Dinamica de Caminhos:
   - Proibido chumbar caminhos absolutos de ambientes locais ou de nuvem.
   - Todos os caminhos de arquivos e modulos devem ser resolvidos dinamicamente de forma relativa via Path(__file__).resolve().parent.parent.

8. Bloqueio de Versionamento de Binarios:
   - Proibido rastrear ou submeter binarios pesados no Git (*.pt, *.gguf, *.bin, *.safetensors, *.zim, .env).

---

## 3. Fluxo de Execucao Obrigatorio

Ao executar qualquer tarefa, o agente deve seguir quatro etapas sequenciais:

Etapa 1 - Analise de Dependencias:
Inspecione os arquivos envolvidos e seus contratos de interface antes de aplicar alteracoes.

Etapa 2 - Implementacao:
Aplique as alteracoes necessarias mantendo aderencia estrita as assinaturas, contratos de tipo e nomes de tensores estabelecidos.

Etapa 3 - Compilacao e Testes no Terminal:
Antes de finalizar a tarefa, execute os seguintes comandos no terminal do ambiente:
1. Compilacao de sintaxe em toda a arvore:
   python3 -m compileall -q src/ export/ scripts/ models/ tests/
2. Execucao dos testes automatizados:
   pytest tests/ -v

Se houver falha, o agente deve analisar o traceback, corrigir a causa raiz e reexecutar a validacao ate obter aprovacao completa.

Etapa 4 - Auditoria de Diff:
Verifique o git diff e confirme que nenhuma das invariantes do projeto foi violada, nenhum comentario desnecessario foi inserido e nenhum arquivo binario foi colocado em stage.
