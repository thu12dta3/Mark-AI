# Mark Engineer 300M (302M) - LM Studio & Ollama Plug-and-Play

Este modelo esta pronto para execucao direta em plataformas de inferencia local como **LM Studio**, **Ollama**, **llama.cpp** e **vLLM**.

---

## 1. Plug-and-Play no LM Studio

### Passo 1: Obter o Arquivo GGUF
Baixe o arquivo GGUF com um comando unico:
```bash
curl -L -O https://github.com/thu12dta3/Mark-AI/releases/download/v1.0.0-302m/mark_engineer_302m.gguf
```

### Passo 2: Copiar para o Diretorio do LM Studio
Copie ou crie um link simbolico do arquivo para a pasta de modelos do LM Studio:
```bash
mkdir -p ~/.lmstudio/models/Mark-AI/mark-engineer-302m
cp mark_engineer_302m.gguf ~/.lmstudio/models/Mark-AI/mark-engineer-302m/
```

### Passo 3: Abrir no LM Studio
1. Abra o **LM Studio**.
2. No menu superior de selecao de modelo, clique em **Mark-AI / mark-engineer-302m**.
3. O modelo sera carregado consumindo apenas **~350 MB de VRAM** com suporte a janela de contexto de 4.096 tokens.

---

## 2. Plug-and-Play no Ollama

No diretorio desta pasta, execute:
```bash
ollama create mark-engineer -f Modelfile
ollama run mark-engineer "Otimize este algoritmo em Rust com lock-free ringbuffer."
```

---

## 3. Especificacoes Tecnicas

- **Parametros Totais:** 302.019.848 (302M)
- **Parametros Ativos:** ~110M por token
- **Atencao:** Multi-Head Latent Attention (MLA)
- **MoE:** 16 Especialistas Roteados + 1 Especialista Compartilhado
- **Consumo de VRAM:** ~350 MB
- **Formato:** GGUF (float16 / IQ4_NL compatibilidade universal)
