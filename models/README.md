# Mark AI - Models Hub (LM Studio, Ollama & llama.cpp Plug-and-Play)

Este diretorio centraliza as definicoes arquiteturais, arquivos de configuracao e templates para execucao direta dos modelos Mark no **LM Studio**, **Ollama**, **llama.cpp** e **vLLM**.

---

## Modelos Disponiveis

| Modelo | Parametros | Formato | VRAM Local | Status | Pasta |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Mark Engineer 300M (302M)** | **302M** | **GGUF / IQ4_NL** | **~350 MB** | **Disponivel (Producao)** | [`mark-engineer-302m/`](./mark-engineer-302m/) |
| **Mark Engineer 1.5B** | **1.50B** | **GGUF / IQ4_NL** | **~1.25 GB** | **Em Treinamento A100** | [`mark-engineer-1.5b/`](./mark-engineer-1.5b/) |
| **Mark General 1.5B** | **1.50B** | **GGUF / IQ4_NL** | **~1.25 GB** | **Em Treinamento A100** | [`mark-general-1.5b/`](./mark-general-1.5b/) |

---

## 1. Como Usar no LM Studio (Plug-and-Play)

### Metodo 1: Download Direto do Arquivo GGUF
Baixe o modelo 302M diretamente no terminal para a pasta de modelos do seu LM Studio:

```bash
# Cria a pasta de destino do LM Studio
mkdir -p ~/.lmstudio/models/Mark-AI/mark-engineer-302m

# Baixa o arquivo GGUF compilado direto da Release oficial
curl -L -o ~/.lmstudio/models/Mark-AI/mark-engineer-302m/mark_engineer_302m.gguf \
  https://github.com/thu12dta3/Mark-AI/releases/download/v1.0.0-302m/mark_engineer_302m_iq4_nl.gguf
```

### Metodo 2: Carregamento
1. Abra o **LM Studio**.
2. Na barra de busca ou no menu de modelos locais, selecione **Mark-AI / mark-engineer-302m**.
3. O modelo e carregado instantaneamente com consumo de apenas **350 MB de VRAM**.

---

## 2. Como Usar no Ollama

Navegue ate a pasta do modelo desejado e execute:

```bash
cd models/mark-engineer-302m
ollama create mark-engineer -f Modelfile
ollama run mark-engineer "Otimize este algoritmo em Rust."
```

---

## 3. Como Usar no llama.cpp

```bash
./llama-cli \
  -m ~/.lmstudio/models/Mark-AI/mark-engineer-302m/mark_engineer_302m.gguf \
  -p "User: Analise este codigo em C++.\nAssistant: <think>" \
  -n 256
```
