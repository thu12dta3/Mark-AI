# Mark-SLM-V2-302M

Mark-SLM-V2-302M e um modelo compacto de linguagem para sistemas e raciocinio analitico baseado na arquitetura nativa DeepSeek2 MoE com Multi-Head Latent Attention (MLA).

- **Licenca:** AGPL-3.0
- **Parametros Fisicos:** 302M
- **Camadas:** 16 camadas (2 camadas densas iniciais + 14 camadas MoE com 16 especialistas roteados e 1 compartilhado)
- **Atencao:** MLA (q_lora_rank=384, kv_lora_rank=192)
- **Status de Treinamento:** 10.000 passos convergidos (Cross-Entropy Loss: 0.2639)
- **Formato:** GGUF nativo deepseek2 (tensores 3D de especialistas, roteador e bias)
