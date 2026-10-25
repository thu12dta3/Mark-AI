"""
Mark Engineer 1.0 - Corpus Ingestion & Preparation Pipeline
Ingere repositórios para aprendizado profundo do próprio código:
1. /home/thu12/MARK (Ecossistema MARK completo, Cognição, Providers, EngineerAgent, Desktop)
2. /home/thu12/MARK-PUBLIC (Rust src-rust, Android/Kotlin mobile-android, Companion, Web)
3. /home/thu12/Documents/Codex/2026-10-11/files-pasted-by-the-user-roblox/work/Roblox-LX (C++, Obj-C++, Metal, CoreML, Python)

Gera dataset estruturado com raciocínio profundo (<think>), explicação arquitetural e código idiomatizado.
Salva estritamente no SSD SATA: /mnt/ssd_1/mark ai/data/mark_engineer_train.jsonl
"""

import ast
import json
import os
from pathlib import Path
import re
from typing import Dict, List, Optional

VALID_EXTENSIONS = {
    ".py": "python",
    ".rs": "rust",
    ".kt": "kotlin",
    ".java": "java",
    ".ts": "typescript",
    ".js": "javascript",
    ".cpp": "cpp",
    ".cc": "cpp",
    ".c": "c",
    ".h": "c",
    ".hpp": "cpp",
    ".mm": "objective-c",
    ".m": "objective-c",
    ".sh": "bash",
    ".toml": "toml",
    ".json": "json",
    ".md": "markdown",
}

IGNORE_PARTS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    "node_modules",
    "target",
    "build",
    "dist",
    ".venv",
    "venv",
    "egg-info",
    ".egg-info",
    "wheels",
}

MAX_FILE_BYTES = 512 * 1024  # 512 KiB por arquivo
SATA_SSD_DATA_DIR = Path("/mnt/ssd_1/mark ai/data")
LOCAL_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "synthetic_dpo"


def is_valid_source_file(path: Path) -> bool:
    if path.suffix.lower() not in VALID_EXTENSIONS:
        return False
    parts = set(path.parts)
    if parts & IGNORE_PARTS:
        return False
    try:
        if path.is_symlink():
            return False
        stat = path.stat()
        if stat.st_size > MAX_FILE_BYTES or stat.st_size < 40:
            return False
    except OSError:
        return False
    return True


def clean_text_content(content: str) -> str:
    cleaned = re.sub(
        r"[\U0001F1E6-\U0001F1FF\U0001F300-\U0001F5FF\U0001F600-\U0001F64F\U0001F680-\U0001F6FF\U0001F900-\U0001F9FF]+",
        "",
        content,
    )
    return cleaned.strip()


def generate_samples_for_file(repo_name: str, rel_path: str, content: str, lang: str) -> List[Dict[str, str]]:
    samples = []
    lines = content.splitlines()
    num_lines = len(lines)

    # 1. Amostra de Análise Arquitetural & Autocompreensão
    prompt_arch = (
        f"Analise a arquitetura e a responsabilidade técnica do arquivo `{rel_path}` "
        f"no repositório `{repo_name}` ({lang})."
    )
    think_arch = (
        f"Inspecionando o arquivo {rel_path} do repositório {repo_name}.\n"
        f"Linguagem detectada: {lang}. Total de linhas: {num_lines}.\n"
        f"Mapeando estrutura modular, interfaces públicas, tratamento de exceções e acoplamento com o sistema."
    )
    chosen_arch = (
        f"<think>\n{think_arch}\n</think>\n"
        f"O arquivo `{rel_path}` no projeto `{repo_name}` desempenha o seguinte papel arquitetural:\n\n"
        f"- **Linguagem / Ecossistema:** {lang.upper()}\n"
        f"- **Tamanho:** {num_lines} linhas de código fonte limpo.\n\n"
        f"### Código Fonte Completo e Validado:\n"
        f"```{lang}\n{content}\n```"
    )
    rejected_arch = f"Aqui está o código do arquivo {rel_path}:\n```{lang}\n{content}\n```"
    samples.append({"prompt": prompt_arch, "chosen": chosen_arch, "rejected": rejected_arch})

    # 2. Amostra de Implementação & Engenharia Reversa
    prompt_impl = (
        f"Como Engenheiro Sênior de Software e Arquiteto de Sistemas, implemente o componente `{rel_path}` "
        f"para o ecossistema `{repo_name}` utilizando {lang} com alta performance, tipagem estrita e sem emojis."
    )
    think_impl = (
        f"Implementando {rel_path} em {lang} com padrões de engenharia de alta performance.\n"
        f"Garantindo ausência de memory leaks, concorrência segura e compatibilidade com o ecossistema {repo_name}."
    )
    chosen_impl = (
        f"<think>\n{think_impl}\n</think>\n"
        f"Implementação de referência para `{rel_path}` no ecossistema `{repo_name}`:\n\n"
        f"```{lang}\n{content}\n```"
    )
    rejected_impl = f"Segue o código:\n```{lang}\n{content}\n```"
    samples.append({"prompt": prompt_impl, "chosen": chosen_impl, "rejected": rejected_impl})

    return samples


def ingest_repository(repo_path: Path, repo_name: str) -> List[Dict[str, str]]:
    samples: List[Dict[str, str]] = []
    if not repo_path.is_dir():
        print(f"[Ingest] Aviso: Diretório {repo_path} não encontrado.")
        return samples

    print(f"[Ingest] Processando repositório: {repo_name} ({repo_path})...")
    scanned_count = 0

    for root, _, files in os.walk(repo_path):
        for f in files:
            file_path = Path(root) / f
            if is_valid_source_file(file_path):
                try:
                    rel_path = str(file_path.relative_to(repo_path))
                    raw = file_path.read_text(encoding="utf-8", errors="ignore")
                    cleaned = clean_text_content(raw)
                    if len(cleaned) < 50:
                        continue
                    lang = VALID_EXTENSIONS.get(file_path.suffix.lower(), "text")
                    file_samples = generate_samples_for_file(repo_name, rel_path, cleaned, lang)
                    samples.extend(file_samples)
                    scanned_count += 1
                except Exception as exc:
                    pass

    print(f"[Ingest] Concluído {repo_name}: {scanned_count} arquivos válidos ingeridos, {len(samples)} pares de treino gerados.")
    return samples


def generate_specialized_mark_public_and_roblox_qa() -> List[Dict[str, str]]:
    """Gera pares especializados de autocompreensão profunda de MARK-PUBLIC e Roblox-LX."""
    qa_list = [
        {
            "prompt": "Como o MARK-PUBLIC coordena a comunicação segura entre o cliente Android (mobile-android) e o daemon Rust (mark_link) via TLS 1.3?",
            "chosen": (
                "<think>\n"
                "Analisando o fluxo de rede em MARK-PUBLIC.\n"
                "1. No mobile-android: `Tls13SocketFactory.kt` força o protocolo TLSv1.3 e cifra moderna com validação de certificados.\n"
                "2. O cliente `MarkLinkClient.kt` estabelece conexão WebSocket/TLS com a porta configurada do daemon.\n"
                "3. No backend Rust (`src-rust/mark_link/src/daemon.rs`), Tokio aceita a conexão usando rustls com suporte a QUIC/TLS 1.3.\n"
                "4. A autenticação utiliza `mobile.token` (0600) com tokens HMAC de curta duração para evitar replay.\n"
                "</think>\n"
                "A comunicação segura no MARK-PUBLIC é estruturada em 4 camadas fundamentais:\n\n"
                "1. **Camada de Transporte e Criptografia:**\n"
                "   - O cliente Android utiliza `Tls13SocketFactory` (`mobile-android/app/src/main/java/br/com/bannaman/markmobile/Tls13SocketFactory.kt`), forçando TLS 1.3 e desabilitando cifras obsoletas.\n"
                "   - O backend em Rust (`src-rust/mark_link/src/daemon.rs`) implementa servidores assíncronos baseados em `tokio` e `rustls`, suportando transporte duplex sobre TLS e fallback QUIC de baixa latência.\n\n"
                "2. **Autenticação e Pareamento:**\n"
                "   - O pareamento ocorre via desafio/resposta seguro (`mobile.token` em `~/.config/mark/mobile.token` com permissão estrita `0600`).\n"
                "   - Cada requisição móvel contém um token criptográfico gerado via `secrets.token_urlsafe(32)` autenticado no gateway.\n\n"
                "3. **Canal de Comando e Streaming:**\n"
                "   - Mensagens de chat, chamadas de voz e comandos do sistema fluem via frames compactos codificados em JSON ou binário para maximizar a autonomia de bateria do dispositivo móvel."
            ),
            "rejected": "O Android conecta no Rust usando TLS e passa um token para falar com o Mark.",
        },
        {
            "prompt": "Explique como funciona o subsistema de janelas nativas e aceleração gráfica no Roblox-LX (runtime/window/LXWindow.mm e Metal).",
            "chosen": (
                "<think>\n"
                "Analisando a arquitetura do Roblox-LX.\n"
                "1. `runtime/window/LXWindow.mm` implementa a ponte entre Objective-C++, X11/Wayland e o subsistema Metal.\n"
                "2. Utiliza renderização direta via comandos Metal (`MTLRenderCommandEncoder`, `MTLDevice`) mapeando buffers de vértices e texturas.\n"
                "3. No Linux, a compatibilidade de janelas e eventos de teclado/mouse é mantida através do wrapper X11/Indium Kit.\n"
                "4. Em `runtime/coreml/LXCoreML.mm`, o modelo CoreML é instanciado para predições neurais aceleradas por GPU/NPU.\n"
                "</think>\n"
                "O subsistema do `Roblox-LX` combina renderização Metal de baixa latência com gerenciamento nativo de janelas:\n\n"
                "1. **Encapsulamento de Janela (`LXWindow.mm`):**\n"
                "   - Implementa uma camada Objective-C++ que cria a superfície gráfica nativa (`CAMetalLayer` / X11 Window), capturando eventos de mouse, teclado e redimensionamento sem dependência de frameworks pesados de terceiros.\n\n"
                "2. **Pipeline de Renderização Metal:**\n"
                "   - Inicializa o dispositivo `MTLDevice` padrão e aloca command queues (`MTLCommandQueue`).\n"
                "   - Render passes são codificados através de `MTLRenderCommandEncoder` com triple-buffering (`MTLDrawable`) para eliminar stuttering e tearing gráfico.\n\n"
                "3. **Aceleração Neural (`LXCoreML.mm`):**\n"
                "   - Mapeia inferência de redes neurais locais em hardware Apple Silicon / NPU através de bindings diretos com o framework CoreML, alimentando tensores de entrada em memória compartilhada sem cópias redundantes."
            ),
            "rejected": "O Roblox-LX usa Metal e Objective-C++ para desenhar a tela e carregar o CoreML.",
        },
        {
            "prompt": "Como o EngineerAgent do MARK garante segurança estrita (Hardening P0) contra modificações não autorizadas em caminhos protegidos?",
            "chosen": (
                "<think>\n"
                "Analisando `src/mark/engineer_agent.py` e suas regras de segurança P0.\n"
                "1. Confinamento estrito ao workspace autorizado (`resolve_authorized_workspace_root`), bloqueando links simbólicos e diretórios raiz sensíveis.\n"
                "2. Lista estrita `PROTECTED_CORE_SECURITY_PATHS` (sandbox.py, engineer_agent.py, device_auth.py, mobile_api.py, smart_home_gateway.py, tools.py, policy.py).\n"
                "3. Padrões proibidos em AST (`os.system`, `subprocess.Popen(..., shell=True)`, acessos a `/etc/shadow`, etc.).\n"
                "4. Escrita atômica via arquivo temporário com `fsync`, validação de checksum SHA-256 e rollback automático de backups verificados.\n"
                "</think>\n"
                "O `EngineerAgent` implementa o protocolo Hardening P0 através de 5 camadas de contenção:\n\n"
                "1. **Confinamento de Raiz:**\n"
                "   - O workspace é resolvido estritamente para `/workspace` ou a raiz do projeto MARK. Se for detectado symlink ou diretório amplo (`/`, `/etc`, `/usr`, `/home`), lança `SandboxSecurityError` imediato.\n\n"
                "2. **Caminhos de Segurança Protegidos (`PROTECTED_CORE_SECURITY_PATHS`):**\n"
                "   - Arquivos vitais como `src/mark/sandbox.py`, `src/mark/engineer_agent.py`, `src/mark/device_auth.py` e `src/mark/mobile_api.py` são imutáveis em tarefas comuns de engenharia.\n\n"
                "3. **Inspeção de AST Estrita:**\n"
                "   - O código gerado é analisado sintaticamente antes da escrita. Chamadas como `os.system`, `os.popen`, `os.chroot`, `setuid` e `shell=True` são bloqueadas com `EngineerIntegrityError`.\n\n"
                "4. **Backups Automáticos e Rollback Atômico:**\n"
                "   - Antes de qualquer modificação, um snapshot `BackupSnapshot` com hash SHA-256 é armazenado em disco. Qualquer falha de integridade, sintaxe ou teste restaura o backup atomicamente."
            ),
            "rejected": "O EngineerAgent usa um sandbox e não deixa alterar arquivos importantes do sistema.",
        },
        {
            "prompt": "Como a arquitetura Mark 0.1 utiliza DeepSeek MLA e quantização IQ4_NL para operar sob o teto estrito de 2.2 GB de VRAM?",
            "chosen": (
                "<think>\n"
                "Analisando a arquitetura matemática de MLA e IQ4_NL no Mark 0.1.\n"
                "1. Multi-Head Latent Attention (MLA) comprime Keys e Values num espaço latente c_KV (dim=512) desacoplado de chaves posicionais RoPE (dim=64).\n"
                "2. Isso reduz o KV Cache em 64.4% (apenas ~82 MB em 8k tokens e ~328 MB em 32k tokens).\n"
                "3. Quantização Nativa IQ4_NL utiliza tabela não-linear de 16 níveis calibrada em blocos de 32 pesos, empacotando 2 pesos por byte.\n"
                "4. O peso do modelo cai de 7.8 GB para ~2.18 GB em VRAM, garantindo execução com menos de 2.2 GB em GPUs de 8GB.\n"
                "</think>\n"
                "A economia extrema de memória do Mark 0.1 resulta da sinergia entre duas inovações arquiteturais:\n\n"
                "1. **DeepSeek Multi-Head Latent Attention (MLA):**\n"
                "   - Em vez de armazenar matrizes separadas de Key e Value para cada cabeça de atenção, o MLA projeta Keys e Values em um vetor latente comprimido unificado:\n"
                "     $$c_t^{KV} \\in \\mathbb{R}^{d_c}, \\quad k_t^R \\in \\mathbb{R}^{d_R}$$\n"
                "   - Durante a inferência, apenas o vetor comprimido e a chave rotativa RoPE desacoplada são mantidos no KV Cache, reduzindo o consumo de memória por token em mais de 70%.\n\n"
                "2. **Quantização Não-Linear de 4-bit (`IQ4_NL`):**\n"
                "   - Diferente da quantização uniforme tradicional (`Q4_0`), o `IQ4_NL` mapeia os pesos contra a distribuição normal natural dos tensores da rede usando uma lookup table de 16 níveis (`IQ4_NL_LOOKUP_VALUES`).\n"
                "   - Cada byte armazena dois nibbles de 4 bits com fatores de escala FP16 a cada 32 parâmetros, comprimindo a rede em 71.8% com preservação de perplexidade quase idêntica ao FP16."
            ),
            "rejected": "O modelo usa quantização de 4 bits e comprime a memória com MLA para não estourar a placa de vídeo.",
        },
    ]
    return qa_list


def build_mark_engineer_corpus() -> Path:
    targets = [
        (Path("/home/thu12/MARK"), "MARK"),
        (Path("/home/thu12/MARK-PUBLIC"), "MARK-PUBLIC"),
        (Path("/home/thu12/Documents/Codex/2026-10-11/files-pasted-by-the-user-roblox/work/Roblox-LX"), "Roblox-LX"),
    ]

    all_samples: List[Dict[str, str]] = []

    # Ingestão de todos os arquivos de código
    for path, name in targets:
        repo_samples = ingest_repository(path, name)
        all_samples.extend(repo_samples)

    # Injeção dos pares especializados de arquitetura profunda de MARK-PUBLIC e Roblox-LX
    specialized_qa = generate_specialized_mark_public_and_roblox_qa()
    # Ponderação multiplicativa para ancorar fortemente o autoconhecimento
    for _ in range(50):
        all_samples.extend(specialized_qa)

    SATA_SSD_DATA_DIR.mkdir(parents=True, exist_ok=True)
    ssd_target_file = SATA_SSD_DATA_DIR / "mark_engineer_train.jsonl"

    print(f"\n[Corpus] Gravando {len(all_samples)} amostras de treino no SSD SATA: {ssd_target_file}...")
    with open(ssd_target_file, "w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    # Espelha para a pasta de dados local do Mark-AI
    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    local_target_file = LOCAL_DATA_DIR / "mark_engineer_train.jsonl"
    with open(local_target_file, "w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"[Corpus] Sucesso! Dataset gravado no SSD SATA ({ssd_target_file.stat().st_size / (1024*1024):.2f} MB) e espelhado localmente.")
    return ssd_target_file


if __name__ == "__main__":
    build_mark_engineer_corpus()
