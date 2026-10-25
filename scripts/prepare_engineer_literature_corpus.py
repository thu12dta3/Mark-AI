"""
Mark Engineer - Literature-Grounded Vague Intent Disambiguation Pipeline
=======================================================================
Fundamenta a resolução de pedidos vagos nas principais obras da ciência da computação e engenharia de software:
1. ISO/IEC/IEEE 29148 (Requirements Engineering & Ambiguity Elimination)
2. Brendan Gregg ("Systems Performance: Enterprise and the Cloud")
3. Michael T. Nygard ("Release It!: Design and Deploy Production-Ready Software")
4. Martin Fowler ("Patterns of Enterprise Application Architecture")
5. Gregor Hohpe & Bobby Woolf ("Enterprise Integration Patterns")
6. Bertrand Meyer ("Design by Contract")
7. Jim Gray & Andreas Reuter ("Transaction Processing: Concepts and Techniques")
8. NIST SP 800-63B & OWASP ASVS (Application Security Standards)
"""

import json
from pathlib import Path
from typing import Dict, List

SATA_SSD_DATA_DIR = Path("/mnt/ssd_1/mark ai/data")
LOCAL_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "synthetic_dpo"


def generate_literature_grounded_vague_intent_pairs() -> List[Dict[str, str]]:
    """Gera pares de treino onde pedidos vagos são desmembrados segundo a literatura clássica."""
    return [
        # 1. Cache em Memória (Gregg / Nygard / Meyer)
        {
            "prompt": "preciso de um cache rapido na memoria pro meu backend em go",
            "chosen": (
                "<think>\n"
                "Desambiguação de pedido vago segundo a literatura de Engenharia de Software:\n"
                "1. ISO/IEC/IEEE 29148 (Engenharia de Requisitos): O pedido omite requisitos não-funcionais (NFRs) críticos como taxa de leitura vs escrita, latência p99, limite de memória e política de concorrência.\n"
                "2. Brendan Gregg ('Systems Performance', Cap. 3 - Caches): Caches com Mutex global sofrem de saturação por lock contention sob alto paralelismo. A solução canônica é o Sharding de chaves por hash FNV-1a para particionar os locks em N shards independentes.\n"
                "3. Michael T. Nygard ('Release It!', Padrões de Estabilidade): Prevenção de Out-of-Memory (OOM) através de Time-To-Live (TTL) ativo com rotina de limpeza concorrente (Janitor).\n"
                "4. Bertrand Meyer ('Design by Contract'): Invariante de que operações de leitura utilizam RWMutex compartilhado sem travar leitores paralelos.\n"
                "</think>\n"
                "Segundo as diretrizes de **Sistemas de Alta Performance (Gregg)** e **Padrões de Estabilidade (Nygard)**, a arquitetura ideal em Go para eliminar contenção de locks é um **Cache Shardado com Hashing FNV-1a e TTL Ativo**:\n\n"
                "```go\n"
                "package cache\n"
                "\n"
                "import (\n"
                "\t\"hash/fnv\"\n"
                "\t\"sync\"\n"
                "\t\"time\"\n"
                ")\n"
                "\n"
                "type item struct {\n"
                "\tvalue      any\n"
                "\texpiration int64\n"
                "}\n"
                "\n"
                "type shard struct {\n"
                "\tsync.RWMutex\n"
                "\tentries map[string]item\n"
                "}\n"
                "\n"
                "type ShardedCache struct {\n"
                "\tshardCount uint32\n"
                "\tshards     []*shard\n"
                "}\n"
                "\n"
                "func NewShardedCache(shardCount uint32, cleanupInterval time.Duration) *ShardedCache {\n"
                "\tc := &ShardedCache{\n"
                "\t\tshardCount: shardCount,\n"
                "\t\tshards:     make([]*shard, shardCount),\n"
                "\t}\n"
                "\tfor i := uint32(0); i < shardCount; i++ {\n"
                "\t\tc.shards[i] = &shard{entries: make(map[string]item)}\n"
                "\t}\n"
                "\tgo c.startJanitor(cleanupInterval)\n"
                "\treturn c\n"
                "}\n"
                "\n"
                "func (c *ShardedCache) getShard(key string) *shard {\n"
                "\th := fnv.New32a()\n"
                "\th.Write([]byte(key))\n"
                "\treturn c.shards[h.Sum32()%c.shardCount]\n"
                "}\n"
                "\n"
                "func (c *ShardedCache) Set(key string, val any, ttl time.Duration) {\n"
                "\ts := c.getShard(key)\n"
                "\ts.Lock()\n"
                "\tdefer s.Unlock()\n"
                "\texp := time.Now().Add(ttl).UnixNano()\n"
                "\ts.entries[key] = item{value: val, expiration: exp}\n"
                "}\n"
                "\n"
                "func (c *ShardedCache) Get(key string) (any, bool) {\n"
                "\ts := c.getShard(key)\n"
                "\ts.RLock()\n"
                "\tdefer s.RUnlock()\n"
                "\ti, found := s.entries[key]\n"
                "\tif !found || (i.expiration > 0 && time.Now().UnixNano() > i.expiration) {\n"
                "\t\treturn nil, false\n"
                "\t}\n"
                "\treturn i.value, true\n"
                "}\n"
                "\n"
                "func (c *ShardedCache) startJanitor(interval time.Duration) {\n"
                "\tticker := time.NewTicker(interval)\n"
                "\tfor range ticker.C {\n"
                "\t\tnow := time.Now().UnixNano()\n"
                "\t\tfor _, s := range c.shards {\n"
                "\t\t\ts.Lock()\n"
                "\t\t\tfor k, v := range s.entries {\n"
                "\t\t\t\tif v.expiration > 0 && now > v.expiration {\n"
                "\t\t\t\t\tdelete(s.entries, k)\n"
                "\t\t\t\t}\n"
                "\t\t\t}\n"
                "\t\t\ts.Unlock()\n"
                "\t\t}\n"
                "\t}\n"
                "}\n"
                "```"
            ),
            "rejected": "Você precisa especificar qual tipo de cache quer. Use um map global com lock.",
        },

        # 2. Resiliência e Falhas em Cascata (Nygard / Fowler)
        {
            "prompt": "minha api ta caindo quando o servico de fora engasga",
            "chosen": (
                "<think>\n"
                "Desambiguação segundo Michael T. Nygard ('Release It! - Stability Patterns') e Martin Fowler ('Circuit Breaker'):\n"
                "1. Diagnóstico do problema vago: O sintoma indica 'Cascading Failure' (falha em cascata) por exaustão de thread pool e socket pool quando dependências externas aumentam a latência.\n"
                "2. Padrões Formais da Literatura:\n"
                "   - Circuit Breaker: Estados Closed (normal), Open (falha rápida / fail-fast sem onerar a rede) e Half-Open (sonda de recuperação gradual).\n"
                "   - Bulkhead Isolation: Confinamento de recursos para que a falha da dependência externa não esgote as threads do serviço principal.\n"
                "   - Timeouts determinísticos com Exponential Backoff e Jitter estocástico.\n"
                "</think>\n"
                "Conforme documentado em **Release It! (Nygard)**, a proteção contra dependências lentas exige um **Circuit Breaker com Máquina de Estados Finita e Janela Deslizante**:\n\n"
                "```python\n"
                "import time\n"
                "import threading\n"
                "from enum import Enum\n"
                "from typing import Callable, Any, Optional\n"
                "\n"
                "class State(Enum):\n"
                "    CLOSED = 1\n"
                "    OPEN = 2\n"
                "    HALF_OPEN = 3\n"
                "\n"
                "class CircuitBreaker:\n"
                "    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 30.0):\n"
                "        self.failure_threshold = failure_threshold\n"
                "        self.recovery_timeout = recovery_timeout\n"
                "        self.state = State.CLOSED\n"
                "        self.failure_count = 0\n"
                "        self.last_state_change = time.monotonic()\n"
                "        self.lock = threading.Lock()\n"
                "\n"
                "    def call(self, func: Callable[..., Any], *args, **kwargs) -> Any:\n"
                "        with self.lock:\n"
                "            now = time.monotonic()\n"
                "            if self.state == State.OPEN:\n"
                "                if now - self.last_state_change > self.recovery_timeout:\n"
                "                    self.state = State.HALF_OPEN\n"
                "                    self.last_state_change = now\n"
                "                else:\n"
                "                    raise RuntimeError(\"CircuitBreaker aberto: chamada rejeitada para evitar falha em cascata\")\n"
                "\n"
                "        try:\n"
                "            res = func(*args, **kwargs)\n"
                "            with self.lock:\n"
                "                if self.state == State.HALF_OPEN:\n"
                "                    self.state = State.CLOSED\n"
                "                    self.failure_count = 0\n"
                "                    self.last_state_change = time.monotonic()\n"
                "            return res\n"
                "        except Exception as exc:\n"
                "            with self.lock:\n"
                "                self.failure_count += 1\n"
                "                if self.failure_count >= self.failure_threshold:\n"
                "                    self.state = State.OPEN\n"
                "                    self.last_state_change = time.monotonic()\n"
                "            raise exc\n"
                "```"
            ),
            "rejected": "Aumenta o timeout da requisição ou reinicia o servidor quando travar.",
        },

        # 3. Integração de Mensageria & Backpressure (Hohpe / Lamport)
        {
            "prompt": "faz uma fila pra nao perder mensagem",
            "chosen": (
                "<think>\n"
                "Desambiguação segundo Gregor Hohpe & Bobby Woolf ('Enterprise Integration Patterns') e Leslie Lamport:\n"
                "1. Diagnóstico do pedido vago: O usuário necessita de um canal de mensagens resiliente. Filas sem limite em memória (*unbounded*) quebram o sistema sob pico de tráfego por OOM.\n"
                "2. Padrões Formais:\n"
                "   - Bounded Channel com Backpressure: O produtor é forçado a esperar quando a capacidade máxima da fila é atingida (Nygard 'Shed Load').\n"
                "   - Dead Letter Channel (DLC): Mensagens com falha de processamento após N tentativas são preservadas em quarentena sem bloquear a fila principal.\n"
                "   - Graceful Drain: Encerramento ordenado onde todos os workers processam as mensagens já enfileiradas antes da finalização do processo.\n"
                "</think>\n"
                "Segundo os **Enterprise Integration Patterns (Hohpe & Woolf)**, a implementação canônica para entrega confiável sem esgotamento de memória em Rust utiliza **Bounded MPSC com Dead-Letter Queue e CancellationToken**:\n\n"
                "```rust\n"
                "use std::sync::Arc;\n"
                "use thiserror::Error;\n"
                "use tokio::sync::{mpsc, Mutex};\n"
                "use tokio_util::sync::CancellationToken;\n"
                "\n"
                "#[derive(Debug, Error)]\n"
                "pub enum MessageQueueError {\n"
                "    #[error(\"Fila encerrada\")]\n"
                "    Closed,\n"
                "    #[error(\"Dead Letter Queue saturada\")]\n"
                "    DeadLetterOverflow,\n"
                "}\n"
                "\n"
                "pub struct ResilientChannel<T> {\n"
                "    sender: mpsc::Sender<T>,\n"
                "    receiver: Arc<Mutex<mpsc::Receiver<T>>>,\n"
                "    dlq_sender: mpsc::Sender<T>,\n"
                "    token: CancellationToken,\n"
                "}\n"
                "\n"
                "impl<T: Send + 'static> ResilientChannel<T> {\n"
                "    pub fn new(capacity: usize, dlq_capacity: usize) -> (Self, mpsc::Receiver<T>) {\n"
                "        let (tx, rx) = mpsc::channel(capacity);\n"
                "        let (dlq_tx, dlq_rx) = mpsc::channel(dlq_capacity);\n"
                "        (\n"
                "            Self {\n"
                "                sender: tx,\n"
                "                receiver: Arc::new(Mutex::new(rx)),\n"
                "                dlq_sender: dlq_tx,\n"
                "                token: CancellationToken::new(),\n"
                "            },\n"
                "            dlq_rx,\n"
                "        )\n"
                "    }\n"
                "\n"
                "    pub async fn publish(&self, item: T) -> Result<(), MessageQueueError> {\n"
                "        if self.token.is_cancelled() {\n"
                "            return Err(MessageQueueError::Closed);\n"
                "        }\n"
                "        self.sender.send(item).await.map_err(|_| MessageQueueError::Closed)\n"
                "    }\n"
                "\n"
                "    pub async fn send_to_dlq(&self, poison_item: T) -> Result<(), MessageQueueError> {\n"
                "        self.dlq_sender.send(poison_item).await.map_err(|_| MessageQueueError::DeadLetterOverflow)\n"
                "    }\n"
                "}\n"
                "```"
            ),
            "rejected": "Crie um vetor global ou use RabbitMQ.",
        },

        # 4. Transações de Banco de Dados e Contenção de Lock (Jim Gray / Reuter)
        {
            "prompt": "arruma esse update que ta dando lock no banco",
            "chosen": (
                "<think>\n"
                "Desambiguação segundo Jim Gray & Andreas Reuter ('Transaction Processing: Concepts and Techniques'):\n"
                "1. Diagnóstico do pedido vago: Deadlocks e esperas excessivas em instruções UPDATE decorrem de:\n"
                "   - Ordenação inconsistente de locks (Lock Inversion) quando múltiplas transações simultâneas atualizam registros em ordens diferentes (ex: Tx1 atualiza ID 1 depois ID 2; Tx2 atualiza ID 2 depois ID 1).\n"
                "   - Ausência de índice cobrindo a cláusula WHERE, forçando o motor a promover lock de linha para lock de tabela (*Lock Escalation*).\n"
                "2. Diretrizes da Literatura:\n"
                "   - Regra de Dijkstra para prevenção de Deadlocks: Ordenação canônica e determinística dos recursos (IDs ordenados de forma crescente antes do UPDATE).\n"
                "   - Redução da janela transacional: Executar computações fora da transação e manter aberto o bloco BEGIN...COMMIT apenas durante o I/O estrito.\n"
                "</think>\n"
                "Segundo os princípios clássicos de **Processamento de Transações (Gray & Reuter)**, a eliminação definitiva de deadlocks em operações de atualização concorrente exige duas intervenções estruturais:\n\n"
                "1. **Ordenação Canônica de Identificadores (Prevenção de Deadlock por Inversão de Locks):**\n"
                "```python\n"
                "def update_batch_accounts(conn, updates: list[tuple[int, float]]) -> None:\n"
                "    # Ordena os registros por ID antes de adquirir qualquer lock de linha\n"
                "    sorted_updates = sorted(updates, key=lambda x: x[0])\n"
                "\n"
                "    with conn.cursor() as cur:\n"
                "        cur.execute(\"BEGIN;\")\n"
                "        for account_id, delta in sorted_updates:\n"
                "            cur.execute(\n"
                "                \"\"\"\n"
                "                UPDATE accounts\n"
                "                SET balance = balance + %s,\n"
                "                    updated_at = NOW()\n"
                "                WHERE id = %s;\n"
                "                \"\"\",\n"
                "                (delta, account_id)\n"
                "            )\n"
                "        cur.execute(\"COMMIT;\")\n"
                "```\n\n"
                "2. **Indexação Estrita para Impedir Lock Escalation:**\n"
                "Garantir que a coluna de predicado possua índice B-Tree dedicado para que o PostgreSQL/MySQL bloqueie estritamente a tupla (`RowExclusiveLock`) sem travar a relação inteira:\n"
                "```sql\n"
                "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_accounts_id ON accounts (id);\n"
                "```"
            ),
            "rejected": "Tenta colocar um sleep ou rodar com nolock.",
        },
    ]


def build_literature_corpus() -> Path:
    base_file = SATA_SSD_DATA_DIR / "mark_engineer_train.jsonl"
    all_samples: List[Dict[str, str]] = []

    if base_file.exists():
        with open(base_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    all_samples.append(json.loads(line))

    # Injeta os pares fundamentados na literatura técnica
    lit_pairs = generate_literature_grounded_vague_intent_pairs()
    for _ in range(60):
        all_samples.extend(lit_pairs)

    target_ssd_file = SATA_SSD_DATA_DIR / "mark_engineer_train.jsonl"
    print(f"[Literature Corpus] Gravando {len(all_samples)} amostras de treino atualizadas no SSD SATA: {target_ssd_file}...")

    with open(target_ssd_file, "w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    LOCAL_DATA_DIR.mkdir(parents=True, exist_ok=True)
    local_target = LOCAL_DATA_DIR / "mark_engineer_train.jsonl"
    with open(local_target, "w", encoding="utf-8") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"[Literature Corpus] Sucesso! Tamanho final: {target_ssd_file.stat().st_size / (1024*1024):.2f} MB")
    return target_ssd_file


if __name__ == "__main__":
    build_literature_corpus()
