# MLOPS_IMD
IMD3005 - MLOps — Prof. Adelson de Araújo

Repositório com as atividades práticas da disciplina.

| Laboratório | Tema | Onde está |
|---|---|---|
| Laboratório 1 | ETL + Feature Engineering com Apache Airflow | raiz deste repo (`dags/`, `data/`) — detalhes abaixo |
| Laboratório 2 | Rastreamento de experimentos com Weights & Biases | [lab2_wandb/](lab2_wandb/) — ver README próprio |

---

## Laboratório 1 — ETL + Feature Engineering com Apache Airflow

## O pipeline

`API DataJud/CNJ → extract → transform (feature engineering) → load (features.csv)`

DAG: [dags/dag_feature_engineering.py](dags/dag_feature_engineering.py) — `dag_id = feature_engineering_datajud`

- **extract**: consulta a [API Pública do DataJud](https://datajud-wiki.cnj.jus.br/api-publica/) (CNJ) —
  índice `api_publica_tjrn` — e obtém os N processos mais recentemente atualizados do TJRN (busca ampla,
  sem filtro de classe/assunto, pois o alvo do modelo preditivo ainda não foi definido).
- **transform**: aplica feature engineering (ver técnicas abaixo) sobre os metadados processuais.
- **load**: salva o resultado em `data/features.csv`, sobrescrevendo a cada execução (idempotente).

> Este é o ponto de partida de um projeto real do escritório. Objetivo já definido: features para um
> futuro modelo que preveja **quando um processo será encerrado**. O rótulo ("processo encerrado")
> ainda não foi fechado — ver seção "Sobre o rótulo de encerramento" abaixo.

## Técnicas de Feature Engineering aplicadas (Aula 4)

| Categoria | Técnica | Colunas geradas |
|---|---|---|
| Extração | Features temporais | `dias_desde_ajuizamento`, `dias_desde_ultima_movimentacao`, `tempo_medio_entre_movimentacoes`, `dias_desde_transito_em_julgado` |
| Extração | Sinal derivado de evento | `teve_transito_em_julgado`, `movimentos_apos_transito_em_julgado` |
| Transformação | Padronização (z-score) | `*_zscore` para as 7 variáveis numéricas |
| Transformação | Codificação categórica (one-hot) | `grau_*` — grau da instância (G1/G2/...) |

Colunas descritivas mantidas sem transformação (para exploração futura): `numero_processo`, `tribunal`,
`classe_codigo`/`classe_nome`, `assunto_principal_codigo`/`assunto_principal_nome`, `orgao_julgador_nome`,
`sistema_nome`, `data_ajuizamento`.

## Sobre o rótulo de encerramento (importante para o relatório)

O objetivo final é prever quando um processo será encerrado, mas a API do DataJud **não tem** um campo
`status`/`situação` — só dá pra inferir a partir de `movimentos`. Testamos a hipótese óbvia
("Trânsito em julgado" = processo encerrado) contra os dados reais do TJRN e ela **não se sustentou**:
em vários processos, dezenas de movimentações continuam depois do trânsito em julgado (recursos
especiais, sobrestamentos por recurso repetitivo etc.) — ou seja, uma decisão específica transitou em
julgado, mas o processo como um todo não estava encerrado.

Por isso, nesta etapa (ETL + Feature Engineering) construímos as features relacionadas a esse sinal —
`teve_transito_em_julgado`, `dias_desde_transito_em_julgado`, `movimentos_apos_transito_em_julgado` —
**sem fechar um rótulo binário de "encerrado"**. Isso é uma limitação identificada e documentada, não
resolvida: definir o critério de encerramento com mais rigor fica para uma etapa futura de modelagem.
Essa é uma boa observação para o relatório (é literalmente o tema "rotulação" da Aula 3).

## Sobre a API DataJud

- **Base URL**: `https://api-publica.datajud.cnj.jus.br/api_publica_{tribunal}/_search` (um índice
  Elasticsearch por tribunal — usamos `tjrn`).
- **Autenticação**: header `Authorization: APIKey <chave>`. A chave é **pública e compartilhada**
  (divulgada na [documentação oficial](https://datajud-wiki.cnj.jus.br/api-publica/acesso/)), o CNJ pode
  rotacioná-la a qualquer momento — se o `extract` começar a falhar com 401/403, é o primeiro lugar a checar.
- **Query**: Elasticsearch Query DSL via POST. Busca atual usa `match_all` ordenado por `@timestamp desc`
  (processos mais recentemente atualizados), com paginação simples via `size` (sem `search_after` — não
  necessário para o volume desta atividade).
- **Observação sobre os dados**: como a ordenação é por atualização mais recente, a amostra do TJRN ficou
  concentrada em processos de 2ª instância (`grau=G2`, principalmente Apelação Cível) — é um viés de
  amostragem por conveniência, não um erro do pipeline. Vale mencionar no relatório.

## Configuração via Airflow Variables

Nenhum parâmetro fica hardcoded no código: `DATAJUD_API_KEY`, `DATAJUD_TRIBUNAL`, `DATAJUD_PAGE_SIZE`,
`FEATURES_OUTPUT_PATH` — editáveis em **Admin → Variables** na UI.

## Como rodar

1. Airflow em http://localhost:8081 (login `airflow`/`airflow`).
2. Localize o DAG `feature_engineering_datajud`, habilite-o e dispare com ▶ (Trigger DAG).
3. Acompanhe a Graph View — as 3 tasks devem ficar verdes (extract → transform → load).
4. Confira `data/features.csv` gerado (200 processos × 26 colunas).

```bash
docker compose exec airflow-worker airflow dags trigger feature_engineering_datajud
```

## Estrutura do projeto

```
MLOPS_IMD/
├── dags/
│   └── dag_feature_engineering.py       (Laboratório 1)
├── data/
│   └── features.csv
├── lab2_wandb/                          (Laboratório 2 — ver README próprio)
│   ├── train.py
│   └── sweep.yaml
├── docker-compose.yaml
└── README.md
```

## Notas para o relatório

- **Dificuldade na instalação**: a porta padrão 8080 do Airflow conflitava com outro serviço já rodando
  localmente via WSL. Resolvido remapeando `airflow-apiserver` para a porta `8081` no `docker-compose.yaml`.
- **Dificuldade no pipeline (datas)**: `dataAjuizamento` (formato `AAAAMMDDHHMMSS`, sem timezone) e
  `movimentos[].dataHora` (ISO-8601 com timezone) exigiram tratamento de parsing separado — misturar
  datetime "naive" e "aware" na subtração de datas gerava `TypeError`. Corrigido normalizando tudo para UTC.
- **Dificuldade no pipeline (XCom/JSON)**: como nem todo processo tem `teve_transito_em_julgado`, as
  colunas derivadas desse evento ficam `NaN` para quem não tem. `NaN` não é JSON válido e quebrava a
  passagem de dados entre as tasks (`transform` → `load`) com `ValueError: Out of range float values are
  not JSON compliant: nan`. Corrigido convertendo essas colunas para `None` antes do XCom — mas é preciso
  primeiro converter o DataFrame para dtype `object`, porque o pandas reconverte `None` para `NaN`
  automaticamente em colunas float (`df.astype(object).where(df.notna(), None)`).
- Airflow rodando via **Docker Compose** (CeleryExecutor + PostgreSQL + Redis), imagem oficial
  `apache/airflow:3.3.1`, pois o Airflow não roda nativamente no Windows.
