"""
DAG: feature_engineering_datajud
IMD3005 - MLOps | Laboratório 1 - ETL + Feature Engineering com Apache Airflow

Pipeline ETL que extrai dados processuais públicos da API do DataJud (CNJ), aplica
Feature Engineering e persiste o resultado em um arquivo CSV, para uso futuro em um
sistema de Machine Learning (ex: prever tempo de tramitação, priorizar processos, etc.).

Este pipeline é o ponto de partida de um projeto real do escritório: os dados vêm da
API Pública do DataJud (https://datajud-wiki.cnj.jus.br/api-publica/), a base oficial
do CNJ com metadados de processos de todos os tribunais do país.

Objetivo do futuro modelo (já definido, rótulo ainda não): prever quando/quanto tempo
falta para um processo ser encerrado. A API não tem um campo "status"/"situação" --
"encerrado" só pode ser inferido a partir de `movimentos`. Investigando dados reais do
TJRN, "Trânsito em julgado" pareceu o sinal mais óbvio, mas isoladamente NÃO é
confiável: em vários processos há dezenas de movimentações depois dele (recursos
especiais, sobrestamentos etc.) -- ou seja, uma decisão transitou em julgado, mas o
processo como um todo não necessariamente encerrou. Por isso, nesta etapa (ETL +
Feature Engineering), construímos as features relacionadas a esse sinal SEM fechar
um rótulo binário de "encerrado" -- essa é uma limitação identificada e documentada,
não resolvida, e fica registrada aqui de propósito para a definição do rótulo (e do
critério de encerramento) ser feita com mais cuidado numa etapa futura de modelagem.

Fluxo: DataJud (CNJ) -> Extract -> Transform (Feature Engineering) -> Load -> CSV

Técnicas de Feature Engineering aplicadas (vistas na Aula 4):
  1. Extração      -> features temporais: tempo desde o ajuizamento, tempo desde a
                       última movimentação, tempo médio entre movimentações, tempo
                       desde o último "trânsito em julgado" (quando existe)
  2. Transformação  -> padronização (z-score) das variáveis numéricas
  3. Transformação  -> codificação de variável categórica (one-hot do grau)

Parâmetros de configuração ficam em Airflow Variables (tribunal, quantidade de
processos, chave de API, caminho de saída) — não é necessário alterar o código do
DAG para mudar o tribunal consultado ou o volume de dados.
"""

from __future__ import annotations

import datetime as dt
import unicodedata

import pandas as pd
import requests

from airflow.sdk import DAG, Variable, task

# Nome do movimento (Tabela Processual Unificada do CNJ) usado como sinal de que uma
# decisão do processo transitou em julgado. Ver docstring do módulo: sozinho, esse
# sinal NÃO garante que o processo inteiro está encerrado.
MOVIMENTO_TRANSITO_EM_JULGADO = "transito em julgado"

# ---------------------------------------------------------------------------
# Configuração via Airflow Variables (com defaults sensatos para a 1ª execução)
# ---------------------------------------------------------------------------
DEFAULTS = {
    # Chave pública divulgada pelo CNJ na documentação oficial do DataJud.
    # É compartilhada por todos os usuários da API e pode ser rotacionada pelo CNJ
    # a qualquer momento (ver https://datajud-wiki.cnj.jus.br/api-publica/acesso).
    "DATAJUD_API_KEY": "cDZHYzlZa0JadVREZDJCendQbXY6SkJlTzNjLV9TRENyQk1RdnFKZGRQdw==",
    "DATAJUD_TRIBUNAL": "tjrn",       # alias do índice: api_publica_{tribunal}
    "DATAJUD_PAGE_SIZE": "200",       # quantos processos extrair por execução
    "FEATURES_OUTPUT_PATH": "/opt/airflow/data/features.csv",
}

DATAJUD_BASE_URL = "https://api-publica.datajud.cnj.jus.br"


def get_config() -> dict:
    """Lê os parâmetros do pipeline das Airflow Variables, criando-as com um
    valor padrão na primeira execução caso ainda não existam."""
    return {key: Variable.get(key, default=default) for key, default in DEFAULTS.items()}


def _parse_datajud_datetime(value: str | None) -> dt.datetime | None:
    """dataAjuizamento vem como 'AAAAMMDDHHMMSS' (sem separadores); já os campos
    dentro de `movimentos[].dataHora` vêm em ISO-8601 ('...T..Z'). Esta função
    trata os dois formatos com tolerância a valores ausentes/inválidos."""
    if not value:
        return None
    try:
        if "T" in value:
            return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
        # 'AAAAMMDDHHMMSS' não traz timezone; assume UTC para ficar comparável
        # com os demais datetimes (todos "aware") usados no pipeline.
        return dt.datetime.strptime(value, "%Y%m%d%H%M%S").replace(tzinfo=dt.timezone.utc)
    except ValueError:
        return None


def _normalize_text(value: str | None) -> str:
    """Remove acentos e caixa, para comparar nomes de movimentação de forma
    tolerante (ex: 'Trânsito em julgado' == 'TRANSITO EM JULGADO')."""
    if not value:
        return ""
    sem_acento = unicodedata.normalize("NFKD", value)
    sem_acento = "".join(c for c in sem_acento if not unicodedata.combining(c))
    return sem_acento.lower()


with DAG(
    dag_id="feature_engineering_datajud",
    description="ETL + Feature Engineering (dados processuais do CNJ/DataJud) com Apache Airflow",
    schedule=None,  # execução manual pela interface, conforme o laboratório
    start_date=dt.datetime(2026, 1, 1),
    catchup=False,
    tags=["mlops", "etl", "feature-engineering", "laboratorio1", "datajud", "cnj"],
) as dag:

    @task(task_id="extract")
    def extract_task() -> list[dict]:
        """1. Extração: busca processos públicos na API do DataJud (CNJ).

        Busca ampla (match_all, ordenado pelos mais recentemente atualizados),
        sem filtro de classe/assunto -- o alvo do modelo preditivo ainda será
        definido, então a extração deve permanecer genérica."""
        config = get_config()
        tribunal = config["DATAJUD_TRIBUNAL"]
        url = f"{DATAJUD_BASE_URL}/api_publica_{tribunal}/_search"

        headers = {
            "Authorization": f"APIKey {config['DATAJUD_API_KEY']}",
            "Content-Type": "application/json",
        }
        body = {
            "size": int(config["DATAJUD_PAGE_SIZE"]),
            "query": {"match_all": {}},
            "sort": [{"@timestamp": {"order": "desc"}}],
        }

        response = requests.post(url, headers=headers, json=body, timeout=60)
        response.raise_for_status()  # falha explicitamente se a API não responder OK
        payload = response.json()

        hits = payload.get("hits", {}).get("hits", [])
        if not hits:
            raise ValueError(f"A API não retornou nenhum processo para o tribunal '{tribunal}'.")

        # mantém apenas o corpo do documento (_source) de cada processo
        return [hit["_source"] for hit in hits]

    @task(task_id="transform")
    def transform_task(records: list[dict]) -> list[dict]:
        """2. Feature Engineering: extrai e transforma features a partir dos
        metadados processuais brutos retornados pela API."""
        now = dt.datetime.now(dt.timezone.utc)
        rows = []

        for proc in records:
            classe = proc.get("classe") or {}
            assuntos = proc.get("assuntos") or []
            orgao = proc.get("orgaoJulgador") or {}
            sistema = proc.get("sistema") or {}
            movimentos = proc.get("movimentos") or []

            data_ajuizamento = _parse_datajud_datetime(proc.get("dataAjuizamento"))

            # movimentações com data válida, ordenadas cronologicamente, mantendo
            # o nome junto da data (necessário para localizar o trânsito em julgado)
            movimentos_ordenados = sorted(
                (
                    (data, m.get("nome"))
                    for m in movimentos
                    if (data := _parse_datajud_datetime(m.get("dataHora")))
                ),
                key=lambda item: item[0],
            )
            datas_movimentos = [data for data, _ in movimentos_ordenados]

            row = {
                "numero_processo": proc.get("numeroProcesso"),
                "tribunal": proc.get("tribunal"),
                "grau": proc.get("grau"),
                "classe_codigo": classe.get("codigo"),
                "classe_nome": classe.get("nome"),
                "assunto_principal_codigo": assuntos[0].get("codigo") if assuntos else None,
                "assunto_principal_nome": assuntos[0].get("nome") if assuntos else None,
                "orgao_julgador_nome": orgao.get("nome"),
                "sistema_nome": sistema.get("nome"),
                "data_ajuizamento": data_ajuizamento.date().isoformat() if data_ajuizamento else None,
                "numero_assuntos": len(assuntos),
                "numero_movimentacoes": len(movimentos),
            }

            # --- Técnica 1 (Extração): features temporais ---
            # Capturam a "idade" e o ritmo de tramitação do processo, essenciais
            # para qualquer modelo que dependa de tempo (ex: prever duração,
            # priorizar processos parados há muito tempo).
            if data_ajuizamento:
                row["dias_desde_ajuizamento"] = (now - data_ajuizamento).days
            else:
                row["dias_desde_ajuizamento"] = None

            if datas_movimentos:
                row["dias_desde_ultima_movimentacao"] = (now - datas_movimentos[-1]).days
            else:
                row["dias_desde_ultima_movimentacao"] = None

            if len(datas_movimentos) >= 2:
                duracao_dias = (datas_movimentos[-1] - datas_movimentos[0]).days
                row["tempo_medio_entre_movimentacoes"] = duracao_dias / (len(datas_movimentos) - 1)
            else:
                row["tempo_medio_entre_movimentacoes"] = 0.0

            # --- Features para o futuro modelo de "tempo até o encerramento" ---
            # Sinalizam a presença e a posição do trânsito em julgado no histórico,
            # SEM assumir que ele significa "processo encerrado" (ver docstring do
            # módulo). "movimentos_apos_transito_em_julgado" alto é evidência de que
            # o processo seguiu tramitando -- útil justamente para não confundir com
            # encerramento real.
            indices_transito = [
                i
                for i, (_, nome) in enumerate(movimentos_ordenados)
                if MOVIMENTO_TRANSITO_EM_JULGADO in _normalize_text(nome)
            ]
            if indices_transito:
                ultimo_idx = indices_transito[-1]
                data_ultimo_transito = movimentos_ordenados[ultimo_idx][0]
                row["teve_transito_em_julgado"] = 1
                row["dias_desde_transito_em_julgado"] = (now - data_ultimo_transito).days
                row["movimentos_apos_transito_em_julgado"] = len(movimentos_ordenados) - 1 - ultimo_idx
            else:
                row["teve_transito_em_julgado"] = 0
                row["dias_desde_transito_em_julgado"] = None
                row["movimentos_apos_transito_em_julgado"] = None

            rows.append(row)

        df = pd.DataFrame(rows)

        # linhas sem data de ajuizamento válida não permitem calcular as features
        # temporais principais; descartadas para manter o dataset consistente
        df = df.dropna(subset=["data_ajuizamento", "dias_desde_ajuizamento"]).reset_index(drop=True)

        # --- Técnica 2 (Transformação): padronização (z-score) ---
        # Coloca as variáveis numéricas na mesma escala (média 0, desvio padrão 1),
        # importante para algoritmos sensíveis à escala (ex: k-NN, regressão linear).
        # Observação: dias_desde_transito_em_julgado e movimentos_apos_transito_em_julgado
        # só existem para processos que já tiveram trânsito em julgado -- o z-score
        # dessas colunas fica NaN nos demais, o que é o comportamento correto (não é
        # um valor "zero", é "não se aplica").
        numeric_cols = [
            "numero_assuntos",
            "numero_movimentacoes",
            "dias_desde_ajuizamento",
            "dias_desde_ultima_movimentacao",
            "tempo_medio_entre_movimentacoes",
            "dias_desde_transito_em_julgado",
            "movimentos_apos_transito_em_julgado",
        ]
        for col in numeric_cols:
            mean, std = df[col].mean(), df[col].std()
            df[f"{col}_zscore"] = 0.0 if std == 0 or pd.isna(std) else (df[col] - mean) / std

        # --- Técnica 3 (Transformação): codificação de variável categórica ---
        # "Grau" (G1/G2/...) é categórica; one-hot encoding evita impor uma ordem
        # numérica artificial entre as instâncias processuais.
        grau_dummies = pd.get_dummies(df["grau"], prefix="grau").astype(int)
        df = pd.concat([df, grau_dummies], axis=1)

        # NaN (ex: colunas de trânsito em julgado para processos que nunca tiveram
        # esse movimento) não é JSON válido -- convertido para None antes do XCom.
        # Precisa virar dtype "object" primeiro: em coluna float, o pandas reconverte
        # None para NaN automaticamente (comportamento conhecido do .where/.fillna).
        df = df.astype(object)
        df = df.where(df.notna(), None)
        return df.to_dict(orient="records")

    @task(task_id="load")
    def load_task(records: list[dict]) -> str:
        """3. Persistência: salva o conjunto de features em features.csv.

        O arquivo é sempre sobrescrito com o resultado da execução atual, o que
        torna o pipeline idempotente (pode ser executado novamente sem exigir
        limpeza manual e sem depender de execuções anteriores)."""
        config = get_config()
        output_path = config["FEATURES_OUTPUT_PATH"]

        df = pd.DataFrame(records)
        df.to_csv(output_path, index=False)

        return f"{len(df)} linhas x {len(df.columns)} colunas salvas em {output_path}"

    load_task(transform_task(extract_task()))
