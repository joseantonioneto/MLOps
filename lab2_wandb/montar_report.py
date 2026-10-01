"""
montar_report.py - Laboratorio 2 (W&B)
IMD3005 - MLOps

Monta o W&B Report programaticamente via wandb_workspaces.reports.v2, com
tudo que o enunciado pede: texto do espaco de busca/estrategia/justificativa,
parallel coordinates do sweep (painel ao vivo), melhor resultado, grafico de
feature importance do melhor modelo (painel ao vivo), investigacao de
overfitting, e links dos artifacts.

    python montar_report.py
"""
import wandb_workspaces.reports.v2 as wr

ENTITY = "tomsouzneto-ufrn"
PROJECT = "lab2-varredura-hiperparametros"
SWEEP_ID = "w12n99u1"
MELHOR_RUN_NAME = "melhor-modelo-final"

report = wr.Report(
    entity=ENTITY,
    project=PROJECT,
    title="Laboratorio 2 - Rastreamento de Experimentos com W&B",
    description="IMD3005 - MLOps | RandomForestClassifier no dataset Breast Cancer Wisconsin, otimizado via sweep bayes",
    blocks=[
        wr.H1("Dataset e modelo"),
        wr.P(
            "Dataset: Breast Cancer Wisconsin (scikit-learn), classificacao binaria, "
            "569 amostras, 30 features numericas. Modelo: RandomForestClassifier."
        ),
        wr.H1("Espaco de busca e estrategia do sweep"),
        wr.P(
            "5 hiperparametros: n_estimators [50,100,200,400], max_depth [3,5,10,20,sem limite], "
            "min_samples_split [2,5,10], min_samples_leaf [1,2,4], max_features [sqrt,log2,sem limite] "
            "-- 540 combinacoes possiveis."
        ),
        wr.P(
            "Estrategia: bayes. Um grid exaustivo custaria 540 runs; random search nao usa o "
            "resultado dos runs anteriores para guiar a busca. O bayes aprende com os runs ja "
            "executados e foca nas regioes mais promissoras -- melhor resultado por run gasto, "
            "considerando o limite de runs executados (67)."
        ),
        wr.H1("Parallel Coordinates do Sweep"),
        wr.P(
            "Colunas: os 5 hiperparametros, com media_f1_macro_cv como ultima coluna (metrica de "
            "performance)."
        ),
        wr.PanelGrid(
            runsets=[
                wr.Runset(
                    entity=ENTITY,
                    project=PROJECT,
                    name="Sweep completo",
                    filters=f"Sweep = '{SWEEP_ID}'",
                )
            ],
            panels=[
                wr.ParallelCoordinatesPlot(
                    columns=[
                        wr.ParallelCoordinatesPlotColumn(metric=wr.Config("n_estimators")),
                        wr.ParallelCoordinatesPlotColumn(metric=wr.Config("max_depth")),
                        wr.ParallelCoordinatesPlotColumn(metric=wr.Config("min_samples_split")),
                        wr.ParallelCoordinatesPlotColumn(metric=wr.Config("max_features")),
                        wr.ParallelCoordinatesPlotColumn(metric=wr.Config("min_samples_leaf")),
                        wr.ParallelCoordinatesPlotColumn(metric=wr.SummaryMetric("media_f1_macro_cv")),
                    ],
                    title="Hiperparametros x media_f1_macro_cv",
                )
            ],
        ),
        wr.H1("Melhor resultado"),
        wr.P(
            "n_estimators=400, max_depth=sem limite, min_samples_split=2, min_samples_leaf=2, "
            "max_features=log2. media_f1_macro_cv = 0.9577, entre 67 runs executados."
        ),
        wr.H1("Pergunta exploratoria: ha sinais de overfitting?"),
        wr.P(
            "Comparando treino vs. teste do melhor modelo: pontuacao_treino = 0.9934 vs "
            "pontuacao_teste = 0.9561 -- diferenca de ~3,7 pontos percentuais. Ha sinal de "
            "overfitting leve/moderado: o modelo memoriza o treino melhor do que generaliza, "
            "mas a queda nao e drastica."
        ),
        wr.H1("Feature Importance do melhor modelo"),
        wr.PanelGrid(
            runsets=[
                wr.Runset(
                    entity=ENTITY,
                    project=PROJECT,
                    name="Melhor modelo",
                    filters=f"Name = '{MELHOR_RUN_NAME}'",
                )
            ],
            panels=[
                wr.CustomChart.from_table(
                    table_name="importancia_features",
                    chart_fields={"label": "caracteristica", "value": "importancia_media"},
                    chart_strings={"title": "Importancia das Features (melhor modelo)"},
                )
            ],
        ),
        wr.H1("Artifacts"),
        wr.P(f"Dataset: https://wandb.ai/{ENTITY}/{PROJECT}/artifacts/dataset/dataset-cancer-mama"),
        wr.P(f"Melhor modelo (alias best): https://wandb.ai/{ENTITY}/{PROJECT}/artifacts/modelo/modelo-8fgorzix"),
    ],
)

report.save()
print("Report criado:", report.url)
