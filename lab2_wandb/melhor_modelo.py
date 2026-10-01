"""
melhor_modelo.py - Laboratorio 2 (W&B)
IMD3005 - MLOps

Retreina o MELHOR modelo encontrado pelo sweep (hiperparametros do run
"floresta-400arv-profNone", media_f1_macro_cv=0.95769 -- o vencedor entre os
67 runs do sweep w12n99u1), com dois objetivos que a interface do W&B nao
deixou fazer direto a partir de uma tabela ja logada:

  1. Gerar o grafico de feature importance como um BAR CHART nativo do W&B
     (wandb.plot.bar), que renderiza um grafico de verdade -- nao uma tabela
     crua que precisa ser convertida manualmente na interface.
  2. Salvar o artifact do modelo ja com o alias "best" (wandb permite marcar
     isso na hora de logar o artifact, sem precisar ir na interface depois).

    python melhor_modelo.py
"""
import wandb
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.inspection import permutation_importance

DATA_PATH = "data/dataset.csv"

# Hiperparametros do melhor run do sweep w12n99u1 (floresta-400arv-profNone)
MELHOR_CONFIG = {
    "n_estimators": 400,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 2,
    "max_features": "log2",
}


def main():
    run = wandb.init(
        project="lab2-varredura-hiperparametros",
        name="melhor-modelo-final",
        job_type="melhor-modelo",
        config=MELHOR_CONFIG,
        tags=["best"],
    )
    config = wandb.config

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["target"])
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=config.n_estimators,
        max_depth=config.max_depth,
        min_samples_split=config.min_samples_split,
        min_samples_leaf=config.min_samples_leaf,
        max_features=config.max_features,
        random_state=42,
    )
    model.fit(X_train, y_train)

    pontuacao_treino = model.score(X_train, y_train)
    pontuacao_teste = model.score(X_test, y_test)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="f1_macro")

    wandb.log({
        "pontuacao_treino": pontuacao_treino,
        "pontuacao_teste": pontuacao_teste,
        "media_f1_macro_cv": cv_scores.mean(),
        "desvio_f1_macro_cv": cv_scores.std(),
    })

    perm = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    tabela_importancia = wandb.Table(
        data=list(zip(X.columns, perm.importances_mean, perm.importances_std)),
        columns=["caracteristica", "importancia_media", "importancia_desvio"],
    )
    wandb.log({"importancia_features": tabela_importancia})

    # Grafico de barras nativo -- isso E o "grafico de feature importance"
    # pedido no enunciado, pronto pra incorporar ao vivo no Report
    wandb.log({
        "grafico_importancia_features": wandb.plot.bar(
            tabela_importancia,
            "caracteristica",
            "importancia_media",
            title="Importancia das Features (melhor modelo)",
        )
    })

    dataset_artifact = wandb.Artifact("dataset-cancer-mama", type="dataset")
    dataset_artifact.add_file(DATA_PATH)
    run.log_artifact(dataset_artifact)

    joblib.dump(model, "model.joblib")
    model_artifact = wandb.Artifact(
        f"modelo-{run.id}", type="modelo", metadata=dict(config)
    )
    model_artifact.add_file("model.joblib")
    # aliases=["best"] marca o artifact direto, sem precisar ir na interface
    run.log_artifact(model_artifact, aliases=["best"])

    print(f"\nRun: {run.url}")
    print(f"media_f1_macro_cv: {cv_scores.mean():.5f}")
    print("Artifact do modelo marcado com alias 'best'.")

    run.finish()


if __name__ == "__main__":
    main()
