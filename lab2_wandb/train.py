"""
train.py - Laboratorio 2 (W&B Sweep)
IMD3005 - MLOps

Treina um RandomForestClassifier no dataset Breast Cancer Wisconsin (data/dataset.csv,
gerado por prepare_data.py), logando no W&B a cada run: metrica de treino, teste,
cross-validation, importancia de features (via permutation_importance) e a
configuracao de hiperparametros usada. Ao final, versiona o dataset e o modelo
deste run como W&B Artifacts.

Nao roda sozinho para varios hiperparametros -- e chamado pelo `wandb agent`
conforme o espaco de busca definido em sweep.yaml.
"""
import wandb
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.inspection import permutation_importance

DATA_PATH = "data/dataset.csv"


def main():
    run = wandb.init(project="lab2-varredura-hiperparametros")
    config = wandb.config

    # nome do run refletindo os hiperparametros principais, mais facil de
    # identificar na lista de runs do que um nome aleatorio
    run.name = f"floresta-{config.n_estimators}arv-prof{config.max_depth}"

    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=["target"])
    y = df["target"]

    # stratify=y: mantem a proporcao de classes em treino/teste -- o dataset nao
    # e 50/50 (63% benigno / 37% maligno), entao isso deixa a avaliacao mais
    # robusta. Nao estava no template do enunciado, adicionado aqui de proposito.
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

    # Importancia das features via permutation importance (funciona p/ qualquer
    # estimador, diferente do feature_importances_ nativo de arvores)
    perm = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    tabela_importancia = wandb.Table(
        data=list(zip(X.columns, perm.importances_mean, perm.importances_std)),
        columns=["caracteristica", "importancia_media", "importancia_desvio"],
    )
    wandb.log({"importancia_features": tabela_importancia})

    # Artifact do dataset (o W&B deduplica pelo hash do arquivo entre runs do sweep,
    # entao isso nao duplica dados a cada run)
    dataset_artifact = wandb.Artifact("dataset-cancer-mama", type="dataset")
    dataset_artifact.add_file(DATA_PATH)
    run.log_artifact(dataset_artifact)

    # Artifact do modelo deste run especifico
    joblib.dump(model, "model.joblib")
    model_artifact = wandb.Artifact(
        f"modelo-{run.id}", type="modelo", metadata=dict(config)
    )
    model_artifact.add_file("model.joblib")
    run.log_artifact(model_artifact)


if __name__ == "__main__":
    main()
