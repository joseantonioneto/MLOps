"""
hello_world.py - Etapa 0 (Laboratorio 2)
IMD3005 - MLOps

Checkpoint obrigatorio antes do desafio principal: roda um modelo FIXO
(RandomForestClassifier com hiperparametros padrao) no dataset Iris e loga a
acuracia no W&B. O objetivo aqui nao e o modelo -- e confirmar que conta,
instalacao e autenticacao do W&B estao funcionando.

Pre-requisito: rodar `wandb login` no terminal antes (uma unica vez).

    python hello_world.py
"""
import wandb
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


def main():
    run = wandb.init(project="introducao-wandb", name="linha-base-iris")

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(random_state=42)  # hiperparametros padrao, de proposito
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acuracia = accuracy_score(y_test, y_pred)
    wandb.log({"acuracia": acuracia})

    print(f"acuracia: {acuracia:.4f}")
    print(f"Confira o run em: {run.url}")

    run.finish()


if __name__ == "__main__":
    main()
