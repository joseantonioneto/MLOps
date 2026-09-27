"""
prepare_data.py - Laboratorio 2 (W&B)
IMD3005 - MLOps

Exporta o dataset Breast Cancer Wisconsin (embutido no scikit-learn) para
data/dataset.csv. Esse arquivo alimenta o train.py e tambem e o arquivo
versionado como W&B Artifact do dataset. Rode uma vez antes do sweep:

    python prepare_data.py
"""
import pandas as pd
from sklearn.datasets import load_breast_cancer


def main():
    data = load_breast_cancer(as_frame=True)
    df = data.frame  # ja vem com as colunas de features + coluna "target"

    df.to_csv("data/dataset.csv", index=False)

    print(f"data/dataset.csv salvo: {df.shape[0]} linhas x {df.shape[1]} colunas")
    print("Distribuicao do target (0=maligno, 1=benigno):")
    print(df["target"].value_counts())


if __name__ == "__main__":
    main()
