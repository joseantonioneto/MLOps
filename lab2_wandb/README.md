# Laboratório 2 — Rastreamento de Experimentos com Weights & Biases
IMD3005 - MLOps

## Dataset e modelo escolhidos

**Breast Cancer Wisconsin** (embutido no scikit-learn) — classificação binária,
569 amostras, 30 features numéricas, 63%/37% de balanceamento entre as classes.
Escolhido no lugar do `features.csv` do Lab 1 porque este último não tem uma
coluna de rótulo definida (decisão tomada de propósito no Lab 1 — ver o README
daquele laboratório). O foco aqui é praticar o W&B, não a complexidade do modelo.

**Modelo**: `RandomForestClassifier`, com 5 hiperparâmetros no espaço de busca
(`n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`).

## O que já está pronto

| Arquivo | O que faz |
|---|---|
| [prepare_data.py](prepare_data.py) | Exporta o dataset para `data/dataset.csv` (já rodei, o arquivo existe) |
| [hello_world.py](hello_world.py) | Etapa 0 — RandomForest com hiperparâmetros padrão no Iris |
| [train.py](train.py) | Script principal, chamado pelo `wandb agent` a cada run do sweep |
| [sweep.yaml](sweep.yaml) | Espaço de busca + estratégia `bayes` (justificada no próprio arquivo) |

Ambiente virtual em `.venv/`, com `wandb`, `scikit-learn`, `pandas` e `joblib`
já instalados.

## O que só você consegue fazer (Etapa 0 do enunciado)

Isso envolve sua conta pessoal — não posso fazer por você:

1. **Criar conta gratuita** em https://wandb.ai (se ainda não tiver)
2. Abrir um terminal **nesta pasta** (`lab2_wandb/`) e ativar o ambiente virtual:
   ```powershell
   .venv\Scripts\activate
   ```
3. Rodar `wandb login` — ele abre o navegador ou pede pra colar a API key
   (Settings → API keys, no site do wandb.ai). Isso fica salvo localmente, não
   precisa repetir depois.
4. Rodar o hello world:
   ```powershell
   python hello_world.py
   ```
5. Conferir no dashboard (o terminal imprime o link do run) que a métrica
   `accuracy` apareceu.

## Depois disso — o desafio principal

Com o login feito, o sweep todo pode ser feito comigo daqui:

```powershell
wandb sweep sweep.yaml        # retorna um SWEEP_ID
wandb agent <SWEEP_ID> --count 40
```

Isso vai rodar 40 combinações de hiperparâmetros, logando tudo automaticamente.
Depois disso, entramos na parte de análise (parallel coordinates, feature
importance, marcar o melhor artifact) e montagem do Report.

## Entregável

Link do W&B Report (não é zip como o Lab 1) — vamos montar isso juntos depois
do sweep rodar.
