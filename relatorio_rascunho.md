# Laboratório 1 — ETL + Feature Engineering com Apache Airflow
IMD3005 - MLOps

**Nome:** _[preencher]_

---

## 1. Dados

**API utilizada:** API Pública do DataJud, do CNJ (Conselho Nacional de Justiça) —
https://datajud-wiki.cnj.jus.br/api-publica/.

**Tipo de dados:** informações públicas sobre processos judiciais brasileiros —
número do processo, tribunal, classe, assunto, órgão julgador, data de abertura e
o histórico de movimentações. Não traz nomes de partes ou advogados (por proteção
de dados).

Foram extraídos os 200 processos mais recentes do TJRN.

**Por que essa fonte:** é o começo de um projeto real para o escritório onde
trabalho — usar dados de processos para, no futuro, ajudar a estimar quanto tempo
falta para um processo ser encerrado.

---

## 2. Feature Engineering

| Técnica | Features criadas | Para que serve |
|---|---|---|
| Features de tempo | dias desde a abertura, dias desde a última movimentação, tempo médio entre movimentações | Medir a "idade" e o ritmo do processo |
| Sinal de trânsito em julgado | teve trânsito em julgado (sim/não), dias desde esse evento, quantas movimentações vieram depois | Indício de processo perto do fim |
| Padronização (z-score) | versão padronizada de cada número acima | Coloca tudo na mesma escala, sem uma variável "pesar mais" só por ter números maiores |
| One-hot encoding | grau (G1 / G2) virou duas colunas de 0 e 1 | Modelo não entende texto, só número — e não faz sentido dar uma ordem para "G1" e "G2" |

**Por que encerramento não virou um rótulo pronto:** a API não diz se um processo
está "encerrado". Tentei usar "trânsito em julgado" como esse sinal, mas percebi
nos dados que isso não é confiável — achei processos com dezenas de movimentações
*depois* do trânsito em julgado (recursos, por exemplo). Por isso só deixei as
features prontas; decidir a regra exata de "processo encerrado" fica para uma
etapa futura.

---

## 3. Pipeline no Airflow

**[Print do DAG executado com sucesso aqui]**

**Fluxo:** API do CNJ → `extract` (busca os processos) → `transform` (calcula as
features) → `load` (salva em `features.csv`).

**Como executei e verifiquei:** disparei o DAG pela interface do Airflow e
acompanhei as 3 tarefas ficarem verdes na tela. Depois conferi o arquivo
`features.csv` gerado (200 linhas) e rodei o DAG de novo para confirmar que ele
funciona sem precisar mexer em nada manualmente.

**Logs:** foram úteis para achar dois erros durante o desenvolvimento — um por
misturar datas com e sem fuso horário, outro porque valores vazios (`NaN`) não
podem ser passados entre as tarefas em formato JSON. Os dois foram corrigidos no
código.

**Benefícios do Airflow para MLOps:** automatiza a execução (não preciso rodar
scripts manualmente), garante a ordem certa entre as etapas, permite agendar
execuções futuras, mostra o status de cada tarefa e seus logs, e permite reexecutar
só a parte que falhou em vez do pipeline inteiro.

**ETL x ELT:** uso ETL aqui porque o destino final é um arquivo CSV simples, sem
poder de processamento — então a transformação precisa acontecer antes de salvar.
ELT faz mais sentido quando o destino é um banco/data warehouse robusto, e vale a
pena guardar o dado bruto para transformar de várias formas depois.

**Dificuldades:** Airflow não roda direto no Windows (usei Docker); a porta padrão
8080 conflitava com outro programa já rodando no meu PC (troquei para 8081); e os
dois bugs de código já citados acima.

---

## Checklist

- [x] API funcionando
- [x] DAG com 3 tarefas
- [x] Duas ou mais técnicas de Feature Engineering, justificadas
- [ ] Print do DAG rodando com sucesso — **pendente**
- [x] `features.csv` gerado
- [x] Testei rodar de novo
- [x] Três perguntas do relatório respondidas
- [ ] ZIP final com `dag_feature_engineering.py` + `features.csv` + `relatorio.pdf`
