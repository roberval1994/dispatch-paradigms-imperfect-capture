# Comparação de Paradigmas de Despacho sob Captação Imperfeita de Demanda

Projeto de Pesquisa Operacional, didático e reprodutível, que compara empiricamente
três paradigmas de **despacho de recursos** sob **captação imperfeita de demanda**
(nem toda a demanda real é observada pelo sistema no momento da decisão):

1. **Predict-then-Optimize (PTO)** — previsão de parâmetros operacionais seguida de otimização matemática.
2. **Reinforcement Learning (RL)** — política de despacho sequencial aprendida por reforço.
3. **MIP/Online** — programação inteira mista com decisões online e reotimização em *rolling horizon*.

Todo o estudo está encapsulado em um **único Jupyter Notebook**: [`main_project.ipynb`](main_project.ipynb),
executável de cima para baixo em um ambiente limpo.

---

## Pergunta científica

> Sob captação imperfeita de demanda, qual paradigma de despacho entrega maior
> **disponibilidade** (fração da demanda **real** atendida) e como cada um degrada
> quando a confiabilidade da captação cai?

A avaliação é sempre feita contra a **demanda real** *a posteriori*, evitando vazamento
de informação: no momento da decisão, os paradigmas só enxergam a **demanda observada**.

---

## Formulação do problema (resumo)

- **Recursos**: frota finita de `K` veículos alocáveis a `Z` zonas por passo de tempo.
- **Horizonte**: dia discretizado em passos de `Δt` (padrão 15 min → 96 passos/dia).
- **Processos separados explicitamente**:
  - Demanda real `D[z,t]` (ground truth, usada só na avaliação);
  - Demanda observada `O[z,t]` (filtrada pela captação imperfeita);
  - Informação na decisão `I_t` (histórico de observadas + calendário/zona);
  - Avaliação contra `D[z,t]`.
- **Métrica principal — disponibilidade**:

  ```
  disponibilidade = Σ min(D[z,t], c · a[z,t]) / Σ D[z,t]
  ```

  onde `a[z,t]` é o número de veículos alocados e `c` a capacidade de serviço por veículo.

- **Objetivo comum (maximização)**: maximizar demanda atendida, com penalidades
  opcionais de ociosidade e de reposicionamento.

### Cenários de captação imperfeita

| Cenário | Captação média | Mecanismo |
|---------|----------------|-----------|
| Alta    | ~0.90          | quase completa, ruído leve |
| Média   | ~0.65          | dependência de hora/zona (MAR) |
| Baixa   | ~0.40          | forte dependência + componente MNAR (picos subcaptados) |

---

## Dados

- **Fonte**: [NYC TLC Yellow Taxi Trip Records](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
  (domínio público), formato Parquet.
- **Abordagem híbrida**: os dados públicos fornecem a **dinâmica da demanda real**;
  a **captação imperfeita** é uma camada de simulação explícita e configurável por cima.
- O notebook baixa o arquivo na primeira execução e faz cache em `data/`
  (ignorado pelo Git).

---

## Como executar

```bash
# 1. (opcional) crie um ambiente virtual
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# Linux/macOS:
# source .venv/bin/activate

# 2. instale as dependências
pip install -r requirements.txt

# 3. abra o notebook e execute de cima para baixo
jupyter notebook main_project.ipynb
```

> **Solver MIP**: o projeto usa **Gurobi** se houver licença disponível e cai
> automaticamente para **CBC** (via PuLP) caso contrário. Nenhuma ação manual é necessária.

---

## Estrutura do notebook

| Seção | Conteúdo |
|-------|----------|
| 1 | Introdução e definição formal do problema |
| 2 | Carga de dados, EDA e simulação da captação imperfeita |
| 3 | Ambiente de simulação comum e baselines (ingênua, oráculo) |
| 4 | Paradigma 1 — Predict-then-Optimize |
| 5 | Paradigma 2 — Reinforcement Learning |
| 6 | Paradigma 3 — MIP/Online com reotimização |
| 7 | Protocolo experimental e comparação justa |
| 8 | Discussão, conclusões e limitações |

---

## Reprodutibilidade

- Sementes fixas para geração de captação, treino de RL e amostragens.
- Split **temporal** (sem embaralhar no tempo) em treino/validação/teste.
- Mesmo ambiente de simulação, mesmos cenários e mesmas sementes para os três paradigmas.

---

## Licença

Código sob licença MIT (ver [`LICENSE`](LICENSE)). Os dados do NYC TLC são de domínio
público e **não** são redistribuídos neste repositório.
