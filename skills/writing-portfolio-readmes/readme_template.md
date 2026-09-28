# <Título: resultado em uma frase, não o nome da técnica>

[![CI](https://github.com/<owner>/<repo>/actions/workflows/ci.yml/badge.svg)](https://github.com/<owner>/<repo>/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/<owner>/<repo>/graph/badge.svg)](https://codecov.io/gh/<owner>/<repo>)
[![docs](https://github.com/<owner>/<repo>/actions/workflows/docs.yml/badge.svg)](https://<owner>.github.io/<repo>/)
[![License](https://img.shields.io/github/license/<owner>/<repo>)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue) <!-- = matriz do CI -->

> **Resultado:** <modelo> atinge <métrica> = **<valor> (IC 95%: <a>–<b>)** contra **<valor>** do baseline <nome>, em <dataset, n>. <Uma frase sobre o que isso permite decidir.>

## 🎯 Problema

- **Quem decide o quê:** <ex.: equipe de retenção escolhe 20% dos clientes para contato mensal>.
- **Custo dos erros:** <FP custa X, FN custa Y> → **métrica principal:** <métrica> porque <justificativa>.
- **Unidade de predição e momento `t0`:** <1 linha por cliente-mês; features anteriores a t0>.

## 🗂️ Dados

| Item | Valor |
|---|---|
| Fonte e licença | <link> — <licença> |
| Tamanho / período | <n linhas>, <início–fim> |
| Alvo e prevalência | <alvo>, <x%> positivos |
| Documentação | [Datasheet](reports/datasheets/<dataset>.md) |

## 📊 Resultados

Protocolo: <split / CV>, seed <42>; IC por <bootstrap 1000x no teste | t de Student sobre os folds>.

| Modelo | <Métrica> (IC 95%) | <Métrica secundária> |
|---|---|---|
| Baseline: <nome> | <valor (a–b)> | <valor> |
| <Modelo final> | **<valor (a–b)>** | <valor> |

- Diferença vs baseline: <Δ>, <teste> p = <p>. <Se não significativa, dizer.>
- Pior segmento: <segmento> = <valor> (ver [Model Card](reports/model_cards/<modelo>.md)).

![<descrição da figura>](reports/figures/<figura>.png)

## 🔁 Reproduzir

Requisitos: [uv](https://docs.astral.sh/uv/), `make`, <credenciais, se houver: ver `.env.example`>. Tempo: ~<N> min em <CPU/GPU>.

```bash
git clone https://github.com/<owner>/<repo>.git
cd <repo>
```

<!-- verify:start -->
```bash
uv sync --locked
make data train   # os alvos reais do Makefile do projeto
```
<!-- verify:end -->

Saída esperada (Python <3.12>, `uv.lock` do commit `<sha>`): `<métrica>=<valor>`. Os números de Resultados vêm de [`reports/metrics/<arquivo>.json`](reports/metrics/<arquivo>.json).

## 🧱 Estrutura

<Árvore curta: só pastas que o leitor precisa abrir.>

## ⚠️ Limitações

- <Onde o modelo falha, o que não foi testado, viés conhecido.>

## 📄 Licença e citação

MIT — ver [LICENSE](LICENSE). Para citar (só se houver artigo): [`CITATION.cff`](CITATION.cff).
