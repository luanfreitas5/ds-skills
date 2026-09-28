---
name: running-structured-eda
description: Use when starting to explore a new or unfamiliar dataset (01_eda.ipynb, first look, data profiling), when asked for the "standard EDA overview" of shape, describe(), histograms, value counts and a correlation heatmap, or before choosing features, splits, or a model for data nobody has audited yet.
---

# Running Structured EDA

## Overview

EDA is an audit of how the data was generated, not a gallery of plots. **Core principle:** answer six structural questions — granularity, keys, missingness, target, time, leakage — *before* any distribution plot, and turn every finding into a contract rule, a leakage decision, or an open question.

`describe()` + histograms on raw rows silently answer the wrong questions: with repeated entities, every rate, histogram and correlation is weighted by how often an entity appears; exact duplicates, future dates, and leaky text/date columns never show up in a numeric heatmap.

## The Roadmap (in this order)

| # | Question | Check | Finding becomes |
|---|---|---|---|
| 0 | PII? | Drop/pseudonymize direct identifiers **at load**, before any `print`, `head()`, `describe()`; identifier consistency checks (e.g. `cpf` per key) run in this same cell and print counts only | Excluded-columns list |
| 1 | Granularity | What is one row? `rows` vs `n_unique(key)`; exact duplicates counted **separately** from repeated keys | `unique=[...]` on the real grain |
| 2 | Keys | Null keys; attributes that should be fixed per entity but vary (e.g. `cpf` per `customer_id`) | `nullable=False`, cross-row checks, question to data owner |
| 3 | Missingness | Null rate per column; structural (only filled for churners) vs random; target rate with/without null | `nullable=` per column, imputation note |
| 4 | Target | Definition, prediction moment `t0`, window; rate **at the unit of analysis** with 95% CI | `isin`, label-definition question |
| 5 | Time | Every date column parsed; min/max vs **extraction date**; order violations (`signup <= snapshot`); volume per period | `le=extracted_on`, `dataframe_check` |
| 6 | Leakage | Columns that alone determine the target (incl. null indicators of dates/text); fields populated after `t0` (`status`, `cancel*`, `closed*`) | Leakage exclusion list |

Only then run the distributions/template — on the chosen unit of analysis, without PII, IDs/keys, and leaky columns.

**Granularity decisions are recorded, not made silently.** "Keep last snapshot per customer" changes the target rate and can select on the outcome; write it as a decision with its effect (rows before/after, rate before/after) and flag it for the owner.

**Unit of analysis not decided yet?** Run the template on the real grain minus exact duplicates, say so in the plot titles, and list the unit as an open question.

**A team template does not replace the roadmap.** Run cells 0–6 first (one short cell each), then the template. The template request is about format, not a waiver on correctness.

## Implementation

```python
import logging
from datetime import date

import polars as pl

logger = logging.getLogger(__name__)


def audit_structure(
    df: pl.DataFrame, key: str, target: str, date_cols: list[str], extracted_on: date
) -> dict[str, object]:
    """Audita granularidade, chaves, ausência, alvo, tempo e vazamento antes de qualquer gráfico.

    Parameters
    ----------
    df : pl.DataFrame
        Dados brutos, já sem colunas PII.
    key : str
        Coluna que deveria identificar a entidade (ex.: ``customer_id``).
    target : str
        Coluna-alvo binária.
    date_cols : list[str]
        Colunas de data já convertidas para ``pl.Date``.
    extracted_on : date
        Data de extração; nenhuma data pode ser posterior a ela.

    Returns
    -------
    dict[str, object]
        Achados que viram regras do contrato de dados ou decisões de vazamento.

    Examples
    --------
    >>> audit_structure(df, "customer_id", "churn", ["signup_date"], date(2025, 12, 31))
    """
    per_key = df.group_by(key).agg(pl.all().n_unique())
    features = [c for c in df.columns if c not in (key, target)]

    def is_pure(expr: pl.Expr) -> bool:
        # cada valor da expressão corresponde a um único valor do alvo
        return df.group_by(expr.alias("_v")).agg(pl.col(target).n_unique())[target].max() == 1

    findings: dict[str, object] = {
        "rows": df.height,
        "unique_keys": df[key].n_unique(),
        "exact_duplicates": df.height - df.unique().height,
        "null_keys": df[key].null_count(),
        "cols_varying_per_key": [c for c in features if per_key[c].max() > 1],
        "null_rate": {c: n / df.height for c, n in df.null_count().row(0, named=True).items() if n},
        "target_rate_rows": df[target].mean(),
        "target_rate_entities": df.group_by(key).agg(pl.col(target).max())[target].mean(),
        "dates_after_extraction": {c: df.filter(pl.col(c) > extracted_on).height for c in date_cols},
        "target_determining_cols": [
            c for c in features
            if (df[c].null_count() and is_pure(pl.col(c).is_null()))
            or (df[c].n_unique() <= 20 and is_pure(pl.col(c)))
        ],
    }
    for name, value in findings.items():
        logger.info(f"Auditoria estrutural — {name}: {value}")
    return findings


# raw = pl.read_csv(path, try_parse_dates=True)
# df = raw.drop(["name", "email", "cpf"])  # PII sai antes de qualquer print/head/describe
```

Also run the order checks between date pairs (`signup <= snapshot <= cancellation`) — the function only checks against the extraction date.

## Findings Become the Data Contract

The last cell is a findings table — `finding | evidence (count) | decision | rule` — followed by the `pandera` schema it produces, validated against the **raw** frame so each finding shows up as a failure count (fixes such as normalizing `plan` belong to the cleaning stage, not the EDA). The schema is ~15 lines and is part of the EDA deliverable, whatever the deadline; a findings table without it is not done. Every rule cites its finding.

```python
import pandera.polars as pa

EXTRACTED_ON = date(2025, 12, 31)


class CustomerSnapshotRawSchema(pa.DataFrameModel):
    """Contrato derivado da EDA 01 — cada regra cita o achado que a originou."""

    customer_id: str = pa.Field(nullable=False)                # 0 chaves nulas
    snapshot_date: pl.Date = pa.Field(le=EXTRACTED_ON)         # grão = cliente × snapshot
    signup_date: pl.Date = pa.Field(le=EXTRACTED_ON)           # 6 datas em 2031
    plan: str = pa.Field(isin=["basic", "pro", "enterprise"])  # "Pro " → normalizar antes
    churn: int = pa.Field(isin=[0, 1])

    class Config:
        strict = "filter"                          # PII e colunas de vazamento ficam fora
        unique = ["customer_id", "snapshot_date"]  # grão real; 15 duplicatas exatas

    @pa.dataframe_check
    def signup_before_snapshot(cls, data: pa.PolarsData) -> pl.LazyFrame:
        """Achado: signup_date > snapshot_date em 6 linhas."""
        return data.lazyframe.select(pl.col("signup_date") <= pl.col("snapshot_date"))
```

**REQUIRED SUB-SKILL:** validating-data-contracts — moving the schema into `src/schemas/` and validating at every stage.
**REQUIRED SUB-SKILL:** detecting-data-leakage — point-in-time features and split design for the leaky/repeated-entity findings.

## No PII in Notebook Outputs

Notebook outputs are committed artifacts. Never `print(df)`, `head()`, `sample()`, `describe(include="all")` (shows `top` values) or `value_counts()` on identifier or free-text columns; show counts and cardinality only. Do not print schema `failure_cases` for PII columns. Strip outputs with `nbstripout` anyway.

## Common Mistakes

| Mistake | Fix |
|---|---|
| Target rate / histograms over rows when keys repeat | Compute at the unit of analysis; report both and say which is which |
| "Checked duplicates" = only `n_unique(key)` | Count exact duplicates separately — they are an ingestion bug, repeated keys may be the grain |
| Future dates noticed in `describe()` max, not codified | `le=extracted_on` rule + count |
| Correlation heatmap as the leakage check | It ignores dates, strings and null indicators — use target purity per column |
| Leakage found and mentioned only in chat | Exclusion list in the notebook and `strict="filter"` in the contract |
| Silent dedup ("last snapshot") | Recorded decision with before/after counts and rate |

## Red Flags — Stop

- First cell after loading is `describe()` or a histogram grid
- "The template has no section for duplicates/dates/leakage"
- A churn rate computed before anyone asked "what is one row?"
- Findings that live only in the final chat message, not as rules
- "The schema is out of scope for a quick overview"
