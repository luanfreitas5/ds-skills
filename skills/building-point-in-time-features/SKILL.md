---
name: building-point-in-time-features
description: Use when building a training table from historical or event tables (transactions, payments, logs, CRM snapshots) for labels that have a date, when joining a "current state" table to past labels, or when a model trained on such features scores suspiciously high.
---

# Building Point-in-Time Features

## Overview

A label from 2021 joined to `renda_atual` exported today gives the model information from 2021–2026: it predicts the past using the future. The metric rises, and production (which only has the present) falls. **Core principle:** every label row carries its own cutoff; every feature is computed only from records with timestamp `<` that cutoff; a test fails if any feature uses a later record.

## Rules

1. **Cutoff per label row** (`cutoff_ts`): the moment the prediction would be made (e.g. `data_concessao`). Not one global date.
2. **Event tables** (payments, queries, transactions): join on entity, keep `event_ts < cutoff_ts`, then aggregate over a window ending at the cutoff (`cutoff_ts - 180d <= event_ts < cutoff_ts`).
3. **Slowly changing attributes** (income, score, plan): as-of join — the latest value with `valid_from <= cutoff_ts`. Needs a history table (SCD2 / snapshots).
4. **Only a current snapshot exists?** Its columns are **excluded**, not "flagged". Record the exclusion in the datasheet and ask the data owner for history. A kept leaky feature makes every downstream metric false.
5. **Event must be known at cutoff:** use the timestamp when the record became available (`data_pagamento`, load time), not when it refers to (`data_vencimento`), if they differ.
6. **Split by time** after building features: train on older cutoffs, test on newer (REQUIRED SUB-SKILL: detecting-data-leakage).

## Implementation

```python
import polars as pl


def build_asof_attributes(labels: pl.DataFrame, history: pl.DataFrame) -> pl.DataFrame:
    """Anexa o último valor conhecido de cada atributo até a data de corte (as-of join)."""
    return labels.sort("cutoff_ts").join_asof(
        history.sort("valid_from"),
        left_on="cutoff_ts",
        right_on="valid_from",
        by="cliente_id",
        strategy="backward",  # valid_from <= cutoff_ts
        allow_exact_matches=False,  # registro no próprio instante do corte não é conhecido
    )


def build_window_counts(labels: pl.DataFrame, events: pl.DataFrame, days: int) -> pl.DataFrame:
    """Conta eventos na janela [corte - days, corte) por linha de rótulo."""
    return (
        labels.select("contrato_id", "cliente_id", "cutoff_ts")
        .join(events, on="cliente_id", how="left")
        .with_columns(
            in_window=(pl.col("event_ts") < pl.col("cutoff_ts"))
            & (pl.col("event_ts") >= pl.col("cutoff_ts") - pl.duration(days=days))
        )
        .group_by("contrato_id")
        .agg(pl.col("in_window").sum().alias(f"n_eventos_{days}d"))
    )


def assert_no_future_records(labels: pl.DataFrame, used: pl.DataFrame, ts_col: str) -> None:
    """Falha se algum registro usado numa feature for posterior ou igual ao corte da linha."""
    violations = used.join(labels.select("contrato_id", "cutoff_ts"), on="contrato_id").filter(
        pl.col(ts_col) >= pl.col("cutoff_ts")
    )
    if violations.height:
        raise AssertionError(f"{violations.height} registros posteriores ao corte usados em features")
```

`build_window_counts` keeps labels without events (left join; `sum` of nulls → 0). In the test suite, call `assert_no_future_records` on every intermediate frame that feeds an aggregate, and add a synthetic case: an event one day **after** the cutoff must not change the feature.

```python
from datetime import datetime


def test_event_after_cutoff_is_ignored() -> None:
    """Evento posterior ao corte não altera a contagem."""
    labels = pl.DataFrame({"contrato_id": [1], "cliente_id": [7], "cutoff_ts": [datetime(2023, 6, 1)]})
    before = pl.DataFrame({"cliente_id": [7], "event_ts": [datetime(2023, 5, 1)]})
    after = pl.concat([before, pl.DataFrame({"cliente_id": [7], "event_ts": [datetime(2023, 6, 2)]})])
    n_before = build_window_counts(labels, before, 180)["n_eventos_180d"].item()
    n_after = build_window_counts(labels, after, 180)["n_eventos_180d"].item()
    assert n_before == n_after == 1
```

## Common Mistakes

| Mistake | Fix |
|---|---|
| Join CRM export (`*_atual`) to 2021–2024 labels | Exclude, or as-of join against history |
| "Flag it and keep it, time is short" | Drop the feature; report AUC without it |
| Global cutoff (`< '2024-01-01'`) for all rows | Per-row `cutoff_ts` |
| `<=` cutoff | Strict `<`; same-timestamp records are usually not available yet |
| Window aggregation with inner join | Left join; labels with no events get 0, not dropped |
| Filtering on due date for payments not yet paid at cutoff | Use availability timestamp |
