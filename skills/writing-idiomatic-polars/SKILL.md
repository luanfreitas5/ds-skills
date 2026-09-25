---
name: writing-idiomatic-polars
description: Use when writing or reviewing polars code, when migrating pandas code (apply, groupby-transform, iterrows, loops over groups) to polars, when data is larger than RAM, or when polars code is not faster than the pandas it replaced.
---

# Writing Idiomatic Polars

## Overview

Polars is fast only when the query engine sees the whole computation. Python row functions (`map_elements`), per-group loops, and early `collect()` hide it. **Core principle:** describe the full pipeline as expressions on a `LazyFrame`, then execute once, at the end, with the engine that fits the data size.

## Translation Table

| pandas habit | polars idiom |
|---|---|
| `pd.read_parquet(p)` | `pl.scan_parquet(p)` (lazy; projection and predicate pushdown) |
| `s.apply(f)` with if/elif | `pl.when(...).then(...).when(...).then(...).otherwise(...)` |
| `s.apply(lambda x: x.split("-")[-1].strip())` | `pl.col("s").str.split("-").list.last().str.strip_chars()` |
| `groupby(k)[c].transform(f)` | `f(pl.col(c)).over(k)` |
| `for k, g in df.groupby(k): ...` | `group_by(k).agg(...)` with `sort_by`, `tail`, `filter` inside `agg` |
| `df[c] = ...` repeated | one `with_columns(expr1, expr2, ...)` (parallel) |
| `pd.to_datetime(s)` | `pl.col(c).str.to_datetime("%Y-%m-%d")` — pass the format |
| `df.to_parquet(p)` after load | `lf.sink_parquet(p)` (streams, never materializes) |

## Execution Rules

1. **Lazy by default.** `scan_*` → expressions → one `collect()` / `sink_*` at the end. A `collect()` in the middle ends optimization and loads everything.
2. **Larger than RAM:** `sink_parquet(...)` or `collect(engine="streaming")`. Verify before the full run: `lf.explain(engine="streaming")` shows the plan and whether filters/projections reached the scan; on a sample, `pl.Config.set_verbose(True)` shows the streaming graph.
3. **Two outputs from one scan** re-read the source per `sink_*`. Acceptable for streaming; otherwise `pl.collect_all([lf1, lf2])` shares the common subplan.
4. **Measure.** Time on a sample and compare with the pandas version before claiming "faster".

## When a UDF is Acceptable

Only when no expression exists (external library call, complex parsing), and then:
- vectorised batch: `pl.col(c).map_batches(f, return_dtype=...)` (whole Series per call), not `map_elements` (one Python call per row);
- always pass `return_dtype`, otherwise polars must infer it;
- polars emits `PolarsInefficientMapWarning` with the replacement expression when a lambda maps to a native one — apply the suggestion.

Measured on 200k rows: `map_elements(lambda v: v * 2)` 0.080 s vs `pl.col("v") * 2` 0.008 s; Python loop over groups 0.072 s vs `group_by().agg()` 0.007 s.

## Implementation

```python
import polars as pl


def build_customer_summary(path: str) -> pl.LazyFrame:
    """Resume transações por cliente sem materializar o arquivo inteiro."""
    return (
        pl.scan_parquet(path)
        .filter(pl.col("valor") > 0)
        .with_columns(
            pl.when(pl.col("valor") < 100).then(pl.lit("baixo"))
            .when(pl.col("valor") < 1000).then(pl.lit("medio"))
            .otherwise(pl.lit("alto")).alias("faixa"),
            ((pl.col("valor") - pl.col("valor").mean().over("cliente_id"))
             / pl.col("valor").std().over("cliente_id")).alias("valor_norm"),
        )
        .group_by("cliente_id")
        .agg(
            pl.len().alias("n"),
            pl.col("data").max().alias("ultimo"),
            pl.col("valor").sort_by("data").tail(3).mean().alias("media_3"),
        )
    )


# build_customer_summary("data/raw/transacoes.parquet").sink_parquet("data/processed/resumo.parquet")
```

Validate the output at the stage boundary (REQUIRED SUB-SKILL: validating-data-contracts).

## Common Mistakes

| Mistake | Fix |
|---|---|
| `pl.read_parquet` then filter | `scan_parquet` so the filter is pushed into the reader |
| `.collect()` to "check" mid-pipeline | `lf.head(5).collect()` for inspection; keep the pipeline lazy |
| `map_elements` for if/else or string ops | `when/then/otherwise`, `.str.*`, `.dt.*`, `.list.*` |
| Loop over `group_by` / `partition_by` | Aggregation expressions inside `agg` |
| `str.to_datetime()` without format on big data | Pass the format; inference is slow and can differ per chunk |
| Claiming speedup without timing | Benchmark on a sample |
