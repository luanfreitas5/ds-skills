---
name: optimizing-sql-queries
description: Use when a SQL query is slow, times out, got slower than it used to be, or costs too much in a warehouse (BigQuery bytes billed, Snowflake credits, Postgres/DuckDB runtime), when someone asks to "just make this query faster", or before adding an index, partition, or clustering key.
---

# Optimizing SQL Queries

## Overview

Rewriting a slow query by intuition changes several things at once, fixes the wrong bottleneck, and often changes the result. **Core principle:** the plan of the *original* query is the diagnosis. Capture it with real numbers, change one thing aimed at the most expensive node, re-measure, and prove the result is identical.

## The Loop

1. **Baseline the original.** Plan + actual metrics (time, rows/bytes read, partitions scanned) — commands below. If it "used to be fast", compare with the plan/job of a fast run first (data growth, stale statistics, dropped partition filter, schema change).
2. **Find the costliest node** — the one with the most time, rows, or bytes. Optimize only that.
3. **One hypothesis, one change**, stated as "node X reads N rows because Y; after the change it should read M".
4. **Re-run the plan and compare numbers.** No number moved → revert; the hypothesis was wrong.
5. **Prove equivalence** on the same data: `EXCEPT ALL` in both directions returns 0 rows (Implementation). On a billed warehouse, run it on a small slice (one partition, a sample of keys, or a dev copy with the same types and time zone), not the full original again.
6. **Report before → after** per change (time, bytes/rows, partitions).

## When You Cannot Run the Query Yourself

Deliver in this order:
1. The exact commands to capture the baseline plan of the original query, and what to paste back.
2. What each suspicious plan node would mean (table below).
3. Candidate changes, **each separately**, with the hypothesis and the plan signal that confirms it ("`Seq Scan on orders` becomes `Index Scan`"; "bytes processed drops from 3 TB to ~100 GB").
4. The equivalence query.

Never propose `CREATE INDEX`, re-partitioning, or clustering before listing what exists (`\d orders`, `pg_indexes`, `INFORMATION_SCHEMA.TABLE_OPTIONS`, `SHOW TABLES` clustering_key) and checking selectivity: an index on a predicate that keeps 1/12 of the table barely helps (measured below), and each index slows every write.

## Getting the Plan

| Engine | Plan | Actual metrics |
|---|---|---|
| DuckDB (executed) | `EXPLAIN` | `EXPLAIN ANALYZE` (rows + time per operator; operator time is summed over threads) |
| SQLite (executed) | `EXPLAIN QUERY PLAN` (`SCAN` = full, `SEARCH ... USING INDEX` = seek) | time it in the client |
| Postgres (not executed here) | `EXPLAIN` | `EXPLAIN (ANALYZE, BUFFERS)` — runs the query; wrap DML in `BEGIN; ... ROLLBACK;` |
| BigQuery (not executed here) | dry run: `bq query --dry_run` (bytes that will be billed) | job Execution details; `INFORMATION_SCHEMA.JOBS`: `total_bytes_processed`, `total_slot_ms` |
| Snowflake (not executed here) | `EXPLAIN` | Query Profile: *Partitions scanned / total*, *Bytes spilled*; `SYSTEM$CLUSTERING_INFORMATION` |

## Reading the Plan

| Plan symptom | Likely cause | Fix to test |
|---|---|---|
| Full scan (`SCAN`, `Seq Scan`, all partitions) + filter removing most rows | Predicate not sargable: function or cast on the column (`to_char(ts)`, `DATE(ts)`, `CAST(id AS STRING)`) | Half-open range on the raw column; cast the other (small) side |
| Scan lists every column / huge bytes | `SELECT *` on a columnar store | Select only the columns used downstream |
| Window/sort/join over the full table, filter above it | Filter applied after the expensive operator | Move the filter below it (check the semantics stay equal) |
| All partitions / row groups read despite a date filter | Filter not on the partition column, or data not clustered by it | Filter the partition column directly; cluster/sort by the filter key |
| Estimated rows ≠ actual rows by 10×+ | Stale statistics | `ANALYZE table` then re-plan, before any rewrite |
| Sort/hash "spilled to disk" | Too many rows or columns reach it | Filter and project earlier; then memory settings |
| Nested loop with a large outer side | Missing index on the join key, or bad estimate | Check indexes and statistics |

## Measured on Synthetic Data (DuckDB 1.5, SQLite 3.53)

| Change | Plan evidence | Before → after |
|---|---|---|
| Filter after `ROW_NUMBER()` + `SELECT *` → filter first + 4 columns (DuckDB, 20M rows parquet) | WINDOW input 20,000,000 → 1,741,803 rows; scan shows `Filters: event_ts>=...` | 14.7 s → 1.1 s, 0 divergent rows |
| Parquet unsorted → sorted by `event_ts` (clustering analog), same date filter | min/max statistics skip row groups | 0.130 s → 0.016 s |
| `strftime('%Y-%m', created_at) = '2026-08'` → half-open range (SQLite, 2M rows) | `SCAN orders` → `SEARCH ... USING INDEX (created_at>? AND created_at<?)` | 2.68 s → 1.95 s (keeps ~1/12 of rows: low selectivity, small gain) |
| Point lookup, no index → `CREATE INDEX` on `customer_id` | `SCAN` → `SEARCH ... (customer_id=?)` | 0.410 s → 0.003 s |

## Implementation

```python
import time
from pathlib import Path

import duckdb


def compare_queries(con: duckdb.DuckDBPyConnection, original: str, candidate: str, reps: int = 3) -> dict[str, float]:
    """Compara tempo e resultado de duas versões de uma consulta.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Conexão com os dados de teste.
    original, candidate : str
        Consulta original e candidata (mesmas colunas, mesma ordem).
    reps : int, optional
        Repetições; usa o menor tempo, by default 3.

    Returns
    -------
    dict[str, float]
        Tempos mínimos (s) e número de linhas divergentes.

    Raises
    ------
    AssertionError
        Se a candidata não retorna exatamente as mesmas linhas.
    """
    def best(sql: str) -> float:
        tempos = []
        for _ in range(reps):
            inicio = time.perf_counter()
            con.execute(sql).fetchall()
            tempos.append(time.perf_counter() - inicio)
        return min(tempos)

    divergentes = con.execute(f"""
        SELECT (SELECT COUNT(*) FROM (({original}) EXCEPT ALL ({candidate})))
             + (SELECT COUNT(*) FROM (({candidate}) EXCEPT ALL ({original})))
    """).fetchone()[0]
    assert divergentes == 0, f"Resultado mudou: {divergentes} linhas divergentes"
    return {"original_s": best(original), "candidate_s": best(candidate), "divergent_rows": divergentes}


# Plano da original ANTES de qualquer reescrita:
# print(con.execute("EXPLAIN ANALYZE " + original).fetchall()[0][1])
```

For the SQL itself (grain, joins, windows), REQUIRED SUB-SKILL: writing-analytical-sql.

## Red Flags — Stop and Get the Plan

| Thought | Reality |
|---|---|
| "No time, just rewrite it" | `EXPLAIN` / dry run takes seconds and is free; a blind rewrite that misses the bottleneck costs another full run |
| "The problem is obvious from the SQL" | Then the plan confirms it in seconds; often the real cost is elsewhere (stale stats, missing partition filter) |
| "I'll bundle all the fixes" | You cannot tell which change helped or which one changed the result |
| "Add an index, it can't hurt" | Low-selectivity index barely helps; every index costs writes and storage. Check existing indexes first |
| "Same results, by reasoning" | Run the `EXCEPT ALL` check; casts, `LOWER`, and filter moves around windows change rows |
| "EXPLAIN at the end to confirm" | The plan of the original is the diagnosis, not the receipt |
