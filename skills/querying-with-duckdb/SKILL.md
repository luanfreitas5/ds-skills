---
name: querying-with-duckdb
description: Use when analysing large local Parquet or CSV files (many files, globs, hive-partitioned folders like year=/month= or dt=), when pd.read_parquet or pd.read_csv plus concat hits MemoryError or takes minutes to keep a small slice, when choosing between DuckDB and polars lazy, or when handing query results to polars or pyarrow.
---

# Querying with DuckDB

## Overview

`pd.read_parquet(dir)` then filter materializes every row and column before discarding 99% of them. DuckDB runs SQL **directly on the files**: partition folders not matching the `WHERE` are never opened, Parquet row groups are skipped by min/max statistics, and only the selected columns are decoded. **Core principle:** the filter and the column list must reach the scan, and you prove it with `EXPLAIN` instead of assuming.

Measured (DuckDB 1.5, 20M rows, 23 hive partitions, 2.3 GB Parquet; result 22k rows):

| Approach | Time | Peak RAM |
|---|---|---|
| `pd.read_parquet(dir)` then boolean filter | 21.3 s | 5.81 GB |
| `pd.read_parquet(dir, columns=, filters=)` | 5.8 s | 0.15 GB |
| polars `scan_parquet(...).filter().select().collect()` | 0.59 s | 0.07 GB |
| DuckDB `read_parquet(...)` + `WHERE` | 0.53 s | 0.06 GB |
| Same month as CSV (145 MB), DuckDB `read_csv` + `WHERE` | 0.73 s | — |
| Same month as Parquet | 0.01 s | — |

## DuckDB or polars lazy?

Both push filters and projections into the scan. Take the **first** row that matches.

| Situation | Choose |
|---|---|
| Intermediate may exceed RAM (big join, sort, `COUNT(DISTINCT)`) | DuckDB (spills to `temp_directory`) |
| CSV folders by glob, schema drift (`union_by_name`), converting to partitioned Parquet | DuckDB |
| Team thinks in SQL; ad-hoc exploration; window functions / `QUALIFY` | DuckDB |
| Code continues as feature engineering in `src/` and every step fits in RAM | polars lazy (REQUIRED SUB-SKILL: writing-idiomatic-polars) |
| Heavy scan/join, then Python features | DuckDB → `.pl()` → polars expressions (shared Arrow memory) |

## Rules

1. **Glob + hive, never a Python loop:** `read_parquet('events/**/*.parquet', hive_partitioning = true)`; `read_csv('logs/logs_2024-0[1-3]-*.csv', union_by_name = true)`. A file list built in Python goes in as a parameter: `read_csv($files)`.
2. **CSV has no statistics:** `WHERE` prunes nothing inside a CSV, and only the glob or file list limits what gets read. Every matched file is parsed in full. When the same CSVs will be queried more than once, convert them to partitioned Parquet with `COPY (SELECT ...) TO 'dir' (FORMAT parquet, PARTITION_BY (year, month))`. On a large CSV, pin the key and number types (`types = {'user_id': 'BIGINT'}`) so a join isn't defeated by an inferred VARCHAR.
3. **Filter on partition columns** (`year = 2024 AND month = 3` gives `Scanning Files: 1/23`). For non-partition columns, compare the raw column against a range (`event_ts >= TIMESTAMP '2024-03-01' AND event_ts < TIMESTAMP '2024-04-01'`) so row-group statistics can skip data. Wrapping the column in a function, such as `strftime(event_ts, '%Y-%m') = '2024-03'`, reads all 23 files and took 7.56 s instead of 0.59 s.
4. **Hive types:** values are auto-typed. `month=3` becomes BIGINT and `dt=2024-01-15` becomes DATE, so `dt BETWEEN DATE ... AND DATE ...` prunes. Zero-padded `month=03` becomes VARCHAR `'03'`: `month = 3` still matches because DuckDB casts the comparison, but the string literal `month = '3'` returns nothing. Run `DESCRIBE SELECT * FROM read_parquet(...)` first and pin types with `hive_types = {'month': INTEGER}`. If a filter returns zero rows, `SELECT DISTINCT col` on one partition shows the real literal values.
5. **Project explicitly:** name the columns. `SELECT *` decodes all of them.
6. **Verify with plain `EXPLAIN` before the full run.** It does not execute the query and it takes milliseconds. In the `READ_PARQUET` node, check `Projections`, `Filters`, `File Filters`, and `Scanning Files: k/N`. If `k = N` despite a partition filter, pruning failed. `EXPLAIN ANALYZE` executes the query, so run it only on a small slice or instead of the real run.
7. **Resource limits at connect time:** `duckdb.connect(config={"memory_limit": "8GB", "temp_directory": "..."})`. `SET` does not accept `$params`. Close the connection when done. For large `COUNT(DISTINCT)` where an estimate is acceptable, `approx_count_distinct` needs far less memory.
8. **Bind values with parameters** (`$name`, lists included), never f-strings. Globs and file lists work as parameters inside `read_parquet` and `read_csv`; this was verified on DuckDB 1.5. Pass exactly the parameters the statement uses, because an unused key raises `Parameter argument/count mismatch`.
9. **Several outputs from one filter:** `CREATE TEMP TABLE slice AS SELECT ...` once, then aggregate, `.pl()`, `.to_arrow_table()` and `COPY` from `slice`. Calling a result method on a relation re-executes the whole scan each time.
10. **Handoff (DuckDB ≥ 1.4):**
    - `.pl()` gives a polars DataFrame and `.pl(lazy=True)` a LazyFrame.
    - `.to_arrow_table()` gives a `pa.Table`. `.arrow()` now returns a streaming `RecordBatchReader`, and `fetch_arrow_table()` is deprecated.
    - `.df()` copies into pandas, so use it only for small results.
    - For a colleague, send a file: `COPY slice TO 'out.parquet'`.
11. **Zero-copy the other way:** DuckDB can query a polars or Arrow frame by its variable name (`duckdb.sql("SELECT ... FROM df_pl")`). Scanning a 1.6 GB polars frame this way added 6 MB of RSS. `pl.from_arrow(tbl)` shares buffers with the Arrow table.
12. **Measure and state it:** report the time, the peak RAM (`psutil.Process().memory_info().rss`, which works on Windows, unlike `resource`) or the `EXPLAIN` file counts. "Should fit" is not a measurement.

## Implementation

```python
import logging
from pathlib import Path

import duckdb
import polars as pl

logger = logging.getLogger(__name__)


def load_purchases_by_plan(
    events_dir: Path, users_csv: Path, country: str, year: int, month: int, spill_dir: Path
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Filtra compras de um país/mês e agrega por plano sem materializar o dataset inteiro.

    Parameters
    ----------
    events_dir : Path
        Diretório Parquet particionado em hive (``year=/month=``).
    users_csv : Path
        CSV com ``user_id`` e ``plan`` (uma linha por usuário).
    country, year, month : str, int, int
        Filtros aplicados na leitura (poda de partições + estatísticas).
    spill_dir : Path
        Diretório temporário para o DuckDB descarregar dados se faltar RAM.

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame]
        Linhas filtradas (com ``plan``) e o total por plano.

    Raises
    ------
    ValueError
        Se o scan não podar partições ou se o join duplicar linhas.
    """
    params = {"glob": (events_dir / "**" / "*.parquet").as_posix(),
              "year": year, "month": month, "country": country}
    scan_sql = """
        SELECT user_id, event_ts, amount                    -- grão: 1 linha por evento
        FROM read_parquet($glob, hive_partitioning = true,
                          hive_types = {'year': INTEGER, 'month': INTEGER})
        WHERE year = $year AND month = $month
          AND country = $country AND event_type = 'purchase'
    """
    with duckdb.connect(config={"memory_limit": "8GB", "temp_directory": spill_dir.as_posix()}) as con:
        plan = con.execute("EXPLAIN " + scan_sql, params).fetchall()[0][1]  # não executa
        if "Scanning Files" not in plan:
            raise ValueError(f"Poda de partições não ocorreu; revise o filtro:\n{plan}")
        logger.info("Plano de leitura:\n%s", plan)

        con.execute("CREATE TEMP TABLE events_slice AS " + scan_sql, params)  # lê os arquivos 1 vez
        con.execute("""CREATE TEMP TABLE purchases AS
                       SELECT e.*, u.plan FROM events_slice e
                       LEFT JOIN read_csv($users, types = {'user_id': 'BIGINT'}) u USING (user_id)""",
                    {"users": users_csv.as_posix()})
        n_before, n_after = con.sql("SELECT (SELECT COUNT(*) FROM events_slice), "
                                    "(SELECT COUNT(*) FROM purchases)").fetchone()
        if n_before != n_after:
            raise ValueError(f"Join multiplicou linhas ({n_before} -> {n_after}): user_id duplicado")

        rows = con.table("purchases").pl()
        totals = con.sql("""SELECT plan, SUM(amount) AS total_amount, COUNT(*) AS n_events
                            FROM purchases GROUP BY plan ORDER BY total_amount DESC""").pl()
    logger.info("Recorte: %d eventos; %d planos", rows.height, totals.height)
    return rows, totals
```

For the grain and join rules of the SQL itself, REQUIRED SUB-SKILL: writing-analytical-sql. Tuning queries on a remote warehouse is out of scope.

## Common Mistakes

| Mistake | Fix |
|---|---|
| `pd.read_parquet(dir)` or `pd.concat([pd.read_csv(f) ...])`, then filter | DuckDB `read_parquet`/`read_csv` with a glob and a `WHERE` |
| Expecting `WHERE` to skip data inside CSVs | Narrow the glob; convert to Parquet once with `COPY ... PARTITION_BY` |
| `strftime(ts, ...)` or another function wrapping the filtered column | Filter on partition columns or on raw-column ranges |
| Zero rows from `month = 3` | `DESCRIBE`, then `hive_types`; `SELECT DISTINCT` on one partition |
| `EXPLAIN ANALYZE` on the full query, then running it again | Plain `EXPLAIN` (cheap); run the query once |
| `SET memory_limit = $x` | `duckdb.connect(config={...})` |
| `.pl()` then `.to_arrow_table()` on the same relation | `CREATE TEMP TABLE` once, read it many times |
| `LEFT JOIN` to a dimension without a row-count check | Compare counts before and after the join |
| `.df()` then `pl.from_pandas` | `.pl()` or `.to_arrow_table()` |
| Old snippets that treat `.arrow()` as a Table | `.to_arrow_table()` on DuckDB ≥ 1.4 |
