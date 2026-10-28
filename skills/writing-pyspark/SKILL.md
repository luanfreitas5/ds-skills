---
name: writing-pyspark
description: Use when a PySpark job is slow or skewed (one straggler task, long shuffle stage, OOM), when data is bigger than one machine, when code uses a row-by-row Python UDF, collect() or toPandas() on a large table, or when someone asks to "add more executors" or tune configs without measuring.
---

# Writing PySpark

## Overview

Spark jobs are slow for structural reasons: data pulled to the driver, rows crossing JVM-Python, shuffles with skewed keys. Executors fix none of these. **Core principle:** read the plan and the Spark UI first, change one thing, re-measure. A deadline, a sunk cost, or a VP's "just add executors" changes the schedule, never the measurement.

Verified 2026-10-28: the four functions below ran on PySpark 4.2.1 (PyPI) / JDK 21, local mode, synthetic data (skew share found 0.30; salted join sums equal the plain join; `explain("formatted")` shows `AdaptiveSparkPlan`). NOT run here: Python UDF / `pandas_udf` equivalence (Python workers failed to start on Windows + Python 3.14), `write.parquet` (needs `winutils` on Windows), and anything cluster-only (Spark UI numbers, `spark-submit` flags). Confirm those on your cluster. Spark 3.2+ has AQE on by default.

**REQUIRED SUB-SKILL:** `profiling-data-pipelines` (before/after table, one change at a time). **REQUIRED SUB-SKILL:** `optimizing-sql-queries` (read plans before rewriting). For data that fits one machine, **REQUIRED SUB-SKILL:** `writing-idiomatic-polars`; Spark is not the default.

## Rules

| Rule | How |
|---|---|
| Measure before any change | `df.explain("formatted")`, then the Spark UI: per-stage time, shuffle read/write, **task time min/median/max**. Record it. No config or executor change without this table |
| Never `collect()` / `toPandas()` a large table | Aggregate, filter or `limit` in Spark first; write results with `df.write.parquet`. `toPandas()` only on a result you know is small (check `.count()` first) |
| No row-by-row Python UDF | Order: built-in `F.*` / `F.when` -> `pandas_udf` (vectorised) -> plain `udf` only if unavoidable. Replace and assert equal output on a sample |
| Check key skew before join/groupBy | `profile_key_skew` below. One key above ~10% of rows, or max task >> median task, means skew |
| Skew fix order | Filter junk keys (`-1`, null) only after checking the dimension has none; AQE skew join; broadcast the small side; salting hot keys last |
| Executors are the last lever | More executors cannot speed a driver-bound step or a straggler task. Add them only when every task is balanced and CPU-bound |
| Do not guess configs | `shuffle.partitions=1000` copied from a blog is a guess. Take the value from stage metrics (target ~100-200 MB per shuffle partition) or leave AQE coalescing to do it |
| Preserve semantics | Native rewrites must keep null behaviour; compare old vs new on a sample (sums, counts, null count) before the full run |

## Implementation (tested, local mode, synthetic)

```python
import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F


def build_session(app_name: str = "job") -> SparkSession:
    """Cria a sessão com AQE explícito (padrão desde o Spark 3.2; fixar evita surpresas)."""
    return (
        SparkSession.builder.appName(app_name)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.skewJoin.enabled", "true")
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
        .getOrCreate()
    )


def profile_key_skew(df: DataFrame, key: str, top: int = 5) -> pd.DataFrame:
    """Retorna as chaves mais frequentes e a fração de linhas de cada uma."""
    total = df.count()
    out = (
        df.groupBy(key).count().orderBy(F.desc("count")).limit(top)
        .withColumn("share", F.col("count") / F.lit(total))
    )
    return out.toPandas()  # no máximo `top` linhas


def band_amount(col: str = "amount"):
    """Faixa de valor com expressões nativas (substitui o UDF; preserva nulo)."""
    c = F.col(col)
    return F.when(c.isNull(), F.lit(None)).when(c > 1000, "high").when(c > 100, "mid").otherwise("low")


def join_skewed(facts: DataFrame, dims: DataFrame, key: str, hot_keys: list, salt_n: int = 8) -> DataFrame:
    """Join com sal só nas chaves quentes; as demais seguem o join normal."""
    hot = F.col(key).isin(hot_keys)
    cold = facts.filter(~hot).join(dims.filter(~F.col(key).isin(hot_keys)), key)
    salted_f = facts.filter(hot).withColumn("salt", (F.rand(42) * salt_n).cast("int"))
    salted_d = dims.filter(F.col(key).isin(hot_keys)).withColumn(
        "salt", F.explode(F.array([F.lit(i) for i in range(salt_n)]))
    )
    return cold.unionByName(salted_f.join(salted_d, [key, "salt"]).drop("salt"))
```

Pattern for the job: `tx.select(only needed cols)` -> `withColumn("band", band_amount())` -> join -> `groupBy().agg()` -> `write.parquet` (or `toPandas()` on the few aggregate rows). If the dimension is small enough for executor memory, `F.broadcast(dim)` removes the shuffle entirely; if `pandas_udf` is unavoidable, annotate `pd.Series -> pd.Series` and keep it vectorised.

## Rationalizations

| Excuse | Reality |
|---|---|
| "The VP said just add executors" | Authority sets priorities, not the bottleneck. Show the one stage/driver step executors cannot touch (one line), then fix it |
| "No time to look at the UI, deadline is tomorrow" | `explain` + UI is 5 minutes and prevents a failed overnight run |
| "We spent two weeks on this UDF/collect code" | Sunk cost. The `when` rewrite is one expression with the same output; assert it on a sample |
| "Set shuffle.partitions=1000 / more memory to be safe" | A config without a stage metric behind it is a guess. Cite the number or leave AQE on |
| "Drop customer_id = -1 before the join, obviously junk" | Check the dimension has no such row and ask the owner; otherwise results change silently |
| "Salting tonight is too risky, do it later" | Then say what the measured straggler is and ship AQE skew + broadcast first; do not leave it unmeasured |
| "toPandas is fine, it's only the final result" | Only if you checked its size. Aggregate first |

## Red Flags: STOP

- `collect()`, `toPandas()`, `.rdd`, or `for row in df` on an unaggregated table
- `@udf` doing arithmetic or `if/else` that `F.when` expresses
- A tuning change with no before/after stage table
- Filtering a hot key without checking join semantics
- "Add executors" with one task far longer than the median
