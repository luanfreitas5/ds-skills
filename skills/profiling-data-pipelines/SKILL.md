---
name: profiling-data-pipelines
description: Use when a data pipeline or feature job is slow, got slower, runs out of memory (OOM-killed, MemoryError, swap), or when someone proposes switching library (pandas to polars), adding multiprocessing, numba, or swifter to make it faster — before changing any code for performance.
---

# Profiling Data Pipelines

## Overview

Slow pipelines are usually slow for one boring reason: reading columns you do not use, bad dtypes, or a join that multiplies rows. Swapping the library or adding processes speeds up the symptom and keeps the cause. **Core principle:** no performance change without a per-stage table of time, peak memory, and row count *before* and *after*, one change at a time.

## Rules

1. **Baseline first, in the same run you ship.** Instrument the *current* code and record the table before touching anything. "Merge the fix, measure later" leaves no before number, so any speedup claim is a guess.
2. **Per stage, three numbers:** wall time, **peak** RSS (sampled during the stage, not read at the end), rows in → rows out. The row count finds exploding joins that time alone hides. Use the same harness on the full run too. Watching Task Manager or `top` gives no per-stage peak and no record.
3. **Every lookup merge declares its cardinality:** `validate="many_to_one"` (pandas) or `validate="m:1"` (polars `join`). A fan-out then raises an error instead of silently multiplying rows and sums.
4. **Profile a representative sample** (1–10 % of rows, same schema). Profilers slow Python-level code down 2–10x, so confirm the hotspot with plain timers.
5. **Walk the checklist in order.** Fix the cause at the top before anything below it: a row explosion makes every later `apply` look guilty.
6. **One change → re-measure → assert output equal** (`pd.testing.assert_frame_equal` on the sample). If the output changed, it was a bug, not an optimization. Document it.
7. **Library swap or parallelism only after the checklist**, and only if the table still shows CPU-bound Python in a stage. Parallel workers multiply peak memory.

## Checklist (in order)

| # | Check | Evidence | Fix |
|---|---|---|---|
| 1 | **Join fan-out** | rows out > rows in on a lookup join | Correct key grain (`on=["store_id","day"]`), dedupe/aggregate the right side, `validate="many_to_one"` |
| 2 | **I/O** | read stage dominates; wide CSV | `usecols`/`columns=`, `engine="pyarrow"`, Parquet instead of CSV, `scan_*` in polars |
| 3 | **dtypes** | `object` columns, `int64` ids | `category` for low-cardinality strings, downcast ids/flags, explicit `dtype=` on read |
| 4 | **Unneeded copies** | memory steps up with no new rows | drop dead columns early, avoid `df.copy()`/chained reassignments, no mid-pipeline `collect()` |
| 5 | **Python-level loops** | stage time scales with rows, CPU at 100 % of one core | vectorize (`.str`, `.dt`, `cat.rename_categories`); in polars see REQUIRED SUB-SKILL: writing-idiomatic-polars |

## Tools

| Tool | Platform | Use |
|---|---|---|
| `scalene run -o prof.json job.py` then `scalene view --cli prof.json` | Linux, macOS, **Windows** | Line-level CPU (Python vs native) + memory peak per line |
| `memray run -o mem.bin job.py` then `memray flamegraph mem.bin` | **Linux/macOS only** (no Windows build) | Who allocated the peak, including native (pandas/arrow) allocations |
| `measure_stage` (below, `psutil`) | all | Per-stage table; the Windows fallback for memray |
| `pyinstrument job.py` / `tracemalloc` | all | Call-tree timing / Python-traced allocations |

Run scalene on a script **without** the sampler thread below. Scalene attributes the sampler's time to itself.

## Implementation

```python
import logging
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager

import psutil

logger = logging.getLogger(__name__)
_PROC = psutil.Process()


@contextmanager
def measure_stage(name: str, results: list[dict], interval: float = 0.05) -> Iterator[dict]:
    """Mede tempo de parede e pico de RSS de uma etapa do pipeline.

    Uma thread amostra o RSS a cada ``interval`` segundos, capturando picos
    transitórios (ex.: durante um merge) que a leitura no fim da etapa perderia.

    Parameters
    ----------
    name : str
        Nome da etapa.
    results : list[dict]
        Lista que recebe um registro por etapa.
    interval : float, optional
        Intervalo de amostragem do RSS em segundos, by default 0.05.

    Yields
    ------
    dict
        Registro da etapa; o chamador anota campos extras (ex.: linhas).

    Examples
    --------
    >>> res: list[dict] = []
    >>> with measure_stage("join_promos", res) as rec:
    ...     n_in = len(tx)
    ...     tx = tx.merge(promos, on=["store_id", "day"], validate="many_to_one")
    ...     rec["linhas"] = f"{n_in:,} -> {len(tx):,}"
    """
    record: dict = {"etapa": name}
    peak = _PROC.memory_info().rss
    stop = threading.Event()

    def _sample() -> None:
        nonlocal peak
        while not stop.wait(interval):
            peak = max(peak, _PROC.memory_info().rss)

    sampler = threading.Thread(target=_sample, daemon=True)
    sampler.start()
    start = time.perf_counter()
    try:
        yield record
    finally:
        elapsed = time.perf_counter() - start
        stop.set()
        sampler.join()
        peak = max(peak, _PROC.memory_info().rss)
        record.update(segundos=round(elapsed, 2), pico_rss_mb=peak // 2**20)
        results.append(record)
        logger.info("Etapa %s: %.2f s, pico RSS %d MB", name, elapsed, peak // 2**20)
```

## Measured Example

1M-row, 35-column CSV. Promo table has one row per store per day, joined on `store_id` only (Windows, pandas 3.0, polars 1.44):

| Variant | Total | Peak RSS | Rows after join | Slowest stage |
|---|---|---|---|---|
| Original (pandas) | 68 s | 9.1 GB | 30,000,000 | join 26 s, `apply` 24 s |
| "Just use polars" (same join) | 7.4 s | **9.7 GB** | 30,000,000 | still OOM on a 16 GB worker at scale |
| Checklist fix (pandas: key `["store_id","day"]`, `usecols`, dtypes, `category`) | **3.1 s** | **0.59 GB** | 1,000,000 | read 1.7 s |

The `apply` was a symptom: it was slow because it ran on 30x the rows. The fan-out also inflated `total` 30x, so the "optimization" was a correctness fix too.

## Rationalizations

| Excuse | Reality |
|---|---|
| "It's obviously the `.apply`" | Its cost scales with rows. Check row counts first. |
| "Rewrite in polars / add multiprocessing" | Faster on a bad plan, same or higher peak memory. Still OOM. |
| "No time to measure, demo tomorrow" | The harness takes 2 minutes on a sample. A wrong guess costs the night run. |
| "Ship the fix, measure after" | Without a before number there is no speedup, only a claim. |

## Red Flags: stop and measure

- Proposing a library, numba, swifter, or `multiprocessing` before seeing a per-stage table
- "I expect this to be much faster" with no numbers
- Several changes in one diff, so gains can't be attributed
- Recommending memray to a Windows user
