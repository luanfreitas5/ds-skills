---
name: orchestrating-pipelines
description: Use when a cron job runs a script that downloads or builds data on a schedule, when rerunning a job duplicates rows or overwrites data/raw, when a failed run goes unnoticed, when missed days cannot be rebuilt (backfill), or when choosing between cron, Prefect, Dagster and Airflow for step dependencies and retries.
---

# Orchestrating Pipelines

## Overview

A scheduled pipeline fails in four ways: **reruns change the result** (appends, overwritten raw files), **failures are silent**, **outages cannot be replayed**, and **step B starts when step A failed** (two crons 30 minutes apart). **Core principle:** every task is a pure function of its *logical date* that writes one partition, so rerun, retry and backfill are the same operation. A deadline, a sunk cost, or "no new dependencies" changes the *tool*, never these four properties.

Docs verified 2026-10-28: Airflow 3.3.2 (`from airflow.sdk import dag, task`; backfill flags `--from-date/--to-date/--reprocess-behavior` in 3.2+, `--start-date/--end-date` in 3.1), Dagster daily partitions + `RetryPolicy`, Prefect 3 `@flow`/`@task(retries=)`. Snippets run on synthetic data with `prefect` 3.9.0, `dagster` 1.13.27, `pandas` 3.1.0, `pyarrow` 26.0.0 (pytest idempotency + missing-day tests, a forced task failure retried by Prefect, a Dagster partition materialized); the Airflow snippet was not executed locally (Airflow does not install on Windows).

## Rules

| Rule | How |
|---|---|
| Key every run by logical date | `--date YYYY-MM-DD` / flow parameter / partition key. Never `date.today()` inside a task body |
| Bronze is append-only snapshots | **REQUIRED SUB-SKILL:** `ingesting-external-data` (dated snapshot + manifest; never overwrite `data/raw` or `data/bronze`) |
| Downstream writes replace one partition | `dt=YYYY-MM-DD/part.parquet` via `.tmp` + `os.replace`. Never `append` to a CSV. Dedupe by a **verified** unique key, never a guessed one (it silently drops real rows) |
| Retry transient errors only | Bounded count, exponential backoff, catch specific exceptions (timeouts, 5xx). A 4xx or a schema error fails immediately |
| Fail loudly | Non-zero exit, a log line in pt-BR with the logical date, and an alert **plus** a dead-man's-switch (healthchecks.io style) for "the job never ran". **REQUIRED SUB-SKILL:** `setting-up-project-logging` |
| Validate at every step boundary | **REQUIRED SUB-SKILL:** `validating-data-contracts` (the contract failing is what stops step B). Not "this week" |
| Catch-up is automatic | The run computes missing partitions in a lookback window; backfill = same code over a date range |
| One idempotency test | Run the task twice for the same date, assert identical output; run with one missing day, assert it is filled |

## Choosing the Tool

| Situation | Use |
|---|---|
| 1 script, 1-2 steps, one machine, or "no new dependencies" | Keep cron, but **one** chained entry (`a && b`) under `flock`, with the rules above in the script. Write down the trigger to migrate |
| Dependencies, retries, UI and backfill wanted; mostly Python tasks | **Prefect** (lowest ceremony: decorators, `serve(cron=)`) |
| Data assets with lineage, partitions, freshness checks | **Dagster** (partitioned assets; backfill from UI) |
| Many teams, many integrations, existing platform | **Airflow** |

Migrate off cron when: a third dependent step appears, a second person needs run history, or one backfill is done by hand twice. If the owner forbids an orchestrator, comply, apply the cron row, and state the migrate triggers once in writing.

## Tested Patterns

Imports: `os`, `logging`, `datetime.date/timedelta`, `pathlib.Path`, `pandas as pd`, `prefect.flow/task`, `dagster as dg`.

```python
def write_partition(df: pd.DataFrame, root: Path, day: date) -> Path:
    """Grava a partição de um dia de forma atômica e idempotente (substitui só `dt=day`)."""
    target = root / f"dt={day.isoformat()}" / "part.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, target)  # nunca deixa arquivo parcial
    return target


def list_missing_days(root: Path, start: date, end: date) -> list[date]:
    """Lista os dias entre `start` e `end` (inclusive) sem partição gravada."""
    days = (start + timedelta(days=i) for i in range((end - start).days + 1))
    return [d for d in days if not (root / f"dt={d.isoformat()}" / "part.parquet").exists()]
```

Prefect 3 (`prefect` 3.x): retries and catch-up live in the flow; the backfill is `daily_sales(logical_date=..., lookback=30)`.

```python
@task(retries=3, retry_delay_seconds=[30, 120, 600])
def pull_day(day: date) -> pd.DataFrame:
    """Baixa um dia da API; levanta exceção em erro para disparar o retry."""
    ...


@flow
def daily_sales(logical_date: date | None = None, lookback: int = 7) -> None:
    """Preenche os dias faltantes dos últimos `lookback` dias."""
    end = logical_date or date.today() - timedelta(days=1)
    for day in list_missing_days(RAW, end - timedelta(days=lookback - 1), end):
        write_partition(pull_day(day), RAW, day)

# daily_sales.serve(name="vendas-diarias", cron="0 6 * * *")
```

Dagster: `partitions_def=DailyPartitionsDefinition(start_date=...)` on the asset, `context.partition_key` as the logical date, `retry_policy=dg.RetryPolicy(max_retries=3, delay=30, backoff=dg.Backoff.EXPONENTIAL)`; backfill a range from the UI (one run per partition by default).

```python
@dg.asset(partitions_def=daily, retry_policy=dg.RetryPolicy(max_retries=3, delay=30, backoff=dg.Backoff.EXPONENTIAL))
def sales_bronze(context: dg.AssetExecutionContext) -> pd.DataFrame:
    day = context.partition_key  # 'YYYY-MM-DD'
    ...
```

Airflow 3 (not executed locally): `@dag(schedule="@daily", start_date=..., catchup=False, default_args={"retries": 3})`, tasks read the logical date from the run context (`logical_date`), never `datetime.now()`. Backfill: `airflow backfill create --dag-id D --from-date 2026-01-01 --to-date 2026-01-07 --reprocess-behavior failed`.

## Rationalizations

| Excuse | Reality |
|---|---|
| "Demo in 3 hours; validation and tests next week" | The idempotency test is 10 lines and the contract is the only thing that stops a bad day reaching the dashboard. Next week is when the duplicates return |
| "I already built the cron scripts, don't rewrite" | Keep the fetch/feature code; wrap it with partitions and a date argument. Sunk cost decides the effort, not the properties |
| "Tech lead said no orchestrator, so no retries/backfill machinery" | The ban is on a tool, not on idempotency, alerts or catch-up. Do them in the script and write the migrate triggers |
| "I agree at this size / good enough for now" | Agreeing is how the first outage went unnoticed. List what is skipped, with an owner and date |
| "A bare `curl` to Slack is the alert" | It cannot fire when the VM or cron is down. Add a dead-man's-switch ping |
| "Dedupe on the likely key" | An unverified key drops real sales silently. Check uniqueness on the raw data first |
| "Overwrite today's `sales.csv`, it is atomic" | Atomic replace prevents partial files, not loss of last week's snapshot |
| "Logging/alert polish and the test suite can wait; only idempotency now" | Fine to defer *polish*, not the four properties: a loud failure signal, the contract and the two tests ship today. Everything deferred gets an owner and a date in the ticket |
| "pandera is a new dependency; the tech lead said none" | A 15-line check (columns, dtypes, key unique, rows > 0) is no dependency and still raises |
| "Retry on any Exception" | Retrying a schema or 4xx error hides it for minutes and floods the API |

## Red Flags: STOP

- `append`/`mode="a"` in a scheduled job
- `date.today()` / `datetime.now()` inside a task instead of a parameter
- Two cron lines where B assumes A succeeded
- `except Exception: pass`, or a pipeline with no alert and no "last success" check
- A backfill that is a different code path from the daily run
- "Skipped validation/tests, will add later" with no date
