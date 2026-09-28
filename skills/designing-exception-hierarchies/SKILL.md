---
name: designing-exception-hierarchies
description: Use when adding error handling to a data or ML pipeline (batch scoring, training, ETL), when code has `except Exception`, catches builtins like ValueError/KeyError/OSError to "skip bad input", uses `errors="coerce"` to avoid a crash, or when asked to make a job "robust", "never crash again", or "not take down the whole batch".
---

# Designing Exception Hierarchies

## Overview

In a pipeline, a crash is visible. Partial data is not: a run that "logs and keeps going" still writes output, and the only symptom is a metric drift weeks later. **Core principle:** every failure is either **fatal** (fail fast, before work is done) or **tolerated within an explicit budget** (counted, recorded in the output, and turned into a fatal error when the budget is exceeded). Nothing is ever just logged and forgotten.

## The Hierarchy

One small hierarchy in `src/exceptions/`. Domain code raises only these, and catches only these:

```
PipelineError                   # base: any expected pipeline failure
├── ConfigError                 # always fatal, raised at startup
├── DataError                   # input breaks the contract (corrupt, schema, domain)
│   └── DegradationBudgetExceeded
└── ModelError                  # artifact missing/incompatible, predict failure
```

- **Translate at the boundary.** Wrap the specific low-level exception where it happens (`FileNotFoundError`, `pl.exceptions.ComputeError`, `pandera` `SchemaError`), **log with context in pt-BR** (stage, file, counts, no PII), and re-raise the domain error `from e` so the traceback chain survives.
- **Catch only domain errors** in a tolerance loop. `except (ValueError, KeyError, OSError)` also catches bugs in *your* code, and a bug in `clean_data` then shows up as "region skipped" instead of a crash.
- Log once, where the context is added. Upper layers catch `PipelineError`, log a summary, and set the exit code. They do not log the same traceback again.

## Fail Fast vs Degrade

| Situation | Decision |
|---|---|
| Config invalid/missing | `ConfigError` at startup, before any I/O |
| Model missing/incompatible | `ModelError` **before** processing data (bounded retry allowed for deploy races) |
| Training data invalid | Fatal. Never train on silently reduced data |
| One of N independent units (region, file, partition) unreadable | Tolerate as `DataError`, **within budget** |
| Row-level bad values (unparseable date, out of range) | Tolerate **within an invalid-rate budget**; count per rule |
| Unexpected exception type (`TypeError`, `AttributeError`...) | Never caught: it is a bug, so crash |

Degrading is allowed only when all three hold: **budget** (max fraction, from config) + **visibility** (count logged, skipped units written to a run report/output metadata, non-zero "partial" exit code) + **escalation** (`DegradationBudgetExceeded` when over budget). A "minimal change" still includes the budget check. It is 3 lines, and without it the "robust" code is how silent failures happen.

## Implementation

```python
import logging
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)


class PipelineError(Exception):
    """Base de todos os erros esperados do pipeline."""


class ConfigError(PipelineError):
    """Configuração ausente ou inválida — sempre fatal, antes de qualquer trabalho."""


class DataError(PipelineError):
    """Dado de entrada viola o contrato (arquivo corrompido, schema, domínio)."""


class ModelError(PipelineError):
    """Artefato de modelo ausente, incompatível ou falha na predição."""


class DegradationBudgetExceeded(DataError):
    """Falhas toleradas passaram do limite acordado."""


def read_region(path: Path) -> pl.DataFrame:
    """Lê um parquet regional e traduz erros de I/O em ``DataError``.

    Raises
    ------
    DataError
        Se o arquivo não existir, estiver corrompido ou vazio.
    """
    try:
        df = pl.read_parquet(path)
    except (FileNotFoundError, pl.exceptions.ComputeError) as e:
        logger.error(f"Falha ao ler região {path.name}: {e}")
        raise DataError(f"Região ilegível: {path.name}") from e
    if df.is_empty():
        logger.error(f"Região {path.name} sem linhas")
        raise DataError(f"Região vazia: {path.name}")
    return df


def load_regions(paths: list[Path], max_failed_ratio: float) -> tuple[pl.DataFrame, list[str]]:
    """Carrega regiões tolerando ``DataError`` dentro de um orçamento explícito.

    Raises
    ------
    DegradationBudgetExceeded
        Se a fração de regiões com falha exceder ``max_failed_ratio``.
    """
    frames: list[pl.DataFrame] = []
    failed: list[str] = []
    for path in paths:
        try:
            frames.append(read_region(path))
        except DataError:  # só o erro de domínio; bugs (KeyError, TypeError...) derrubam o job
            failed.append(path.name)
    ratio = len(failed) / len(paths)
    if not frames or ratio > max_failed_ratio:
        logger.error(f"Abortando: {len(failed)}/{len(paths)} regiões falharam (limite {max_failed_ratio:.0%}): {failed}")
        raise DegradationBudgetExceeded(f"{ratio:.0%} das regiões falharam")
    if failed:
        logger.warning(f"Execução degradada: {len(failed)}/{len(paths)} regiões descartadas: {failed}")
    return pl.concat(frames), failed
```

Row level: after `pd.to_datetime(..., errors="coerce")`, compute `invalid_rate` and `raise DataError` above `cfg.max_invalid_rate`. Better still, enforce it with a schema (REQUIRED SUB-SKILL: validating-data-contracts) and translate `SchemaError` into `DataError`. The entry point maps `ConfigError`/`ModelError`/`DataError` to distinct exit codes, and it maps a completed run with skipped units to a separate "partial" code.

## Common Mistakes

| Mistake | Fix |
|---|---|
| `except Exception: logger.exception(...); continue` | Catch the domain error; budget + escalate |
| Tuple of builtins as "data errors" | Translate at source into `DataError`; catch only that |
| `errors="coerce"` + warning, no threshold | Invalid-rate budget that raises `DataError` |
| "Threshold is a follow-up" | It ships in the same change, or the degradation doesn't ship |
| `raise DataError(...)` without `from e` | Always chain. The root cause is the debugging evidence |
| Logging in every layer | Log where the context is added; top level logs the summary |
| Partial run looks identical to full run | Record skipped units in the output metadata + partial exit code |
| Fallback to an older model on `ModelError` | Fail. Stale model + new features = silently wrong scores |
