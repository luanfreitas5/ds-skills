---
name: type-checking-with-basedpyright
description: Use when configuring basedpyright or pyright (typeCheckingMode, pyproject.toml), when CI fails on type errors in pandas, polars, numpy or scikit-learn code, when types show as Unknown, Hashable, Scalar or "partially unknown", or when tempted to add "# type: ignore", Any, or a lower checking mode to make CI pass.
---

# Type Checking with basedpyright

## Overview

A type error in DS code has one of four causes: **wrong config**, **missing stubs**, **an untyped library**, or **a real bug** (a `None` you never handled, a `Hashable` you treat as `str`). Each has a fix at its source. `# type: ignore`, `Any` and lowering the mode hide all four at once, including the bugs.

**Core principle:** fix the cause. A suppression is allowed only when the checker is provably wrong, it names one rule, and it gives a reason.

## Triage Order (do it in this order)

1. **`pythonVersion` = your `requires-python` floor, and ≥ your libraries' floor.** pandas 3 and pandas-stubs 3 need Python ≥ 3.11. With `pythonVersion = "3.10"`, the stubs' `typing.Self` resolves to Unknown, so `df[cols]`, `select_dtypes` and `assign` become Unknown. That produces a flood of "partially unknown" errors that no code change fixes.
2. **Install stubs** in the dev group: `pandas-stubs` (pinned to your pandas minor version). numpy and polars ship their own types. Without pandas-stubs you get `Unknown` everywhere (measured: 20 → 26 warnings on the same file).
3. **Untyped libraries** (scikit-learn, many research libs): `allowedUntypedLibraries = ["sklearn"]`. Don't use a per-line ignore.
4. **Fix the code** with the table below.
5. **Too much legacy to fix now?** Run `basedpyright --writebaseline` and commit `.basedpyright/baseline.json`. Old errors are accepted and **new** ones still fail CI. It's tracked debt that shrinks, which scattered ignores are not.

## Configuration by Level

| Mode | Use for | Notes |
|---|---|---|
| `basic` / `standard` | Adopting on an existing codebase | Catches real bugs; tolerates missing annotations |
| `strict` | Stepping stone | Unknowns are errors |
| **`recommended`** | **Target for new projects** | Everything on, fails CI on warnings, **ignores bare `# type: ignore`** |
| `all` | Rarely | Adds `reportAny` etc. Don't switch to it under a deadline. |

Raise the level one step at a time, writing a baseline at each step.

```toml
[tool.basedpyright]
include = ["src", "tests"]
pythonVersion = "3.11"                  # = requires-python; never below the pandas/numpy floor
typeCheckingMode = "recommended"
allowedUntypedLibraries = ["sklearn"]   # libs without stubs, not per-line ignores
```

## Error → Fix

| Error | Cause | Fix |
|---|---|---|
| Unknown / partially unknown on `df[...]`, `select_dtypes`, `assign` | `pythonVersion` below stubs' floor, or no pandas-stubs | Steps 1–2 |
| `"Hashable" is not assignable to "str"` (`idxmax`, column labels) | Labels really are `Hashable` | `str(...)` at the boundary |
| `"/" not supported for "Series[Any]" and "Scalar"` (`df.loc[0, c]`) | `.loc` returns any scalar | `s = df[c].astype(float)` then `s.iloc[0]` (typed `float`) |
| `"PythonLiteral \| None" not assignable to "float"` (polars `mean`) | **Real bug**: empty/all-null column | `isinstance` narrow + raise |
| `__getitem__` no overload on `predict_proba(X)[:, 1]` | sklearn types it `ndarray \| list[ndarray]` (multi-output) | `cast(FloatArray, ...)`: 1-D `y` guarantees an array |
| Missing type argument for `dict` / `np.ndarray` | Bare generic | `dict[str, float]`, `npt.NDArray[np.float64]`, or a Pydantic model for configs |

## cast, Protocol, TypeAlias

- **`TypeAlias`**: name array types once (`FloatArray: TypeAlias = npt.NDArray[np.float64]`) and reuse them in every signature.
- **`cast`**: only when you know a fact the checker can't see (e.g. single-output `y`). Put it where the value leaves the untyped library, never on your own values. basedpyright rejects casts between unrelated types (`reportInvalidCast`), so casting a `RandomForestClassifier` to a Protocol fails.
- **`Protocol`**: for your own interchangeable components (data sources, feature steps). Don't use it to describe sklearn estimators: their inferred signatures don't match. Use the `cast` at the output instead.

## Implementation

Verified: 0 errors in `recommended` **and** `all`, with no ignores (basedpyright 1.40, pandas 3.0 + pandas-stubs, numpy 2.5, polars 1.44, scikit-learn 1.9).

```python
import logging
from pathlib import Path
from typing import Protocol, TypeAlias, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import polars as pl
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger(__name__)
FloatArray: TypeAlias = npt.NDArray[np.float64]
LabelArray: TypeAlias = npt.NDArray[np.int64]


class TableSource(Protocol):
    """Qualquer fonte que devolva um DataFrame (CSV, Parquet, banco...)."""

    def load(self) -> pd.DataFrame: ...


class CsvSource:
    """Fonte CSV local; satisfaz ``TableSource`` sem herdar dela."""

    def __init__(self, path: Path) -> None:
        self.path: Path = path

    def load(self) -> pd.DataFrame:
        """Lê o CSV bruto."""
        return pd.read_csv(self.path)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adiciona ``valor_log``, ``ticket_medio`` e ``ratio`` sem mutar ``df``."""
    valor = df["valor"].astype(float)  # Series[float]: .iloc[0] vira float
    ticket = df.groupby("cliente_id")["valor"].mean()
    return df.assign(valor_log=np.log1p(valor), ticket_medio=df["cliente_id"].map(ticket),
                     ratio=valor / valor.iloc[0])


def top_region(df: pd.DataFrame) -> str:
    """Retorna a região mais frequente (``idxmax`` devolve ``Hashable``)."""
    return str(df["regiao"].value_counts().idxmax())


def mean_value(df: pl.DataFrame) -> float:
    """Média de ``valor``; ``Series.mean`` do polars pode devolver ``None``."""
    media = df["valor"].mean()
    if not isinstance(media, (int, float)):
        logger.error("Média indefinida para 'valor': %r", media)
        raise ValueError("Coluna 'valor' vazia ou só com nulos.")
    return float(media)


def predict_positive_proba(X: FloatArray, y: LabelArray) -> FloatArray:
    """Treina a floresta e devolve P(classe=1); ``y`` 1-D garante ``ndarray``."""
    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)
    proba = cast(FloatArray, model.predict_proba(X))
    return proba[:, 1]


def run(source: TableSource) -> FloatArray:
    """Carrega, gera features e pontua."""
    df = add_features(source.load())
    X = df[["valor_log", "ticket_medio", "ratio"]].to_numpy(dtype=np.float64)
    y: LabelArray = np.asarray(df["inadimplente"], dtype=np.int64)
    logger.info("Pontuando %d linhas; região dominante: %s", len(df), top_region(df))
    return predict_positive_proba(X, y)
```

## When a Suppression Is Legitimate

Only when a stub is provably wrong. Use the form `# pyright: ignore[ruleName]  # stub bug: <link or reason>`. Bare `# type: ignore` **does nothing in `recommended` and `all`** (the error still fires; it only works up to `strict`), and an ignore naming a rule that doesn't fire is itself reported.

## Red Flags: Stop

| Thought | Reality |
|---|---|
| "Add `# type: ignore` to get CI green" | It doesn't suppress in `recommended`/`all`, and it hides real `None` bugs |
| "Annotate it `Any`" | `Any` spreads: every downstream value becomes unchecked |
| "Set `reportUnknownMemberType = false` project-wide" | Same as lowering the mode for every file; fix the source (steps 1–3) |
| "Lower the mode back to basic" | Use a baseline: same deadline relief, and new code is still checked |
| "The stubs are just broken" | Check `pythonVersion` and that pandas-stubs is installed first |
| "Ignore each sklearn line" | `allowedUntypedLibraries` + one `cast` at the output |
| "I've written the fix, I'll push it" | Run `basedpyright` locally. Fixes written without running it left 7 of 26 errors in testing. |

Project setup, dependency groups, and CI placement (basedpyright runs in CI, not in pre-commit): REQUIRED SUB-SKILL: scaffolding-ds-projects.
