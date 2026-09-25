---
name: refactoring-notebooks-to-src
description: Use when notebook logic must become a pipeline, script, or package in src/, when asked to "clean up", "productionize", or "turn into a script" a Jupyter notebook, or when a notebook contains hardcoded paths, global state, or cells that must run in a hidden order.
---

# Refactoring Notebooks to src/

## Overview

Copying cells into `.py` files keeps every notebook problem — globals, hidden cell order, `C:/Users/...` paths — and adds a new one: nobody can tell whether the script still produces what the notebook produced. **Core principle:** first freeze the notebook's output on a fixed sample (golden), then extract pure functions until the new pipeline reproduces that output exactly; only then change behavior, one deliberate change at a time.

## Workflow

1. **Freeze a golden output.** Run the notebook as-is on a small fixed sample (or synthetic data with the same schema); save the intermediate frames and final outputs to `tests/fixtures/golden_*.parquet`. Seed anything random.
2. **Extract pure functions**, one per cell responsibility: input → output, no globals, no I/O, no `print`. Type hints + NumPy docstring in pt-BR. Magic numbers (`2024`, `0.9`, `0.3`) become parameters.
3. **I/O only at the edges.** `load_*` / `save_*` in `data/` modules; paths come from `configs/paths.yaml` relative to the project root, never the author's desktop (raw input → `data/raw/`, outputs → `data/processed/`, models → `models/`).
4. **Equivalence test:** new functions on the golden input must equal the golden output (`assert_frame_equal`, explicit tolerance for floats).
5. **Notebook becomes a thin client:** `from features.engineering import build_features`; cells only call functions and display results. Strip outputs (`nbstripout`).
6. **Then fix behavior**, each fix its own commit that updates the golden file on purpose and says why (e.g. "quantile now fit on train only — removes leakage"). A fix hidden inside the refactor makes the equivalence test meaningless.

Tests beyond equivalence (properties, edge cases): REQUIRED SUB-SKILL: testing-ml-code. Layout, pyproject, pre-commit: REQUIRED SUB-SKILL: scaffolding-ds-projects.

## Implementation

```python
import pandas as pd


def compute_spend_features(df: pd.DataFrame, high_spend_quantile: float = 0.9) -> pd.DataFrame:
    """Calcula gasto mensal e flag de alto gasto (célula [3] do notebook 01_churn).

    Parameters
    ----------
    df : pd.DataFrame
        Clientes com ``gasto_total`` e ``meses_cliente``.
    high_spend_quantile : float, optional
        Quantil que define alto gasto, by default 0.9.

    Returns
    -------
    pd.DataFrame
        Cópia de ``df`` com ``gasto_mes`` e ``alto_gasto``.
    """
    out = df.copy()
    out["gasto_mes"] = out["gasto_total"] / out["meses_cliente"]
    cutoff = out["gasto_mes"].quantile(high_spend_quantile)
    out["alto_gasto"] = (out["gasto_mes"] > cutoff).astype(int)
    return out


# tests/test_equivalence.py
def test_spend_features_match_notebook_golden() -> None:
    """Equivalência: a função extraída reproduz a saída congelada do notebook."""
    golden_in = pd.read_parquet("tests/fixtures/golden_after_cell2.parquet")
    golden_out = pd.read_parquet("tests/fixtures/golden_after_cell3.parquet")
    pd.testing.assert_frame_equal(compute_spend_features(golden_in), golden_out, check_exact=False, rtol=1e-9)
```

## Common Mistakes

| Mistake | Fix |
|---|---|
| Cells pasted into `main.py` top to bottom | One pure function per responsibility; `main.py` only orchestrates |
| `C:/Users/ana/...` moved into a YAML | Paths relative to project root under `data/`; YAML holds the relative path |
| "Small improvements" during the move (`stratify`, `index=False`, new seed) | Equivalence first; each behavior change is a separate, named commit |
| Leakage found and "documented for later" | Fix after equivalence, in its own commit, with updated golden and a note |
| No test that old = new | Golden fixtures + `assert_frame_equal` |
| Notebook left with its own copy of the logic | Notebook imports from `src/`; logic lives in one place |
| Function mutates its input | `df.copy()`; return a new frame |
