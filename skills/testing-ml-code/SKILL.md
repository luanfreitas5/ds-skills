---
name: testing-ml-code
description: Use when writing tests for data transformations, feature engineering, preprocessing, or trained ML models, when a coverage target (e.g. 80%) must be met, or when model tests would be skipped because the artifact is missing in CI.
---

# Testing ML Code

## Overview

Example-based tests on three hand-picked rows miss the edge cases that break pipelines, and "predict returns 0/1" says nothing about whether the model learned the right thing. **Core principle:** test transformations with **properties** over generated data, and test models with **behavior** (directional, invariance, minimum functionality) plus a **metric regression gate**.

## Test Layers

| Layer | For | Tool | Example |
|---|---|---|---|
| Unit (examples) | Exact known outputs, boundaries | `pytest` | `age=60 → is_senior=True` |
| Edge cases | Empty frame, all-null, zero divisor, wrong dtype | `pytest.mark.parametrize` | `tenure_months=0` has a *specified* result |
| Property-based | Invariants over any valid input | `hypothesis` | Output rows = input rows; no new NaN; columns ⊇ input |
| Data contract | Output matches schema | `pandera` | `Schema.validate(result)` |
| Directional (model) | Known monotonic relation | `pytest` | ↑ `monthly_charges` should not ↓ churn prob (if business agrees) |
| Invariance (model) | Irrelevant change → same prediction | `pytest` | Changing pseudonymous ID does not change score |
| Minimum functionality | Obvious cases | `pytest` | Clear churner scores above clear loyal customer |
| Metric regression | Headline metric ≥ agreed floor on frozen test set | `pytest -m slow` | `average_precision ≥ 0.45` |

## Rules

- **Specify edge behavior, then test it.** `tenure_months = 0` → define (null? total_spend?) in the function and assert that exact value — never `assert val is None or val == inf`.
- **Model tests never silently skip in CI.** Missing artifact → CI pulls it (`dvc pull`) or trains a tiny model in a fixture. `skipif(not exists)` turns the model suite into a no-op that still reports green.
- **Behavioral tests on a fixture model** trained on synthetic data with known structure, plus the same tests run against the real artifact in the slow suite.
- Imports: `from features.engineering import compute_features` (`pythonpath = ["src"]`), never `from src...`.
- Coverage is a floor, not the goal. Meet 80% with the layers above, not with asserts that cannot fail.

## Implementation

```python
import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.base import ClassifierMixin

from features.engineering import compute_features


@given(
    total_spend=st.lists(st.floats(0, 1e6, allow_nan=False), min_size=1, max_size=50),
    data=st.data(),
)
@settings(max_examples=200, deadline=None)
def test_compute_features_preserves_rows_and_columns(total_spend: list[float], data: st.DataObject) -> None:
    """Propriedade: nunca altera o número de linhas nem remove colunas de entrada."""
    n = len(total_spend)
    df = pl.DataFrame({
        "total_spend": total_spend,
        "tenure_months": data.draw(st.lists(st.integers(1, 240), min_size=n, max_size=n)),
        "age": data.draw(st.lists(st.integers(18, 100), min_size=n, max_size=n)),
    })
    result = compute_features(df)
    assert result.height == df.height
    assert set(df.columns) <= set(result.columns)
    assert (result["spend_per_month"] >= 0).all()


def test_churn_probability_non_decreasing_with_monthly_charges(churn_model: ClassifierMixin) -> None:
    """Teste direcional: aumentar monthly_charges não reduz a probabilidade de churn."""
    base = pl.DataFrame({"monthly_charges": [50.0] * 5, "tenure_months": [12] * 5})
    varied = base.with_columns(pl.Series("monthly_charges", [20.0, 50.0, 80.0, 110.0, 140.0]))
    proba = churn_model.predict_proba(varied.to_pandas())[:, 1]
    assert np.all(np.diff(proba) >= -1e-3), f"Probabilidades não monotônicas: {proba}"


@pytest.mark.slow
def test_average_precision_does_not_regress(churn_model, frozen_test_set) -> None:
    """Regressão de métrica: AP no conjunto de teste congelado não cai abaixo do piso acordado."""
    from sklearn.metrics import average_precision_score

    X, y = frozen_test_set
    ap = average_precision_score(y, churn_model.predict_proba(X)[:, 1])
    assert ap >= 0.45, f"Regressão de métrica detectada: AP={ap:.4f}"
```

`churn_model` / `frozen_test_set` live in `tests/conftest.py`: a session fixture loads the DVC-pulled artifact and fails (not skips) when absent.

## Common Mistakes

| Mistake | Fix |
|---|---|
| Only 3-row example tests | Add `hypothesis` properties |
| `skipif(model missing)` | `dvc pull` in CI or fixture-trained model; fail loudly |
| Asserting "predictions are 0/1" as model test | Directional, invariance, minimum-functionality tests |
| No metric floor | Metric regression test on frozen test set |
| Assert accepts several outcomes | Specify one behavior, assert it |
