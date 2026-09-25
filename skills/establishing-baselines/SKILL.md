---
name: establishing-baselines
description: Use when starting any modeling task, when asked to build, tune, or ship a complex model (LightGBM, XGBoost, neural net, LLM) directly, when a stakeholder has "already chosen" the algorithm, or when a reported metric has no reference point to say whether it is good.
---

# Establishing Baselines

## Overview

A metric is only meaningful relative to what a trivial approach achieves. **Core principle:** before (or alongside) any complex model, fit the cheapest credible baselines on the *same* splits and report the complex model's gain over them.

The boss choosing LightGBM does not remove the baseline — it is what proves LightGBM was worth it.

## Baseline Ladder

Always include level 0 and level 1. Add level 2 when interpretability matters.

| Task | Level 0 — trivial | Level 1 — simple, strong | Level 2 — interpretable |
|---|---|---|---|
| Binary / multiclass | `DummyClassifier(strategy="prior")` (PR-AUC = prevalence) | Logistic regression (scaled, one-hot) | Shallow tree / rules |
| Regression | `DummyRegressor(strategy="median")` | Ridge / linear | GAM or shallow tree |
| Time series forecast | Naive: `y[t] = y[t-1]` | Seasonal naive: `y[t] = y[t-12]` (monthly) | ETS / SARIMA per series |
| Ranking / recommendation | Popularity | Recent popularity per segment | Item-kNN |
| Text classification | Majority class | TF-IDF + logistic regression | Keyword rules |
| LLM task | Rule / regex / zero-shot small model | Few-shot prompt | TF-IDF + LR on labeled set |

## Implementation (time series)

```python
import numpy as np
import pandas as pd


def compute_seasonal_naive_mae(
    df: pd.DataFrame, cutoff: pd.Timestamp, season: int = 12
) -> float:
    """Calcula o MAE do baseline sazonal ingênuo (y[t] = y[t - season]) após o corte.

    Usa exatamente o mesmo período de validação do modelo complexo.
    """
    df = df.sort_values(["store_id", "date"]).copy()
    df["pred_seasonal_naive"] = df.groupby("store_id")["sales"].shift(season)
    val = df[(df["date"] >= cutoff) & df["pred_seasonal_naive"].notna()]
    return float(np.mean(np.abs(val["sales"] - val["pred_seasonal_naive"])))
```

Panel data (many series): the complex model's features and folds must be leak-free too (REQUIRED SUB-SKILL: detecting-data-leakage).
- Lags/rolling per series: `df.groupby("store_id")["sales"].transform(lambda s: s.shift(1).rolling(3).mean())` — `g.shift(1).rolling(3)` without `transform` rolls across stores.
- Folds by **date**, not row index: build rolling-origin cutoffs over unique dates; `TimeSeriesSplit(X)` on rows sorted by store mixes periods.

Report forecasts with a scale-free metric too: MASE (MAE / MAE of seasonal naive). MASE ≥ 1 → the model is worse than copying last year.

## Reporting Contract

Results table rows, in order: level 0, level 1, (level 2), complex model. Columns: main metric ± CI, relative gain vs best baseline, training time, inference latency. If the complex model does not beat the best baseline beyond the CI (REQUIRED SUB-SKILL: evaluating-models-rigorously) → recommend the baseline.

## Common Mistakes

| Mistake | Fix |
|---|---|
| Baseline evaluated on different split/period | Same folds, same cutoff, same rows |
| Only a trivial Dummy baseline | Add the level-1 simple model; beating Dummy proves little |
| Walk-forward with 1-month validation folds, 4 folds | More origins (e.g. 12 rolling origins) — 4 single-month scores are noise |
| Skipping baseline because "algorithm already chosen" | Baseline is the justification for that choice |
