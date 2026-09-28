---
name: backtesting-time-series
description: Use when evaluating, comparing, or reporting the accuracy of a forecasting model (sales, demand, load, traffic, prices), when a time-series model was validated with KFold, shuffled splits, or TimeSeriesSplit on lag features, when lag_1/rolling features feed a multi-step forecast, or when stakeholders ask for a forecast error number or an uncertainty band.
---

# Backtesting Time Series

## Overview

A forecast metric is only valid if the evaluation replays production: same forecast origin cadence, same horizon, same data latency, same retrain frequency. **Core principle:** simulate the production calendar with a rolling origin, then score against seasonal naive with MASE and check the empirical coverage of the prediction intervals.

Swapping `KFold(shuffle=True)` for `TimeSeriesSplit` is not enough: if `lag_1` is still a feature, every fold silently becomes a 1-step-ahead forecast.

## Measured on a synthetic daily series (3 years, weekly + yearly seasonality, AR(1) noise, gradient boosting)

| Evaluation | MAE |
|---|---|
| `KFold(shuffle=True)`, features with `lag_1` | 7.76 |
| `TimeSeriesSplit(test_size=7)`, still `lag_1` | 8.11 |
| Rolling origin, 52 weekly origins, h=1..7, lags ≥ 7 | **15.64** (MASE 1.12) |
| Seasonal naive `y[t-7]`, same origins | 14.54 (MASE 1.04) |

The honest error is 2x the shuffled one, and the model **loses to seasonal naive**. 80% intervals: from in-sample residuals → 41% coverage; from backtest residuals per horizon → 75%.

## Production Parity Checklist

Write these down before any code; each maps to a backtest parameter.

| Production fact | Backtest parameter |
|---|---|
| Forecast issued every Monday / every day 10:00 | `step` between origins = issue cadence |
| Forecasts Mon–Sun / next day's 24 hours | `horizon` = number of steps forecast per origin; test window = exactly that |
| Data up to Saturday available Monday / 6h meter delay | `gap` = periods between last known target and first forecast target |
| Model retrained weekly / never | retrain at every origin at that cadence (not once, not monthly if prod is daily) |
| Frequency (daily, hourly) | same frequency; do not aggregate to make scores look stabler |

Feature rule: every target-derived feature uses lag ≥ `gap + horizon` (direct model), or the backtest replays the recursive loop production uses (lag_1 for h>1 filled with the model's own predictions) — never actuals. The seasonal naive obeys the same rule: forecast `y[t - m·k]` with the smallest k such that `m·k ≥ gap + h` (gap=1, m=7: Sunday uses lag_14). Exogenous features (promo, temperature) only as *known at issue time* (planned promo, weather **forecast**), never observed values.

## Required Output Contract

1. **Parity table** — the five rows above, filled in.
2. **Backtest design** — number of origins (≥ 30 when data allows, covering at least one full season), step, horizon, gap, training window (expanding/sliding).
3. **Seasonal naive row** on the same origins (REQUIRED SUB-SKILL: establishing-baselines).
4. **MASE** for model and seasonal naive (scale = in-sample seasonal naive MAE of each training window), plus MAE in business units; ± CI by bootstrapping whole origins.
5. **Error by horizon step** h=1..H — the headline hides that h=7 is worse than h=1.
6. **Prediction intervals** calibrated on backtest residuals per horizon, with **empirical coverage** on later origins vs nominal.

If model MASE ≥ seasonal-naive MASE, the recommendation is seasonal naive.

## Implementation

```python
import logging

import numpy as np
import pandas as pd
from sklearn.base import clone

logger = logging.getLogger(__name__)


def run_rolling_origin_backtest(
    feats: pd.DataFrame, model, x_cols: list[str], horizon: int, gap: int,
    step: int, n_origins: int, season: int,
) -> pd.DataFrame:
    """Executa backtest com origem rolante, re-treinando o modelo a cada origem.

    Parameters
    ----------
    feats : pd.DataFrame
        Série única ordenada por data, com `sales`, `x_cols` e `lag_{season}`;
        lags do alvo devem ser >= gap + horizon (aqui gap + horizon <= season,
        então `lag_{season}` é um naive sazonal emitível).
    model : estimator
        Estimador scikit-learn (clonado a cada origem).
    x_cols : list[str]
        Colunas de features.
    horizon, gap, step, n_origins, season : int
        Horizonte, latência dos dados, cadência de emissão, número de origens e período sazonal.

    Returns
    -------
    pd.DataFrame
        Uma linha por (origem, passo h) com real, previsão, naive sazonal e escala do MASE.
    """
    n, rows = len(feats), []
    for k in range(n_origins):
        test_end = n - (n_origins - 1 - k) * step
        test_start = test_end - horizon
        train = feats.iloc[: test_start - gap]  # só o que existiria na emissão
        test = feats.iloc[test_start:test_end]
        fitted = clone(model).fit(train[x_cols], train["sales"])
        y_tr = train["sales"].to_numpy()
        rows.append(pd.DataFrame({
            "origin": k, "h": np.arange(1, horizon + 1), "y": test["sales"].to_numpy(),
            "pred": fitted.predict(test[x_cols]), "snaive": test[f"lag_{season}"].to_numpy(),
            "scale": np.mean(np.abs(y_tr[season:] - y_tr[:-season])),
        }))
    logger.info(f"Backtest concluído: {n_origins} origens, horizonte {horizon}, gap {gap}")
    return pd.concat(rows, ignore_index=True)


def compute_mase(bt: pd.DataFrame, col: str) -> float:
    """Calcula o MASE médio (erro absoluto / MAE in-sample do naive sazonal)."""
    return float((np.abs(bt["y"] - bt[col]) / bt["scale"]).mean())


def evaluate_interval_coverage(bt: pd.DataFrame, level: float = 0.8) -> float:
    """Calibra intervalos por horizonte na 1ª metade das origens e mede a cobertura na 2ª."""
    a = (1 - level) / 2
    cal = bt[bt["origin"] < bt["origin"].median()]
    ev = bt[bt["origin"] >= bt["origin"].median()]
    q = (cal["y"] - cal["pred"]).groupby(cal["h"]).quantile([a, 1 - a]).unstack()
    lo, hi = ev["pred"] + ev["h"].map(q[a]), ev["pred"] + ev["h"].map(q[1 - a])
    coverage = float(((ev["y"] >= lo) & (ev["y"] <= hi)).mean())
    logger.info(f"Cobertura empírica {coverage:.3f} (nominal {level:.2f})")
    return coverage
```

Panel data (many stores): build lags per series with `groupby(...).shift`, define origins on **dates**, not row positions; `TimeSeriesSplit` on rows sorted by store mixes periods (REQUIRED SUB-SKILL: detecting-data-leakage). Intervals: for production, recalibrate on a sliding window of the most recent backtest/live residuals, per horizon step.

## Common Mistakes

| Mistake | Fix |
|---|---|
| `KFold(shuffle=True)` on a series | Rolling origin; the model interpolates between neighbours otherwise |
| "Switched to TimeSeriesSplit, done" with `lag_1` | Lags ≥ gap + horizon, or recursive forecasting inside the fold |
| Test window of a month from one origin when prod forecasts 7 days | Test window = production horizon; many origins instead |
| No gap despite data latency | `gap` = latency; train ends where production data ends |
| Retrain monthly in backtest, daily in production | Same retrain cadence (or state the mismatch as a limitation) |
| Seasonal naive uses `lag_7` for every h even when gap + h > 7 | Naive "as issued": lag `m·ceil((gap+h)/m)` |
| MAPE/R² as headline | MASE vs seasonal naive (scale-free, defined with zeros); R² is inflated by seasonality |
| Intervals from training residuals | In-sample residuals are too small (41% vs 80% nominal); use out-of-sample residuals per h |
| Reporting interval width without coverage | Coverage vs nominal is the check; width alone is meaningless |
| Observed weather/promo as features | Use values as known at issue time, or label result an "oracle" upper bound |
