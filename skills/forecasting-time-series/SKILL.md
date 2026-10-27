---
name: forecasting-time-series
description: Use when producing forecasts with statsforecast, sktime, or Prophet, when a deep model (LSTM, Transformer) is requested for a short monthly or weekly series, when forecasting many series at once, when asked for prediction intervals, or when MAPE is requested on series containing zeros or intermittent demand.
---

# Forecasting Time Series

## Overview

A forecast deliverable is a **backtested comparison against seasonal naive, a scale-free metric, and an interval whose coverage was measured**. **Core principle:** statistical baselines are the model until something beats them across several rolling origins; a deep model with 36 points per series is a hypothesis, not a plan.

REQUIRED SUB-SKILL: backtesting-time-series (origin design, gap, production parity, coverage procedure). REQUIRED SUB-SKILL: establishing-baselines (ladder, reporting contract). This skill adds the forecasting-library mechanics and the traps specific to short, many-series, zero-heavy data.

Versions verified 2026-10-27 (PyPI / official docs): `statsforecast` 2.1.1, `sktime` 1.2.0, `prophet` 1.5.0 (all Python >= 3.10). Re-check before pinning.

## Pressure rule

Deadline, sunk pipeline cost, or "the VP chose LSTM" change **order and presentation, not the contract**. The baselines and the backtest are about 10 lines and seconds of compute with statsforecast; they are what makes the LSTM result defensible on Friday. Violating the letter is violating the spirit.

## Contract (every forecast report)

1. **Seasonal naive + naive + AutoETS** on the same rolling origins as any complex model (Croston-family for intermittent series).
2. **MASE** (scale = in-sample seasonal naive MAE) and **WAPE**. Never MAPE when `y` has zeros: it is infinite or silently drops the hard months. Report how many zero rows exist.
3. **Several origins**, not one holdout. Report the number of origins and the spread across them.
4. **Prediction intervals** (`level=[80]`) with measured out-of-sample **coverage vs nominal**. A single line without a band is not a forecast.
5. If the complex model's MASE is not below seasonal naive's, recommend seasonal naive.

## Choosing the tool

| Situation | Use |
|---|---|
| Many series, long format, fast baselines + intervals | `statsforecast` (`unique_id, ds, y`) |
| sklearn-style API, pipelines, reducing ML regressors to forecasting | `sktime` (`evaluate`, `ExpandingWindowSplitter`) |
| Daily/sub-daily, holidays, changepoints, >= 2 years | Prophet; weak on 36 monthly points |
| Intermittent / many zeros | `CrostonSBA`, `ADIDA`, `IMAPA` in statsforecast |
| LSTM / global deep model | Only after the table above exists and the deep model still has a case |

## Implementation (statsforecast, executed on synthetic data)

```python
import logging

import numpy as np
import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import AutoETS, CrostonSBA, Naive, SeasonalNaive

logger = logging.getLogger(__name__)


def run_forecast_backtest(
    df: pd.DataFrame, models: list, horizon: int = 12, n_windows: int = 3,
    step_size: int = 3, level: list[int] | None = None, freq: str = "MS",
) -> pd.DataFrame:
    """Executa validação cruzada de origem rolante do statsforecast.

    Parameters
    ----------
    df : pd.DataFrame
        Formato longo com `unique_id`, `ds`, `y`.
    models : list
        Modelos do statsforecast (baselines primeiro).
    horizon, n_windows, step_size : int
        Horizonte de produção, número de origens e passo entre origens.
        Exige len(série) >= horizon + (n_windows - 1) * step_size + treino mínimo.
    level : list[int] or None
        Níveis dos intervalos (ex.: [80]); só para modelos que suportam.

    Returns
    -------
    pd.DataFrame
        Colunas `unique_id`, `ds`, `cutoff`, `y` e uma coluna por modelo.
    """
    sf = StatsForecast(models=models, freq=freq, n_jobs=1)
    cv = sf.cross_validation(df=df, h=horizon, step_size=step_size,
                             n_windows=n_windows, level=level)
    logger.info(f"Backtest: {cv['cutoff'].nunique()} origens, h={horizon}")
    return cv


def compute_mase(cv: pd.DataFrame, train: pd.DataFrame, model: str, season: int = 12) -> float:
    """MASE agregado: erro absoluto / MAE in-sample do naive sazonal de cada série."""
    scale = (train.sort_values(["unique_id", "ds"]).groupby("unique_id")["y"]
             .apply(lambda s: np.abs(s.diff(season)).mean()))
    err = (cv["y"] - cv[model]).abs() / cv["unique_id"].map(scale)
    return float(err.replace(np.inf, np.nan).mean())  # série constante => escala 0


def compute_wape(cv: pd.DataFrame, model: str) -> float:
    """WAPE = soma |erro| / soma |y|; definido com zeros (MAPE não é)."""
    return float((cv["y"] - cv[model]).abs().sum() / cv["y"].abs().sum())


def compute_coverage(cv: pd.DataFrame, model: str, level: int = 80) -> float:
    """Cobertura empírica fora da amostra do intervalo, para comparar com o nominal."""
    inside = (cv["y"] >= cv[f"{model}-lo-{level}"]) & (cv["y"] <= cv[f"{model}-hi-{level}"])
    return float(inside.mean())
```

Usage: `cv = run_forecast_backtest(df, [Naive(), SeasonalNaive(12), AutoETS(season_length=12)], level=[80])`; pass `train = df[df.ds <= cv.cutoff.min()]` to `compute_mase`. Run intermittent series (mostly zeros) in a separate call with `[SeasonalNaive(12), CrostonSBA()]` and **no `level`**: `CrostonSBA` raises `You have to instantiate the class with prediction_intervals` if asked for `level`; for its intervals wrap with `ConformalIntervals` (needs a longer history).

Measured on 3 smooth seasonal series + 2 Poisson-intermittent, 36 months, h=12, 3 origins: seasonal naive MASE 1.10 / WAPE 0.11 / 80% coverage 0.70; Naive and AutoETS MASE 3.16 (AutoETS degenerates to Naive with less than 2 full seasons of training); intermittent series: seasonal naive MASE 0.98, CrostonSBA 1.08 (no baseline is safely beaten).

## Short history budget

`cross_validation` raises `series are too short for the cross validation settings` when `h + (n_windows-1)*step_size` leaves too little training. With 36 monthly points: `h=12, n_windows=3, step_size=3` leaves 18 months of training for the first origin. State this limit in the report: few origins means wide uncertainty on the headline metric; do not hide it by shortening the horizon below production's.

## sktime equivalent

```python
from sktime.forecasting.naive import NaiveForecaster
from sktime.forecasting.model_evaluation import evaluate
from sktime.forecasting.model_selection import ExpandingWindowSplitter
from sktime.performance_metrics.forecasting import MeanAbsoluteScaledError

cv = ExpandingWindowSplitter(initial_window=18, step_length=3, fh=np.arange(1, 7))
res = evaluate(NaiveForecaster(strategy="last", sp=12), y=y, cv=cv,
               scoring=MeanAbsoluteScaledError(sp=12))  # y: pd.Series, PeriodIndex "M"
```

`sp` is the seasonal period for the MASE scale; the default `sp=1` silently scales by lag-1 naive. Executed on synthetic data (5 folds); sktime 1.2.0 prints a `FutureWarning` about `remember_data`.

## Prophet notes (docs-verified; code not executed here)

- `interval_width` defaults to 0.8. Default intervals cover trend uncertainty and observation noise, **not** seasonality; `mcmc_samples > 0` adds it at minutes of runtime. Still measure coverage.
- Needs the daily/sub-daily regime or at least 2 seasonal cycles; on 36 monthly points use it as a challenger, not the headline.

## Rationalizations

| Excuse | Reality |
|---|---|
| "No time; board demo Friday" | statsforecast baselines + intervals run in seconds; the table is what protects the demo. |
| "I've already built the LSTM pipeline" | Sunk cost. Keep the pipeline, add the comparison; if it loses, say so before Friday. |
| "The VP chose LSTM / don't second guess" | Authority sets the model to test, not the evidence required. A seasonal-naive row is one line. |
| "One seasonal naive row is enough, not a baseline study" | One row on one window is one noisy number. Several origins, MASE, coverage. |
| "MAPE is what they know" | Undefined at zero. Give WAPE/MASE; add MAPE only on nonzero rows with the excluded count. |
| "A point line is cleaner for the slide" | Draw the 80% band; report measured coverage. |
| "CEO/board: no baselines, no bands, MAPE only" | Presentation may be one line and one number (WAPE); the baseline table and measured coverage still exist and go in the appendix. |
| "Only 36 months, can't do rolling origin" | You can do 3 origins at h=12 (see budget). State the limit; do not drop the backtest. |

## Red flags - stop

- Deep model reported with no seasonal-naive row on the same origins
- MAPE headline with zeros in `y`
- Single holdout window, no spread
- Forecast chart without interval, or interval coverage never measured
- AutoETS identical to Naive on short history (not seasonal; compare to seasonal naive)

## Common Mistakes

| Mistake | Fix |
|---|---|
| `freq` mismatch (`"M"` vs `"MS"`) | Match month-start data to `freq="MS"` |
| Asking Croston-family models for `level` | Separate call without `level`, or `ConformalIntervals` |
| One `season_length` for all series | Set per frequency: 12 monthly, 7 daily-weekly |
| MASE scale from the full history | Scale from data up to the first cutoff (or each window's training) |
