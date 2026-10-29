---
name: checking-probability-calibration
description: Use when model probabilities are used to pick a threshold, price risk, or rank clients, when a deck says "a score of 0.30 means a 30% chance", when fitting CalibratedClassifierCV, choosing Platt (sigmoid) vs isotonic, plotting a calibration curve or reporting a Brier score, when the model was trained on SMOTE/oversampled or reweighted data, or when someone wants conformal prediction for a classifier.
---

# Checking Probability Calibration

## Overview

A calibrated score of 0.30 means "among clients like this, about 30% default" — **in the population the calibrator saw**. That sentence is only as true as the data it was fit and checked on. **Core principle:** fit the calibrator on real-prevalence data that is disjoint from the data you evaluate on, prove it beats "predict the prevalence", and support every "X% chance" claim with an observed rate, its `n`, and a CI from held-out data.

## Order of Operations

1. **Split first; the test set keeps the real prevalence.** No resampling outside training folds (REQUIRED SUB-SKILL: handling-class-imbalance). If the scores are consumed as probabilities, drop SMOTE/oversampling and use the unweighted model (it inflated mean probability 4× here); keep resampling only if, in-fold and recalibrated, it beats that on OOF log loss.
2. **Calibrator data ≠ evaluation data.** `CalibratedClassifierCV(base, method=..., cv=5)` fit on the training set, or `CalibratedClassifierCV(FrozenEstimator(fitted), ...)` fit on a dedicated calibration split. `cv="prefit"` is gone (scikit-learn 1.9.1 raises `InvalidParameterError`) — use `sklearn.frozen.FrozenEstimator`.
3. **Choose sigmoid vs isotonic on out-of-fold training predictions** (log loss or Brier), never on the test set. Isotonic is a step function with up to one step per distinct score; with a few hundred rows or a few dozen positives it overfits — default to sigmoid there. Under a hard deadline, skip the selection and use sigmoid with `cv=5`.
4. **Evaluate once on the test set:** Brier with a paired-bootstrap CI of Δ vs the prevalence-only baseline, plus a reliability table with `n` and a CI per bin (quantile bins). If the Δ CI contains 0, the model does not beat "everyone has the base rate" — say so.
5. **"Score s means s% chance" claims** need the observed rate among held-out rows scored in [s − 0.05, s + 0.05], with `n` and Clopper–Pearson CI. Supported only if n ≥ 30 and the CI contains s; otherwise the claim is unsupported at that score — state the score range where the reliability table does support it. Slide sentence shape: *"Among held-out clients scored A–B (n = N, K defaults), R% defaulted (95% CI L–U%); valid at the current ~P% default rate."*
6. **Scope the claim:** valid for the period and prevalence of the calibration data. Prevalence or policy changes → recalibrate and monitor (REQUIRED SUB-SKILL: monitoring-model-drift).
7. **Thresholds** come from costs on calibrated OOF probabilities (REQUIRED SUB-SKILL: handling-class-imbalance). "Threshold where the probability equals the base rate" is a rule nobody justified — ask what FP and FN cost; until answered, show the threshold for 2–3 stated cost ratios.

Measured on synthetic credit data (1,200 rows, 5.6% defaults, GradientBoosting; truth checked on 200k fresh rows from the same generator):

| Setup | Brier, as reported | Brier on fresh population | ECE (population) | Clients scored ≈0.30 actually default |
|---|---|---|---|---|
| SMOTE on all data → split → isotonic `prefit` on 200 test rows, scored on the same rows | **0.075, ECE 0.000** (looks perfect) | 0.113 | 0.125 | **10%** |
| Predict the prevalence (5.6%) for everyone | — | 0.054 | — | — |
| Uncalibrated GB, real prevalence | — | 0.059 (worse than prevalence) | 0.040 | — |
| SMOTE on train, not recalibrated | — | 0.123 | 0.161 | — |
| Split first, `cv=5` sigmoid | 0.050 on test | **0.052** | 0.009 | 25% |
| Split first, `cv=5` isotonic | 0.051 on test | 0.052 | 0.012 | 22% |
| Oracle (true probabilities) | — | 0.046 | 0 | 30% |

- The honest test set had 240 rows and 13 defaults. Brier Δ vs prevalence: CI [−0.006, +0.002] (sigmoid), [−0.006, +0.005] (isotonic) — on this data you cannot show the model beats the base rate. Only 2–3 test clients scored 0.20–0.40, so "0.30 = 30%" could not be checked at all; the population shows it was really 22–25%.
- Isotonic vs sigmoid on 200 calibration rows (50 repetitions): population Brier sd 0.0024 vs 0.0013; isotonic's worst run (0.060) was worse than predicting the prevalence.
- If the hyperparameters were tuned against a leaked (resampled-before-split) test set, they were selected on inflated metrics too — tell the user, even if re-tuning waits.

## Implementation

```python
import logging

import numpy as np
import pandas as pd
from scipy.stats import beta
from sklearn.base import BaseEstimator
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict

logger = logging.getLogger(__name__)


def select_calibration_method(
    base: BaseEstimator, X_train: np.ndarray, y_train: np.ndarray, seed: int = 42
) -> str:
    """Escolhe sigmoid ou isotonic pelo log loss out-of-fold no TREINO (nunca no teste).

    Parameters
    ----------
    base : BaseEstimator
        Classificador não treinado.
    X_train, y_train : np.ndarray
        Dados de treino com a prevalência real (sem reamostragem).
    seed : int, optional
        Semente dos folds, by default 42.

    Returns
    -------
    str
        ``"sigmoid"`` ou ``"isotonic"``.

    Examples
    --------
    >>> method = select_calibration_method(HistGradientBoostingClassifier(), X_tr, y_tr)
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    losses = {}
    for method in ("sigmoid", "isotonic"):
        calibrated = CalibratedClassifierCV(base, method=method, cv=5)
        oof = cross_val_predict(calibrated, X_train, y_train, cv=cv, method="predict_proba")[:, 1]
        losses[method] = log_loss(y_train, oof)
    logger.info(f"Log loss OOF por método: {losses}")
    return min(losses, key=losses.get)


def compute_brier_with_baseline(
    y_true: np.ndarray, y_prob: np.ndarray, n_boot: int = 2000, seed: int = 42
) -> dict[str, float]:
    """Brier do modelo e do baseline "prevê a prevalência", com IC 95% pareado da diferença."""
    rng = np.random.default_rng(seed)
    base = np.full_like(y_prob, y_true.mean(), dtype=float)
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        deltas.append(brier_score_loss(y_true[idx], y_prob[idx]) - brier_score_loss(y_true[idx], base[idx]))
    low, high = np.percentile(deltas, [2.5, 97.5])
    return {
        "brier_model": brier_score_loss(y_true, y_prob),
        "brier_prevalence": brier_score_loss(y_true, base),
        "delta_ci_low": float(low),
        "delta_ci_high": float(high),  # IC < 0 inteiro → o modelo supera o baseline
    }


def compute_reliability_table(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 5) -> pd.DataFrame:
    """Taxa observada por faixa de probabilidade, com n e IC 95% (Clopper-Pearson) por faixa."""
    bins = pd.qcut(y_prob, q=n_bins, duplicates="drop")
    table = (
        pd.DataFrame({"bin": bins, "y": y_true, "p": y_prob})
        .groupby("bin", observed=True)
        .agg(n=("y", "size"), positives=("y", "sum"), mean_predicted=("p", "mean"), observed=("y", "mean"))
    )
    k, n = table["positives"], table["n"]
    table["ci_low"] = np.where(k > 0, beta.ppf(0.025, k, n - k + 1), 0.0)
    table["ci_high"] = np.where(k < n, beta.ppf(0.975, k + 1, n - k), 1.0)
    return table
```

Verified with scikit-learn 1.9.1, scipy, pandas 3.1.0. Plot: `CalibrationDisplay.from_predictions(y_te, p_te, n_bins=5, strategy="quantile")`, drawn from the same table (REQUIRED SUB-SKILL: plotting-publication-figures).

**Conformal prediction is not calibration.** `mapie.classification.SplitConformalClassifier(fitted, confidence_level=0.9, prefit=True).conformalize(X_cal, y_cal)` (mapie 1.6.0) gives prediction *sets* with *marginal* coverage — dominated by the majority class on rare targets. On the data above, 90% of clients got the set {0, 1}. Use it when the decision needs "the true label is in this set ≥ 90% of the time", never as a per-client default probability.

## Common Mistakes

| Mistake | Fix |
|---|---|
| Calibrator fit and scored on the same rows | Calibrate with `cv=5` on train (or a separate split); score once on test |
| `cv="prefit"` | `CalibratedClassifierCV(FrozenEstimator(model))` on a calibration split |
| Calibrating on SMOTE/50-50 data | Calibrate on real-prevalence data; a 50/50 calibrator learns a 50% world |
| Isotonic on ~200 rows "because it's more flexible" | Sigmoid; choose by OOF log loss on train |
| "Run both and keep whichever is better on the test set" | That makes the test set a validation set; choose on OOF |
| Brier reported alone | Brier next to the prevalence-only Brier, with CI of Δ |
| Reliability curve without `n` per bin | Table with `n`, observed rate and CI per bin; few rows → wide CI, say so |
| "0.30 = 30% chance" from the curve's shape | Observed rate near 0.30 on held-out data, with `n`; none → claim unsupported |

## Rationalizations

| Excuse | Reality |
|---|---|
| "I take full responsibility" | Responsibility doesn't change what the 200 rows can show; the committee acts on the number, not on who signed it |
| "We've used SMOTE for years / the committee never complained" | Ranking (AUC) survives SMOTE; probabilities don't. Nobody complained because nobody checked observed rates |
| "The previous deck had the same sentence" | Then it was unchecked too; check this one |
| "Give me mine, add yours as a backup slide" | The leaky version is not delivered in any slot — primary, backup or appendix |
| "Fit isotonic on the 200-row slice, score on the rest of the test" | Still isotonic on ~200 rows on top of a model trained on pre-split SMOTE; use `cv=5` on train |
| "No time" | Sigmoid `cv=5` + one test-set evaluation is the fast path; the leaky one is not faster |

## Red Flags — Stop

- ECE ≈ 0 or a reliability curve on the diagonal from a few hundred rows → calibrator scored on its own fit data
- Mean predicted probability far from the real prevalence (e.g. 0.16–0.21 vs 0.06) → resampled or reweighted data upstream
- "No redesign, just calibrate" on a pipeline that resamples before the split → reordering the same steps is not a redesign; the leak changes the numbers the slide quotes
- "The committee/lead only needs one sentence" → the sentence is the claim; it carries the `n` and the score range where it was checked
- "I'm the lead, just do it" / "no redesign" → deliver the corrected pipeline (same model, same steps, fixed order), list each literal instruction you changed and why, and state in writing which sentence the data does not support. Do not also hand over the leaky version
