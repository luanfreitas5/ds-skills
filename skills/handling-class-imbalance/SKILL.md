---
name: handling-class-imbalance
description: Use when the positive class is rare (fraud, churn, disease, defects, default — roughly under 10%), when a model predicts almost everything as the majority class, when someone proposes SMOTE, oversampling or undersampling, when accuracy or F1 near 0.99 is reported on a rare target, or when choosing a decision threshold for a classifier.
---

# Handling Class Imbalance

## Overview

Imbalance is rarely a *training* problem; it is a *decision* and *evaluation* problem. A model that "predicts everything as negative" usually ranks fine — it is the default 0.5 threshold and accuracy that are wrong. **Core principle:** fix the threshold and the metric first; resample only if it beats that, measured in-fold, with a CI.

## Order of Operations

1. **Get the costs.** Ask for the cost of a false positive (review, offer, alert) and a false negative (lost fraud, churner, missed diagnosis). No numbers given → ask; if you must proceed, state an assumed ratio explicitly and show the threshold for 2–3 ratios. Do not fall back to "max F1" — it ignores costs.
2. **Split first.** Stratified (or temporal/grouped — REQUIRED SUB-SKILL: detecting-data-leakage). The test set keeps the real prevalence. Nothing is resampled outside a training fold.
3. **Baselines in money:** cost of "flag nobody" and "flag everybody". A model that doesn't beat both is useless regardless of AUC.
4. **Unweighted model + cost-tuned threshold** chosen on out-of-fold predictions of the training set (never on test).
5. **`class_weight="balanced"`** (or `scale_pos_weight`) + its own tuned threshold.
6. **Resampling (SMOTE, undersampling) only as a candidate**, inside `imblearn.pipeline.Pipeline` so it touches only the training folds. Keep it only if its cost/PR-AUC gain over step 4–5 has a CI excluding 0 (REQUIRED SUB-SKILL: evaluating-models-rigorously).
7. **Report:** PR-AUC with 95% CI next to prevalence (its chance level), ROC-AUC, cost at the chosen threshold vs the two baselines, recall/precision at that threshold. Never accuracy as a headline.

Measured on synthetic data (30k rows, 1.8% positive, HistGradientBoosting, FP=R$20, FN=R$600):

| Setup | PR-AUC | Cost on test |
|---|---|---|
| SMOTE on full data, then split | **0.990** (fake) | — (test is 50% synthetic) |
| SMOTE on train, then CV | **0.994** (fake) | — |
| SMOTE inside fold (honest) | 0.467 [0.386, 0.555] | 39,180 |
| No reweighting, threshold 0.5 | 0.476 [0.393, 0.561] | 54,080 |
| No reweighting, cost-tuned threshold | same | 41,400 (−23%) |
| `class_weight="balanced"` + tuned threshold | 0.492 [0.407, 0.579] | 38,020 |
| Flag nobody / flag everybody | — | 79,200 / 147,360 |

Threshold tuning delivered most of the gain. Weighted vs plain-tuned: Δ −3,380, paired-bootstrap CI [−8,880, 1,820]. SMOTE vs weighted: Δ +1,204, CI [−3,620, 6,121]. Both are indistinguishable. Weighting and SMOTE also inflate probabilities (mean predicted 0.12–0.16 vs prevalence 0.018; Brier 0.041–0.045 vs 0.012). If probabilities are consumed as probabilities, recalibrate (`CalibratedClassifierCV`) or skip reweighting.

## Implementation

```python
import logging

import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline  # sklearn Pipeline não aceita samplers
from sklearn.base import BaseEstimator
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict

logger = logging.getLogger(__name__)


def select_threshold_by_cost(
    y_true: np.ndarray, y_score: np.ndarray, cost_fp: float, cost_fn: float
) -> tuple[float, float]:
    """Escolhe o limiar que minimiza o custo total de erros.

    Parameters
    ----------
    y_true : np.ndarray
        Rótulos binários (0/1) — use predições out-of-fold do treino, nunca o teste.
    y_score : np.ndarray
        Escores/probabilidades da classe positiva.
    cost_fp, cost_fn : float
        Custo de um falso positivo e de um falso negativo.

    Returns
    -------
    tuple[float, float]
        Limiar ótimo (``inf`` = não sinalizar ninguém) e custo mínimo.

    Examples
    --------
    >>> t, c = select_threshold_by_cost(y_tr, oof, cost_fp=20, cost_fn=600)
    """
    thresholds = np.append(np.unique(y_score), np.inf)
    order = np.argsort(y_score, kind="stable")
    cum_pos = np.concatenate([[0], np.cumsum(y_true[order])])
    n_below = np.searchsorted(y_score[order], thresholds, side="left")
    fn = cum_pos[n_below]  # positivos abaixo do limiar
    fp = (len(y_true) - n_below) - (y_true.sum() - fn)
    costs = fp * cost_fp + fn * cost_fn
    best = int(np.argmin(costs))
    logger.info(f"Limiar ótimo {thresholds[best]:.4f} com custo {costs[best]:,.0f}")
    return float(thresholds[best]), float(costs[best])


def build_candidates(seed: int = 42) -> dict[str, BaseEstimator]:
    """Candidatos na ordem correta: sem peso, com peso, reamostragem dentro do fold."""
    return {
        "sem_peso": HistGradientBoostingClassifier(random_state=seed),
        "class_weight": HistGradientBoostingClassifier(class_weight="balanced", random_state=seed),
        "smote_in_fold": ImbPipeline(
            [("smote", SMOTE(random_state=seed)), ("model", HistGradientBoostingClassifier(random_state=seed))]
        ),
    }


# Uso: limiar escolhido em OOF do treino, aplicado uma vez no teste
# cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# oof = cross_val_predict(model, X_tr, y_tr, cv=cv, method="predict_proba")[:, 1]
# threshold, _ = select_threshold_by_cost(y_tr, oof, cost_fp=20, cost_fn=600)
```

PR-AUC CI: stratified bootstrap of `average_precision_score` on the test set (resample positives and negatives separately — with ~100 positives, plain bootstrap draws vary a lot).

## Common Mistakes

| Mistake | Fix |
|---|---|
| `SMOTE().fit_resample(X, y)` before split or before CV | `ImbPipeline([... SMOTE ..., model])` passed to `cross_val_*` / `fit` on train only |
| Accuracy / F1 0.99 on a 50/50 resampled test | Test keeps real prevalence; report PR-AUC vs prevalence |
| Threshold = argmax F1 | Threshold = argmin cost; F1 weights FP and FN equally |
| Threshold chosen on the test set | Choose on OOF training predictions, apply once to test |
| "Approach already approved, just add model B" on a leaky setup | Fix the evaluation (split + in-fold) — same models, different numbers; tell the reviewer |
| Keeping SMOTE because it is "the standard fix" | Compare against weighted + tuned threshold; keep only if the CI of Δ excludes 0 |
| Weighted/SMOTE scores shown as "% chance" | Recalibrate, or present as risk scores |

## Red Flags — Stop

- PR-AUC or ROC-AUC > 0.98 on a rare, hard target → resampling or other leakage
- `fit_resample` anywhere outside a Pipeline
- Test-set prevalence ≠ population prevalence
- No cost numbers anywhere in the analysis
- "Short on time — drop the unweighted / `class_weight` candidates" → those are the control; without them a SMOTE win is unfalsifiable. Cut folds or rows instead
