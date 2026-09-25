---
name: evaluating-models-rigorously
description: Use when reporting ML model performance, comparing two or more models, declaring a "best" model, writing a results section for a paper or stakeholder report, or when metrics differ by small margins (e.g. AUC 0.861 vs 0.853), on imbalanced targets, or when probabilities drive decisions.
---

# Evaluating Models Rigorously

## Overview

A metric without uncertainty is an anecdote. **Core principle:** every reported number carries a confidence interval, every "A beats B" claim carries a *correct* significance test, and aggregate metrics are broken down by slice.

## When to Use

- Writing results for a paper, report, or go/no-go meeting
- Choosing between models whose scores are close
- Target is imbalanced (prevalence < ~20%)
- Model probabilities feed a threshold, a price, or a ranking

Not for: quick exploratory sanity checks that will never be reported.

## Required Output Contract

Every evaluation you deliver contains these parts, in order:

1. **Metric justification** — main metric tied to cost of FP vs FN. Imbalanced target: report PR-AUC (average precision) next to ROC-AUC; state prevalence as PR-AUC chance level.
2. **Baseline row** — dummy / simple model on the *same* splits (REQUIRED SUB-SKILL: establishing-baselines).
3. **Point estimate ± 95% CI** for each model (repeated CV or bootstrap on test set).
4. **Paired comparison** — Δ, CI of Δ, and p-value from a *correct* test (table below).
5. **Calibration** — Brier score + reliability curve when probabilities are used.
6. **Slice table** — main metric per relevant subgroup (region, segment, period).
7. **Verdict wording** that matches the test: not significant → "statistically indistinguishable"; choose on secondary criteria (interpretability, latency, calibration).

## Choosing the Test

| Situation | Test | Do NOT use |
|---|---|---|
| Two models, same CV folds | Nadeau–Bengio corrected resampled *t*-test on repeated CV (e.g. 10×5) | Plain `scipy.stats.ttest_rel` on folds — folds share training data, variance is underestimated, p-values are too small |
| Two classifiers, one held-out test set, hard labels | McNemar (`statsmodels.stats.contingency_tables.mcnemar`) | Comparing accuracies by eye |
| Two models, one test set, threshold-free metric (AUC) | Paired bootstrap of Δ (≥ 2,000 resamples) or DeLong | Unpaired bootstrap of each model |
| 3+ models | Friedman + Nemenyi post-hoc, or pairwise tests with Holm correction | Many uncorrected pairwise tests |

## Implementation

```python
import numpy as np
from scipy import stats
from sklearn.base import BaseEstimator
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score


def compare_models_corrected_ttest(
    model_a: BaseEstimator,
    model_b: BaseEstimator,
    X: np.ndarray,
    y: np.ndarray,
    scoring: str = "average_precision",
    n_splits: int = 5,
    n_repeats: int = 10,
    seed: int = 42,
) -> dict[str, float]:
    """Compara dois modelos com o t-test reamostrado corrigido (Nadeau & Bengio, 2003).

    Usa os MESMOS folds para ambos os modelos (teste pareado).
    """
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    scores_a = cross_val_score(model_a, X, y, cv=cv, scoring=scoring)
    scores_b = cross_val_score(model_b, X, y, cv=cv, scoring=scoring)
    diff = scores_a - scores_b
    k = len(diff)
    test_train_ratio = 1 / (n_splits - 1)  # n_test / n_train
    corrected_var = (1 / k + test_train_ratio) * diff.var(ddof=1)
    t_stat = diff.mean() / np.sqrt(corrected_var)
    p_value = 2 * stats.t.sf(abs(t_stat), df=k - 1)
    half_width = stats.t.ppf(0.975, df=k - 1) * np.sqrt(corrected_var)
    return {
        "mean_a": scores_a.mean(),
        "mean_b": scores_b.mean(),
        "delta": diff.mean(),
        "delta_ci_low": diff.mean() - half_width,
        "delta_ci_high": diff.mean() + half_width,
        "p_value": p_value,
    }
```

Single-model CI on repeated CV uses the same correction — `1.96 * std / sqrt(k)` treats correlated folds as independent and is too narrow:

```python
def compute_corrected_cv_ci(scores: np.ndarray, n_splits: int) -> tuple[float, float]:
    """IC 95% da média de CV com a correção de Nadeau & Bengio para folds correlacionados."""
    k = len(scores)
    se = np.sqrt((1 / k + 1 / (n_splits - 1)) * scores.var(ddof=1))
    half_width = stats.t.ppf(0.975, df=k - 1) * se
    return scores.mean() - half_width, scores.mean() + half_width
```

Calibration: `sklearn.metrics.brier_score_loss` + `sklearn.calibration.CalibrationDisplay.from_predictions`. Slices: group predictions by subgroup and recompute the metric; show `n` per slice (small slices → wide CI, say so).

## Common Mistakes

| Mistake | Fix |
|---|---|
| `ttest_rel` on 5 CV folds | Corrected t-test on repeated CV; 5 paired points is too few anyway |
| "Model A is best" from Δ = 0.008 | Report Δ with CI; if CI contains 0 → indistinguishable |
| ROC-AUC only, 8% positives | Add PR-AUC; ROC-AUC looks good on imbalanced data even when precision is poor |
| `scores.std()` or `1.96 * std / sqrt(k)` reported as CI | Folds are correlated; use `compute_corrected_cv_ci` or bootstrap on a held-out test set |
| Only aggregate metric | Slice table; aggregate hides subgroup failures |
| Threshold 0.5 by default | Choose threshold from cost of FP/FN on validation data, never on test |
| Tuning and reporting on the same folds | Nested CV, or tune on train and report once on untouched test |

## Red Flags — Stop and Fix

- "Reviewers won't dig" / "keep it simple" → the CI and the test are the simple, defensible version
- Claiming superiority without a p-value or CI of the difference
- Any single number in a results table without ±
