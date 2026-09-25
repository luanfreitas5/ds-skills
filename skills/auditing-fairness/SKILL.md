---
name: auditing-fairness
description: Use when a model's outputs affect people (credit, hiring, health, insurance, pricing, public services), when data contains sensitive attributes (sex, age, race, region, disability) or proxies for them, before a go/no-go or production decision, or when someone proposes "just drop the sensitive columns" as the fairness fix.
---

# Auditing Fairness

## Overview

**Core principle:** fairness is measured on outcomes per group, not assumed from inputs. Removing `sex` or `region` from features ("fairness through unawareness") does not remove bias — correlated features (income, CEP, occupation) act as proxies. Keep sensitive attributes *out of the model* but *in the audit*.

## Audit Procedure

1. **Declare sensitive attributes** and the legal/ethical reason for each (LGPD art. 11 for sensitive data; Brazilian Constitution art. 5 for sex/race). Include intersections (sex × region) when group sizes allow.
2. **Proxy check:** predict each sensitive attribute from the model features. AUC > 0.7 → strong proxies exist; unawareness is not protecting anyone.
3. **Per-group metrics** with `fairlearn.metrics.MetricFrame`: selection rate, TPR (recall), FPR, precision, plus `n` per group and bootstrap CI. Small `n` → say "inconclusive", never "fair".
4. **Disparity metrics** vs pre-agreed thresholds (define *before* looking at results):

| Metric | fairlearn | Typical threshold |
|---|---|---|
| Demographic parity ratio | `demographic_parity_ratio` | ≥ 0.8 ("four-fifths rule") |
| Equalized odds difference | `equalized_odds_difference` | ≤ 0.1 |
| TPR gap (equal opportunity) | `MetricFrame(...).difference()` on recall | ≤ 0.1 |

5. **Choose the criterion that matches harm.** Credit denial harms false negatives among creditworthy → equal opportunity (TPR). Fraud flags harm false positives → FPR parity. State why. Parity metrics can conflict; you cannot satisfy all.
6. **Decision rule:** threshold breached → no-go, or mitigate (`ThresholdOptimizer`, `ExponentiatedGradient`, reweighting, better data) and re-audit. Document residual disparity in the model card (REQUIRED SUB-SKILL: writing-model-cards-datasheets).

## Implementation

```python
import pandas as pd
from fairlearn.metrics import (
    MetricFrame,
    demographic_parity_ratio,
    equalized_odds_difference,
    false_positive_rate,
    selection_rate,
    true_positive_rate,
)
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import HistGradientBoostingClassifier


def evaluate_group_fairness(
    y_true: pd.Series, y_pred: pd.Series, sensitive: pd.DataFrame
) -> dict[str, object]:
    """Calcula métricas por grupo e disparidades para a auditoria de fairness."""
    frame = MetricFrame(
        metrics={
            "n": lambda yt, yp: len(yt),
            "selection_rate": selection_rate,
            "tpr": true_positive_rate,
            "fpr": false_positive_rate,
        },
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive,
    )
    return {
        "by_group": frame.by_group,
        "dp_ratio": demographic_parity_ratio(y_true, y_pred, sensitive_features=sensitive),
        "eo_difference": equalized_odds_difference(y_true, y_pred, sensitive_features=sensitive),
    }


def compute_proxy_auc(features: pd.DataFrame, sensitive_binary: pd.Series) -> float:
    """Mede o quanto as features do modelo conseguem prever um atributo sensível (proxy)."""
    proba = cross_val_predict(
        HistGradientBoostingClassifier(random_state=42), features, sensitive_binary,
        cv=5, method="predict_proba",
    )[:, 1]
    return float(roc_auc_score(sensitive_binary, proba))
```

## Common Mistakes

| Mistake | Fix |
|---|---|
| "Dropped sex/region, so it's fair" | Proxy check + outcome audit |
| Thresholds chosen after seeing results | Pre-register thresholds with stakeholders |
| Reporting group metrics without `n` | Always show `n` and CI; tiny groups are inconclusive |
| Only demographic parity | Pick the criterion matching the harm; report TPR/FPR gaps too |
| Audit deferred to "after launch" | Audit is a go/no-go gate |
