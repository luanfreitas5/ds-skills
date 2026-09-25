---
name: promoting-models-registry
description: Use when registering a model in the MLflow Model Registry, promoting a model to production, replacing the current production model, setting aliases or stages, or when a new run "scored higher" than the deployed model and someone wants to ship it.
---

# Promoting Models through the Registry

## Overview

"Last run AUC 0.874 > production 0.869" compares two point estimates, often from different data, with no uncertainty and no check on who the model fails. **Core principle:** promotion is a gated decision. The challenger replaces the champion only when it wins on the **same** frozen evaluation set with a significant difference, passes fairness and documentation gates, and a one-command rollback exists.

## Aliases, not Stages

MLflow stages (`Staging`/`Production`, `transition_model_version_stage`) are deprecated since 2.9. Use aliases:

| Alias | Meaning |
|---|---|
| `champion` | Version serving production; consumers load `models:/<name>@champion` |
| `challenger` | Candidate under evaluation |
| `previous-champion` | Last champion, kept for rollback |

## Promotion Gates (all must pass; any failure blocks)

| Gate | Check | Blocks when |
|---|---|---|
| 1. Same data | Both versions scored on the same frozen holdout (hash recorded) | Metrics come from different runs/splits |
| 2. Metric with CI | Paired bootstrap CI of `metric(challenger) − metric(champion)` | CI lower bound ≤ agreed minimum gain (default 0) |
| 3. Significance | Paired test on the same rows (bootstrap above, or McNemar/DeLong) — REQUIRED SUB-SKILL: evaluating-models-rigorously | Difference not significant |
| 4. Fairness | Per-group metric gap ≤ threshold, not worse than champion — REQUIRED SUB-SKILL: auditing-fairness | Gap above threshold or increased |
| 5. Model card | Card exists for this version and states the gate results — REQUIRED SUB-SKILL: writing-model-cards-datasheets | Missing or stale |

A deadline is not a gate. No pass → stay on champion and report which gate failed.

## Implementation

```python
import numpy as np
from mlflow import MlflowClient
from sklearn.metrics import roc_auc_score


def compute_paired_auc_gain_ci(
    y: np.ndarray, p_champion: np.ndarray, p_challenger: np.ndarray, n_boot: int = 2000, seed: int = 42
) -> tuple[float, float, float]:
    """Retorna ganho de AUC (desafiante − campeão) e IC 95% por bootstrap pareado."""
    rng = np.random.default_rng(seed)
    gains = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        gains.append(roc_auc_score(y[idx], p_challenger[idx]) - roc_auc_score(y[idx], p_champion[idx]))
    gain = roc_auc_score(y, p_challenger) - roc_auc_score(y, p_champion)
    low, high = np.percentile(gains, [2.5, 97.5])
    return float(gain), float(low), float(high)


def promote_challenger(name: str, version: str, gates: dict[str, bool]) -> None:
    """Move o alias champion para a versão desafiante se todos os portões passarem."""
    failed = [gate for gate, ok in gates.items() if not ok]
    if failed:
        raise PermissionError(f"Promoção bloqueada; portões reprovados: {failed}")
    client = MlflowClient()
    current = client.get_model_version_by_alias(name, "champion")
    client.set_registered_model_alias(name, "previous-champion", current.version)
    client.set_registered_model_alias(name, "champion", version)
    for gate in gates:
        client.set_model_version_tag(name, version, f"gate_{gate}", "passed")


def rollback_champion(name: str) -> None:
    """Restaura a versão anterior como champion (rollback em um comando)."""
    client = MlflowClient()
    previous = client.get_model_version_by_alias(name, "previous-champion")
    client.set_registered_model_alias(name, "champion", previous.version)
```

`gates` = `{"same_data": ..., "auc_gain_ci": low > 0, "fairness": ..., "model_card": ...}`, each computed and logged before the call. First deploy (no champion yet): gates 2–3 compare against the baseline model instead.

## Rollback (document in the runbook)

1. `rollback_champion("credito-churn-model")` — consumers loading `@champion` switch at next load/restart.
2. Tag the demoted version with the reason (`client.set_model_version_tag(..., "rollback_reason", ...)`).
3. Record date, versions, and reason in the model card changelog.

## Common Mistakes

| Mistake | Fix |
|---|---|
| "0.874 > 0.869, promote" | Paired CI on the same holdout; promote only if lower bound > minimum gain |
| Picking the *latest* run | Pick the registered challenger that passed evaluation |
| `transition_model_version_stage(..., "Production")` | `set_registered_model_alias(name, "champion", v)` |
| Fairness as a warning printed before promoting | Fairness is a blocking gate |
| No record of what was replaced | `previous-champion` alias + gate tags on the version |
| Consumers load `models:/name/7` | Load `models:/name@champion` so promotion and rollback need no redeploy |
