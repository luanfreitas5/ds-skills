---
name: tuning-hyperparameters-optuna
description: Use when tuning hyperparameters with Optuna, GridSearchCV, RandomizedSearchCV, or by hand, when a search picks parameters by test-set score, when the best search score is about to be reported as expected production performance, or when a long search must survive crashes or reboots.
---

# Tuning Hyperparameters with Optuna

## Overview

The best score of a search is the maximum of many noisy estimates: it is optimistically biased by construction. A test set that chose the parameters is no longer a test set. **Core principle:** the search sees only training data (CV inside it); the reported number comes from data the search never touched — an untouched holdout evaluated once, or nested CV.

## Protocol

1. **Split first.** Test/holdout is set aside before the study exists (time-ordered or grouped if the data is — REQUIRED SUB-SKILL: detecting-data-leakage).
2. **Objective = CV score on the training set only.** Preprocessing inside a `Pipeline` so each fold fits its own transforms.
3. **Final estimate:** refit best params on the full training set → evaluate **once** on the holdout, with a CI. No holdout (small data) → nested CV: the whole search runs inside each outer fold; report the outer-fold scores.
4. **Report both numbers, labeled:** `study.best_value` = "search score (optimistic, selection only)"; holdout/nested score = "expected performance".

## Study Setup

| Setting | Choice | Why |
|---|---|---|
| Sampler | `TPESampler(seed=42)` | Reproducible suggestions (exact only with `n_jobs=1`) |
| Pruner | `MedianPruner(n_startup_trials=10, n_warmup_steps=1)` + `trial.report(score, fold)` | Stops bad trials after a few folds |
| Storage | `RDBStorage("sqlite:///studies/<name>.db", heartbeat_interval=60, grace_period=120)` + `load_if_exists=True` | Resume after crash; trials stuck in `RUNNING` are marked `FAIL` instead of lingering |
| Retry | `failed_trial_callback=RetryFailedTrialCallback(max_retry=1)` | Trial killed by the reboot is re-queued |
| Trial budget | Subtract finished (`COMPLETE` + `PRUNED`) trials when resuming | Re-running `optimize(n_trials=200)` adds 200 more |
| Search space | Justify each range; `log=True` for scale parameters (learning rate, regularization) | Uniform over `[1e-4, 1e-1]` spends 90% of trials above `1e-2` |
| Tracking | Parent MLflow run = study; one nested child per trial (REQUIRED SUB-SKILL: tracking-experiments-mlflow) | Every trial's params and CV score are auditable |

## Implementation

```python
import mlflow
import numpy as np
import optuna
from optuna.storages import RDBStorage, RetryFailedTrialCallback
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold


def run_study(X_train: np.ndarray, y_train: np.ndarray, n_trials: int, seed: int = 42) -> optuna.Study:
    """Otimiza PR-AUC por CV no treino; o conjunto de teste nunca entra aqui."""
    storage = RDBStorage(
        "sqlite:///studies/fraude.db", heartbeat_interval=60, grace_period=120,
        failed_trial_callback=RetryFailedTrialCallback(max_retry=1),
    )
    study = optuna.create_study(
        study_name="fraude-hgb", storage=storage, load_if_exists=True, direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1),
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 8, 128, log=True),
            "l2_regularization": trial.suggest_float("l2_regularization", 1e-6, 10.0, log=True),
        }
        scores = []
        with mlflow.start_run(run_name=f"trial-{trial.number}", nested=True):
            mlflow.log_params(params)
            for fold, (tr, va) in enumerate(skf.split(X_train, y_train)):
                model = HistGradientBoostingClassifier(**params, random_state=seed).fit(X_train[tr], y_train[tr])
                scores.append(average_precision_score(y_train[va], model.predict_proba(X_train[va])[:, 1]))
                trial.report(float(np.mean(scores)), fold)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            mlflow.log_metric("cv_pr_auc", float(np.mean(scores)))
        return float(np.mean(scores))

    finished = (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
    done = len(study.get_trials(states=finished))
    with mlflow.start_run(run_name="study-fraude-hgb", tags={"seed": str(seed)}):
        study.optimize(objective, n_trials=max(0, n_trials - done))
        mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
        mlflow.log_metric("search_best_cv_pr_auc", study.best_value)  # otimista: só para seleção
    return study
```

After the study: refit `HistGradientBoostingClassifier(**study.best_params)` on all of `X_train`, score `X_test` once, and report that PR-AUC with a bootstrap CI (REQUIRED SUB-SKILL: evaluating-models-rigorously).

## Common Mistakes

| Mistake | Fix |
|---|---|
| Objective scores on `X_test` | CV on train inside the objective |
| Reporting `study.best_value` as expected performance | Holdout once, or nested CV; label `best_value` as optimistic |
| Early stopping on the fold that is also scored | Separate inner validation slice for early stopping, or accept and state the small bias |
| In-memory study for an overnight run | SQLite `RDBStorage` + heartbeat + `load_if_exists=True` |
| Tweaking the space after looking at test results | Space fixed before the holdout is touched |
| No sampler seed | `TPESampler(seed=...)` |
