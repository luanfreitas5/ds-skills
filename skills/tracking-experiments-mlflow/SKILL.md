---
name: tracking-experiments-mlflow
description: Use when adding MLflow (or any experiment tracking) to a training, cross-validation, tuning, or model-comparison script, when runs in the MLflow UI cannot be traced back to code and data, or when enabling mlflow autolog.
---

# Tracking Experiments with MLflow

## Overview

A run that logs only the final metric answers "what score?" but not "from which code, which data, which seed?". **Core principle:** every run is a reproducible record: lineage tags + params + per-fold metrics + a model with a signature, fitted on the data the run describes.

## Run Contract

| Element | How | Why |
|---|---|---|
| Tags `git_sha`, `data_hash`, `seed` | `mlflow.start_run(tags=...)` on parent **and** children | Filter/compare runs by code and data version |
| Refuse dirty tree | `git status --porcelain` non-empty → raise | SHA of a dirty tree lies |
| Params | `model.get_params()` + CV setup (`n_splits`, split strategy) | Diff runs in the UI |
| CV structure | One **parent** run per model, one **nested** child per fold | Fold metrics stay inspectable, UI compares parents |
| Parent metrics | `cv_<metric>_mean`, `cv_<metric>_std` | Uncertainty, not a point (REQUIRED SUB-SKILL: evaluating-models-rigorously) |
| Model | Refit on all training data, `log_model(..., signature=, input_example=)` | Last-fold model saw 80% of data; signature enforces input schema at serving |
| Tuning | Parent = study, child per trial (`nested=True`) | Same pattern as CV |

`data_hash` should match the dataset manifest (REQUIRED SUB-SKILL: ensuring-reproducibility).

## Implementation

```python
import hashlib
import subprocess
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.base import ClassifierMixin, clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def compute_git_sha() -> str:
    """Retorna o SHA do HEAD; levanta erro se houver alterações não commitadas."""
    if subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, check=True).stdout.strip():
        raise RuntimeError("Árvore Git suja: faça commit antes de registrar o experimento.")
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()


def compute_file_hash(path: Path) -> str:
    """Calcula o SHA-256 do arquivo em blocos de 1 MiB."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_cv_experiment(
    name: str, model: ClassifierMixin, X: pd.DataFrame, y: pd.Series, data_path: Path, seed: int = 42
) -> float:
    """Executa CV estratificada com um run pai por modelo e um run filho por fold."""
    tags = {"git_sha": compute_git_sha(), "data_hash": compute_file_hash(data_path), "seed": str(seed)}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    with mlflow.start_run(run_name=name, tags=tags):
        mlflow.log_params({**model.get_params(), "cv": "StratifiedKFold", "n_splits": 5})
        scores = []
        for fold, (tr, va) in enumerate(skf.split(X, y)):
            with mlflow.start_run(run_name=f"{name}-fold{fold}", nested=True, tags={**tags, "fold": str(fold)}):
                fitted = clone(model).fit(X.iloc[tr], y.iloc[tr])
                auc = roc_auc_score(y.iloc[va], fitted.predict_proba(X.iloc[va])[:, 1])
                mlflow.log_metric("val_auc", auc)
                scores.append(auc)
        mlflow.log_metrics({"cv_auc_mean": float(np.mean(scores)), "cv_auc_std": float(np.std(scores, ddof=1))})
        final = clone(model).fit(X, y)  # modelo registrado vê todo o treino
        signature = infer_signature(X.head(100), final.predict_proba(X.head(100)))
        mlflow.sklearn.log_model(final, name="model", signature=signature, input_example=X.head(5))
    return float(np.mean(scores))
```

MLflow ≥ 3 uses `name=`; on 2.x use `artifact_path=`. Recent MLflow 3 releases serialize sklearn models with `skops` by default and refuse tree models (`RandomForest`, `GradientBoosting`, ...) with "untrusted types: sklearn.tree._tree.Tree"; for models you trained yourself pass `skops_trusted_types=["sklearn.tree._tree.Tree"]`. XGBoost/LightGBM estimators: use their own flavor (`mlflow.xgboost.log_model`, `mlflow.lightgbm.log_model`), not `mlflow.sklearn`.

## Autolog Caveats

- `mlflow.sklearn.autolog()` logs `training_*` metrics computed **on training data**. Never report them; log validation metrics yourself.
- Autolog inside a CV loop without an active parent creates one top-level run per `fit` and floods the experiment. Open the parent run first, or use `autolog(log_models=False)`.
- Autolog does not add `git_sha`/`data_hash`/`seed` tags from this contract; add them explicitly.

## Common Mistakes

| Mistake | Fix |
|---|---|
| Only `log_metric("auc", mean)` | Tags + params + per-fold child runs + mean/std |
| Folds as `step=` in one run | Nested child run per fold (steps are for training iterations) |
| Logging the last-fold model | Refit on all training data, then log |
| `log_model` without signature | `infer_signature` + `input_example` |
| Same estimator object refit across folds | `clone(model)` per fold |
| Uncommitted changes, run logged anyway | Raise on dirty tree |
