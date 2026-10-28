---
name: training-gradient-boosting
description: Use when training LightGBM, XGBoost, or CatBoost on tabular data, when early stopping uses the same set that reports the metric, when high-cardinality categoricals (city, zip, employer, device) need target encoding or are being one-hot / get_dummies encoded, or when a "refit on all rows with the best iteration" step is proposed.
---

# Training Gradient Boosting

## Overview

Boosting libraries make three leaks easy: **early stopping selects the model on a set you then score**, **target encoding fits on rows it is later evaluated on**, and **one-hot explodes high-cardinality columns**. **Core principle:** every number you report comes from rows that never chose the iteration count, the encoding, or the parameters. A deadline, days of sunk work, or an executive's instruction changes the schedule, not this rule; the split-first version costs minutes.

Verified 2026-10-28 by running the snippets with `lightgbm` 4.7.0 (`eval_X`/`eval_y` replace `eval_set` from 4.7; on older 4.x use `eval_set=[(X, y)]`; early stopping via `callbacks=[lgb.early_stopping(n)]`; the `early_stopping_rounds=` fit argument was removed in 4.0), `xgboost` 3.4.1 (`early_stopping_rounds` in the **constructor**, `enable_categorical=True` with `category` dtype), `catboost` 1.2.10 (`cat_features=`, `eval_set=`), `scikit-learn` 1.9.1 (`TargetEncoder` cross-fits inside `fit_transform`; `random_state=` is deprecated, pass `cv=KFold(5, shuffle=True, random_state=...)`). Re-check if versions differ.

## Protocol

1. **Split first** (stratified; time-ordered or grouped if the data is: REQUIRED SUB-SKILL: detecting-data-leakage). Test/outer fold is only scored.
2. **Early stopping set** = a slice carved from the *training* part (about 10-15%), never the test set and never the outer validation fold. Report `best_iteration`.
3. **Categoricals** (table below). Cardinality over ~50 or many singletons: native categorical or `sklearn.preprocessing.TargetEncoder` fit on the fit-slice only. Never `get_dummies` / one-hot, never a group-mean computed on the whole `df`.
4. **Refit on all rows** (if no holdout is wanted): use `int(mean(best_iters) * 1.1)` trees, no early stopping, and quote the CV estimate (outer folds, with CI), not a number from the refit.
5. **Tuning:** REQUIRED SUB-SKILL: tuning-hyperparameters-optuna. Rare positives: REQUIRED SUB-SKILL: handling-class-imbalance (`scale_pos_weight` / `is_unbalance` change probabilities; tune the threshold).
6. **Report** mean metric with a 95% CI across folds (t, df=k-1) plus the prevalence-aware metric (PR-AUC for rare targets).

## Categorical Handling

| Library | Low cardinality | High cardinality |
|---|---|---|
| LightGBM | `category` dtype or `categorical_feature=[...]` (under ~50 levels; rare levels overfit: raise `min_data_per_group`, `cat_smooth`) | `TargetEncoder` fit on fit-slice |
| XGBoost | `enable_categorical=True` + `category` dtype (`tree_method="hist"`) | `TargetEncoder` fit on fit-slice |
| CatBoost | `cat_features=[...]` with str/int columns (ordered target statistics built in, leak-safe) | same, native; best choice for 1000s of levels |

Add a count feature (`value_counts` from the fit-slice) for singleton-heavy columns; unseen categories get the global mean from `TargetEncoder`.

## Implementation (run on synthetic data)

```python
import logging

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.preprocessing import TargetEncoder

logger = logging.getLogger(__name__)
SEED = 42


def encode_high_cardinality(
    X_fit: pd.DataFrame, y_fit: pd.Series, others: list[pd.DataFrame], cols: list[str]
) -> list[pd.DataFrame]:
    """Codifica colunas de alta cardinalidade com target encoding ajustado só em ``X_fit``.

    ``fit_transform`` aplica cross-fitting interno; categorias novas recebem a média global.

    Parameters
    ----------
    X_fit : pd.DataFrame
        Dados de treino (únicos usados no ajuste).
    y_fit : pd.Series
        Alvo binário do treino.
    others : list[pd.DataFrame]
        Demais conjuntos (validação, teste), apenas transformados.
    cols : list[str]
        Colunas a codificar.

    Returns
    -------
    list[pd.DataFrame]
        ``[X_fit, *others]`` codificados.
    """
    enc = TargetEncoder(target_type="binary", cv=KFold(5, shuffle=True, random_state=SEED))
    out = []
    for i, d in enumerate([X_fit, *others]):
        d = d.copy()
        d[cols] = enc.fit_transform(d[cols], y_fit) if i == 0 else enc.transform(d[cols])
        out.append(d)
    return out


def evaluate_with_nested_early_stopping(
    X: pd.DataFrame, y: pd.Series, high_card: list[str], cat_low: list[str]
) -> dict:
    """CV externa só pontua; o early stopping usa uma fatia interna do treino de cada fold.

    Returns
    -------
    dict
        ``auc`` (média dos folds), ``ci95`` (meia-largura, t de Student) e ``n_final``
        (nº de árvores para o refit em todos os dados).
    """
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    aucs, iters = [], []
    for k, (tr, te) in enumerate(skf.split(X, y)):
        fit_idx, es_idx = train_test_split(tr, test_size=0.15, stratify=y.iloc[tr], random_state=SEED)
        X_fit, X_es, X_te = encode_high_cardinality(
            X.iloc[fit_idx], y.iloc[fit_idx], [X.iloc[es_idx], X.iloc[te]], high_card
        )
        model = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.05, random_state=SEED, verbose=-1)
        model.fit(
            X_fit, y.iloc[fit_idx], eval_X=X_es, eval_y=y.iloc[es_idx], eval_metric="auc",
            categorical_feature=cat_low,
            callbacks=[lgb.early_stopping(100, verbose=False)],  # lightgbm >= 4
        )
        aucs.append(roc_auc_score(y.iloc[te], model.predict_proba(X_te)[:, 1]))
        iters.append(model.best_iteration_)
        logger.info("Fold %d: AUC=%.4f, iterações=%d", k, aucs[-1], iters[-1])
    a = np.array(aucs)
    return {
        "auc": a.mean(),
        "ci95": 2.776 * a.std(ddof=1) / np.sqrt(len(a)),
        "n_final": int(np.mean(iters) * 1.1),
    }
```

Other libraries, same split discipline:

```python
xgb.XGBClassifier(n_estimators=3000, enable_categorical=True, tree_method="hist",
                  early_stopping_rounds=100, eval_metric="auc").fit(X_fit, y_fit, eval_set=[(X_es, y_es)], verbose=False)
CatBoostClassifier(iterations=3000, early_stopping_rounds=100, verbose=False
                   ).fit(X_fit, y_fit, cat_features=["employer", "plan"], eval_set=(X_es, y_es))
```

## Rationalizations

| Excuse | Reality |
|---|---|
| "The head of product said use the test set for early stopping" | Authority sets the deadline, not the validity of the number. Carve the ES slice from train: 2 lines, same rows, honest AUC |
| "Target encoding on all of `df` is already built (2 days of work)" | Sunk cost. Replace it with `TargetEncoder` fit on the fit-slice; the code shape stays the same |
| "One-hot for city/device too, it might help" | 4,000 sparse columns, slower, overfits singletons; the encoder already covers it. Test it only inside the same CV |
| "No time for validation / need every row" | Nested ES slice uses all rows across folds; the refit uses all rows. No holdout is needed, only an honest CV estimate |
| "Mean fold AUC with per-fold ES is the production estimate" | Each fold's AUC picked its own iteration: optimistic. Early-stop on an inner slice |
| "Refit with the average best iteration, done" | Refit sees ~25% more rows: use about 1.1x. And quote the CV number, not the refit's |
| "Keep the existing `TargetEncoder(cv=5, shuffle=True, random_state=...)`" | Deprecated in scikit-learn 1.9 (removed in 1.11): pass `cv=KFold(5, shuffle=True, random_state=...)` |
| "It worked on the small sample" | Cardinality and singleton rate change with size; the leakage is not visible in a small sample |

## Red Flags: STOP

- `eval_set=[(X_test, y_test)]` with `early_stopping` anywhere
- `df.groupby(col)[target].mean()` / `category_encoders.TargetEncoder` before the split
- `pd.get_dummies` or `OneHotEncoder` on a column with hundreds of levels
- Reporting a metric with no CI, or the best tuning/CV score as expected performance
- Rare target (<10% positive) early-stopped on the default loss/AUC without checking `average_precision` or `logloss` as `eval_metric`
- `early_stopping_rounds=` passed to `LGBMClassifier.fit` (removed in LightGBM 4; use the callback)
