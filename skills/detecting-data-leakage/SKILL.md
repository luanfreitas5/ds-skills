---
name: detecting-data-leakage
description: Use when a model scores suspiciously high (e.g. AUC > 0.95 on a hard business problem), when features come from a snapshot taken after the label period, when preprocessing (imputation, scaling, encoding, SMOTE, feature selection) runs before the train/test split, when data has time order or repeated entities, or before tuning or presenting any model.
---

# Detecting Data Leakage

## Overview

Leakage = the model sees information at training time that it will not have at prediction time. **Core principle:** every feature must be computable *as of* the prediction moment, and every fitted transformation must be fit *inside* the training fold only.

Suspiciously good results are a bug report, not a win. Do not tune a leaky model; fix the leak first.

## Leakage Audit (run before tuning or reporting)

1. **Define the prediction moment `t0`** per row. Features use data `< t0`; label uses data `>= t0`.
2. **Point-in-time features.** Snapshot "today" joined to a past label = leakage (e.g. `days_since_last_login` of a customer who already churned). Rebuild with an as-of join.
3. **Split matches deployment.** Time-ordered → temporal split (`TimeSeriesSplit`, or cut by date). Repeated entities (customer, patient, store) → `GroupKFold` / `StratifiedGroupKFold`. Random split only for i.i.d. rows.
4. **All fitted steps inside a `Pipeline`.** Imputer, scaler, encoder, SMOTE (`imblearn.pipeline.Pipeline`), feature selection, PCA — never `fit` on the full dataframe.
5. **Target encoding** uses cross-fitting: `sklearn.preprocessing.TargetEncoder` (cross-fits inside `fit_transform`). `category_encoders.TargetEncoder` does not cross-fit — its train encodings leak the row's own label.
6. **Group-wise rolling / lag features** shift *within* the group. `df.groupby("store")["y"].shift(1).rolling(3).mean()` rolls **across** store boundaries.
7. **Run detectors** (below). Any hit → investigate before continuing.

## Detectors

| Detector | How | Leak signal |
|---|---|---|
| Single-feature AUC | Fit a depth-2 tree on each feature alone | One feature alone gives AUC > 0.9 |
| Drop-column test | Retrain without the top feature | Score collapses from 0.97 to ~0.7 |
| Adversarial validation | Classifier to predict "train vs test" row | AUC > 0.7 → train/test differ (shift or split bug) |
| Timestamp check | `max(feature_timestamp) < t0` per row | Any violation |
| Duplicate check | Same entity/row in train and test | Non-empty intersection |

## Implementation

```python
import polars as pl


def build_point_in_time_features(
    events: pl.DataFrame, labels: pl.DataFrame
) -> pl.DataFrame:
    """Constrói features usando apenas eventos anteriores ao instante de predição t0.

    Parameters
    ----------
    events : pl.DataFrame
        Eventos com colunas ``customer_id``, ``event_ts`` e ``amount``.
    labels : pl.DataFrame
        Rótulos com ``customer_id``, ``t0`` (instante da predição) e ``target``.
    """
    joined = labels.join(events, on="customer_id", how="left").filter(
        pl.col("event_ts") < pl.col("t0")  # nunca usar informação do futuro
    )
    return joined.group_by("customer_id", "t0", "target").agg(
        pl.len().alias("n_events_before_t0"),
        pl.col("amount").sum().alias("total_amount_before_t0"),
        (pl.col("t0").first() - pl.col("event_ts").max()).dt.total_days().alias("days_since_last_event"),
    )


# Lag/rolling correto por grupo (pandas):
# df["roll_mean_3"] = df.groupby("store_id")["sales"].transform(lambda s: s.shift(1).rolling(3).mean())
```

Note: the inner join + filter above drops customers with zero past events; left-join those back with zero-filled counts.

## Common Mistakes

| Mistake | Fix |
|---|---|
| `train_test_split` random on time-ordered data | Split by date; test = most recent period |
| Imputer/encoder fit before split | Put in `Pipeline`; CV fits per fold |
| Same customer in train and test | `GroupKFold(groups=customer_id)` |
| Features built from "current" table | As-of join on `t0` |
| Hyperparameter tuning on the test set | Nested CV or separate validation split |
| Oversampling before split | `imblearn` Pipeline so SMOTE only touches train folds |

## Red Flags — Stop

- "AUC 0.97, awesome — let's tune it"
- "I'll fix the pipeline order later, just add GridSearch"
- Feature importance dominated by one "status"/"last activity" column
