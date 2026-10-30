---
name: building-recsys-baselines
description: Use when starting a recommender project, when someone says "we'll go straight to two-tower / deep learning / transformers" or wants to compare only against the previous version of their own model, or when building popularity, co-occurrence, item-kNN, ALS (implicit), LightFM or Surprise baselines and ALS scores below popularity.
---

# Building Recsys Baselines

## Overview

A recommender baseline ladder is cheap (an afternoon, minutes of compute) and is what makes any neural result interpretable. **Core principle:** build the whole ladder on the same temporal split before the first deep model, and make each rung a *fair* one — an untuned or mis-oriented baseline is a straw man that inflates the lift.

The previous version of your own model is a comparison, not a baseline. A deadline, sunk pipeline work, or "the lead already decided" change the order of work, not the ladder.

Split rules: REQUIRED SUB-SKILL: splitting-recsys-data. Metrics, CIs, full ranking, paired bootstrap: REQUIRED SUB-SKILL: evaluating-recommenders-offline. General ladder and reporting table: REQUIRED SUB-SKILL: establishing-baselines.

## The Ladder

| Rung | Model | Build notes |
|---|---|---|
| 0 | Popularity (all train) | `bincount` of items; mask seen items |
| 1 | **Popularity, last N days before cutoff** (try 7/30/90) | Usually the strongest on catalogs that churn; choose N on the validation cutoff |
| 2 | Item-kNN (cosine, top-k neighbors, shrinkage) | Co-occurrence normalized by item norms; raw co-occurrence only favors the head |
| 3 | ALS (`implicit`), BM25-weighted | Tune `factors`, `regularization`, iterations on validation; same budget as the neural model |
| 4 (optional) | LightFM (WARP, with item features) / Surprise | See Library Gotchas — only when they fit the data |

Measured on synthetic data (3,000 users, 2,000 items launched over 18 months, global temporal cutoff, full ranking, recall@10 ± 95% CI, ~52% of test interactions on cold items counted as misses):

| Model | recall@10 |
|---|---|
| Popularity (all train) | 0.029 ± 0.003 |
| **Popularity, last 7 days** | **0.071 ± 0.004** |
| Popularity, last 30 / 90 days | 0.065 / 0.045 |
| Item-kNN | 0.029 ± 0.003 |
| ALS 64 factors, BM25 weights | 0.013 ± 0.002 |
| ALS 64 factors, raw 0/1 matrix | 0.003 ± 0.001 |

Take-aways: the window changes the baseline 2.5×; unweighted ALS is ~5× worse than BM25-weighted (an "ALS baseline" that loses to popularity is usually this, not a verdict on ALS); "+X% over last quarter's model" says nothing until the 0.071 row exists. Absolute values depend on the data; the *ordering checks* are what transfer.

## Implementation

```python
import numpy as np
import pandas as pd
import scipy.sparse as sp


def build_interaction_matrix(train: pd.DataFrame, users: pd.Index, items: pd.Index) -> sp.csr_matrix:
    """Matriz usuário×item binária (CSR) com índices definidos só pelo treino."""
    rows, cols = users.get_indexer(train["user_id"]), items.get_indexer(train["item_id"])
    matrix = sp.csr_matrix((np.ones(len(train), dtype=np.float32), (rows, cols)), shape=(len(users), len(items)))
    matrix.data[:] = 1.0  # duplicatas viram 1
    return matrix


def score_popularity(train, items, n_users, cutoff=None, window_days=None) -> np.ndarray:
    """Popularidade global ou dos últimos ``window_days`` antes do corte."""
    if window_days is not None:
        train = train[train["timestamp"] >= cutoff - pd.Timedelta(days=window_days)]
    counts = np.bincount(items.get_indexer(train["item_id"]), minlength=len(items)).astype(np.float64)
    return np.broadcast_to(counts, (n_users, len(items)))


def fit_item_knn(matrix: sp.csr_matrix, n_neighbors: int = 50, shrinkage: float = 10.0) -> sp.csr_matrix:
    """Similaridade item-item: cosseno × co/(co+shrinkage), top-k vizinhos por item."""
    cooc = (matrix.T @ matrix).tocsr()
    cooc.setdiag(0)
    cooc.eliminate_zeros()
    norm = np.sqrt(np.asarray(matrix.sum(axis=0)).ravel())
    coo = cooc.tocoo()
    vals = coo.data / (norm[coo.row] * norm[coo.col]) * coo.data / (coo.data + shrinkage)
    sim = sp.csr_matrix((vals, (coo.row, coo.col)), shape=cooc.shape)
    rows, cols, data = [], [], []
    for i in range(sim.shape[0]):
        lo, hi = sim.indptr[i], sim.indptr[i + 1]
        top = np.argsort(-sim.data[lo:hi])[:n_neighbors]
        rows.append(np.full(len(top), i)); cols.append(sim.indices[lo:hi][top]); data.append(sim.data[lo:hi][top])
    return sp.csr_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))), shape=sim.shape)


def fit_als(matrix: sp.csr_matrix, factors: int = 64, regularization: float = 0.05,
            iterations: int = 15, seed: int = 42):
    """ALS (implicit >= 0.7) sobre matriz usuário×item com pesos BM25."""
    from implicit.als import AlternatingLeastSquares
    from implicit.nearest_neighbours import bm25_weight

    model = AlternatingLeastSquares(factors=factors, regularization=regularization,
                                    iterations=iterations, random_state=seed)
    model.fit(bm25_weight(matrix, K1=100, B=0.8).tocsr(), show_progress=False)
    return model

# Escores por usuário avaliado (uidx = índices dos usuários no mapa do treino):
#   itemKNN: (matrix[uidx] @ sim).toarray()     ALS: als.user_factors[uidx] @ als.item_factors.T
# Mascare itens vistos e avalie com UM avaliador para todos (evaluating-recommenders-offline).
```

Verified with implicit 0.7.3, numpy 2.5.4, scipy 1.18.1, pandas 3.1.0, Python 3.12 (Windows, CPU). Set `OPENBLAS_NUM_THREADS=1` (or `threadpoolctl.threadpool_limits(1, "blas")`) before ALS, or `implicit` warns and can slow down severely. ALS/kNN scores are dense `(n_users, n_items)`; batch users when it does not fit in memory.

## Library Gotchas

| Library | Gotcha |
|---|---|
| implicit >= 0.5 | `fit()` takes **user×item** CSR. Old tutorials pass item×user; the model then trains transposed and `recommend` returns nonsense without an error |
| implicit | `ranking_metrics_at_k` "precision" is micro-recall and `train_test_split` is random by row — use the shared evaluator and the temporal split |
| implicit | ALS on a raw 0/1 matrix is a weak baseline (table above); weight (`bm25_weight`) and tune before concluding "ALS loses" |
| Surprise 1.1.5 | Built for **explicit ratings** (RMSE/MAE). On clicks/purchases it predicts a rating, not a ranking, and has no top-k over the catalog; use it only with real ratings and evaluate ranking yourself. Installs from a wheel |
| LightFM | WARP/BPR loss and item/user features give a legitimate rung when side information exists. **Not verified here:** `pip install lightfm` failed to build on Windows/Python 3.12 (no wheel, needs a C compiler) and the project is unmaintained — pin the version, use a Linux container, or substitute implicit BPR/ALS |

## Rationalizations

| Excuse | Reality |
|---|---|
| "Boss chose two-tower, baselines are old-school" | The baselines are the proof the two-tower is worth serving; they run in minutes on an existing split |
| "Demo is Friday, no time" | Popularity + last-N-days is ~10 lines; item-kNN and ALS are minutes. Build the ladder first, spend the remaining days on the neural model |
| "Three weeks went into the pipeline" | The ladder reuses that pipeline; sunk cost is not an argument about the comparison |
| "Compare to last quarter's model, it's the same task" | A lift over your own model is not a baseline comparison; ladder rows come first, previous model is one more row |
| "Popularity is a toy" | It was the best model in the table above; if it wins, the neural result is not yet a result |
| "ALS lost to popularity, so ALS is bad" | Check weighting, tuning and matrix orientation first |

## Red Flags — Stop

- Only a global (all-time) popularity row, or no recent-popularity row
- A single "% improvement" on the slide with no baseline rung and no CI
- The ladder evaluated on a different split or candidate set than the new model
- ALS below popularity with default, unweighted, untuned settings
- Baselines left untuned while the neural model got a tuning budget
