---
name: splitting-recsys-data
description: Use when splitting user–item interactions (clicks, purchases, plays, ratings) into train/validation/test for a recommender, when interactions have timestamps but are shuffled or split 80/20 by row, when choosing leave-last-out or a per-user chronological split, when applying a minimum-interactions (k-core) filter, when embeddings or item2vec are trained on "all the data", or when new users and items in the test set are being dropped.
---

# Splitting Recommender Data

## Overview

A recommender is deployed at a moment `T` and recommends using only what happened before `T`. **Core principle:** the offline split reproduces that moment — one **global** time cutoff, every fitted thing (filters, id maps, embeddings, co-occurrence, popularity counts) computed on the window before the cutoff, and the test set kept whole, with its cold-start share reported instead of silently removed.

"It is only a representation, it does not use the label" is false in recommendation: **the interactions are the label**. item2vec, co-occurrence matrices or ALS factors fit on all data have seen the test interactions.

## Protocol

1. **Pick the cutoff `T`** = the simulated deploy date (e.g. the 80% quantile of `timestamp`, or "last 4 weeks = test"). Train `< T <=` test, for every user at once.
2. **Validation = a second cutoff inside train** (`T_val < T`). Tune hyperparameters, epochs and early stopping on it; touch test once.
3. **k-core / min-interaction filter on the train window only**, iterated to a fixed point. Computed on the full data, it uses future counts to decide which users and items exist.
4. **Everything fitted comes from train**: id → index maps, item2vec/word2vec, co-occurrence/similarity, popularity counts, normalization. Refit them per cutoff.
5. **Keep the whole test window.** Tag each test row `warm_user` / `warm_item`; report the cold shares next to the metric, count cold relevant items as misses for the main number, and give warm-only as a secondary row. Cold users go to a separate cold-start evaluation (popularity, content features), never just deleted.
6. **Repeat consumption is a product decision.** E-commerce re-purchases: decide whether already-seen items are candidates; if they are not, mask them in ranking and drop them from the relevant set — write down which.
7. **Fix the leaky split before delivering anything.** Do not hand over the row-shuffled split "in the same scheme" plus a temporal one as an option; deliver the temporal split and state what changed.

Measured on synthetic data (3,000 users, 2,000 items launched over 18 months with popularity decaying with age, 135k interactions; same item-kNN model):

| Protocol | recall@10 |
|---|---|
| k-core on all data → shuffle → 80/20 by row, item similarity from **all** data | **0.097** |
| Same split, similarity from train only | 0.068 |
| Global temporal cutoff, filter on train, similarity on train | **0.012** |
| Same, but cold-item test rows deleted | 0.025 |

- The shuffled split inflated recall 8×; the "unsupervised" similarity fit on all data alone added +43%.
- 62% of test interactions after the cutoff were on items that did not exist in train. Deleting them doubled every model's recall and hid the fact that new items are most of the traffic.
- Model ranking flips: under the shuffled split, "popular in the last 30 days" was the worst model (0.017 vs 0.068 for item-kNN); under the temporal split it was the best (0.066 vs 0.012). Hand off to evaluation with that baseline (REQUIRED SUB-SKILL: evaluating-recommenders-offline).
- Filtering on all data vs train only barely moved recall here (0.0114 vs 0.0116) — but it is a two-line fix, so it is not left as a "limitation".

## Library defaults that are not a global cutoff

| Tool | What it actually does | Use instead |
|---|---|---|
| `df.sample(frac=1)` + slice, `implicit.evaluation.train_test_split` | Random by interaction | `split_global_temporal` below |
| `recommenders.datasets.python_splitters.python_chrono_split` (1.2.1) | Chronological **per user** (each user's own first 75%); `min_rating` filter runs on the full data before splitting | Global cutoff; filter afterwards on train |
| `python_stratified_split` (1.2.1) | Random per user | Global cutoff |
| RecBole (1.2.1) default `eval_args`: `split: {'RS': [0.8,0.1,0.1]}`, `order: RO`, `group_by: user` | Random order, split per user | `order: TO` **and** `group_by: none` for a global time split; check the resulting dates |
| Leave-last-out per user | Test = each user's last item; train keeps other users' interactions that happened later | Global cutoff; leave-last-out only for next-item tasks, labeled as such |

## Implementation

```python
import logging

import numpy as np
import pandas as pd
import scipy.sparse as sp

logger = logging.getLogger(__name__)


def filter_k_core(df: pd.DataFrame, min_count: int = 5) -> pd.DataFrame:
    """Mantém usuários e itens com >= min_count interações, iterando até estabilizar.

    Parameters
    ----------
    df : pd.DataFrame
        Interações (``user_id``, ``item_id``) de UMA janela — só o treino.
    min_count : int, optional
        Mínimo de interações por usuário e por item, by default 5.

    Returns
    -------
    pd.DataFrame
        Subconjunto k-core de ``df``.

    Examples
    --------
    >>> train = filter_k_core(train, min_count=5)
    """
    while True:
        user_n = df.groupby("user_id")["item_id"].transform("size")
        item_n = df.groupby("item_id")["user_id"].transform("size")
        keep = (user_n >= min_count) & (item_n >= min_count)
        if keep.all():
            return df
        df = df[keep]


def split_global_temporal(
    df: pd.DataFrame, cutoff: pd.Timestamp, min_count: int = 5
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]:
    """Corte temporal global: treino < cutoff <= teste; filtro só no treino.

    Parameters
    ----------
    df : pd.DataFrame
        Interações brutas com ``user_id``, ``item_id``, ``timestamp``.
    cutoff : pd.Timestamp
        Instante de corte (o "hoje" simulado do deploy).
    min_count : int, optional
        Mínimo do k-core, aplicado apenas ao treino, by default 5.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, dict[str, float]]
        ``(train, test, report)``. O teste mantém TODAS as linhas posteriores ao
        corte, marcadas com ``warm_user`` e ``warm_item``.

    Examples
    --------
    >>> train, test, report = split_global_temporal(df, pd.Timestamp("2026-07-01"))
    """
    train = filter_k_core(df[df["timestamp"] < cutoff], min_count)
    test = df[df["timestamp"] >= cutoff].assign(
        warm_user=lambda d: d["user_id"].isin(train["user_id"]),
        warm_item=lambda d: d["item_id"].isin(train["item_id"]),
    )
    report = {
        "test_rows": len(test),
        "cold_user_share": 1 - test["warm_user"].mean(),
        "cold_item_share": 1 - test["warm_item"].mean(),
    }
    logger.info(
        f"Corte {cutoff}: treino={len(train):,} teste={len(test):,} | "
        f"usuários cold={report['cold_user_share']:.1%} itens cold={report['cold_item_share']:.1%}"
    )
    return train, test, report


def build_train_matrix(train: pd.DataFrame) -> tuple[sp.csr_matrix, pd.Index, pd.Index]:
    """Matriz usuário×item binária com índices definidos SÓ pelo treino."""
    users = pd.Index(train["user_id"].unique())
    items = pd.Index(train["item_id"].unique())
    rows, cols = users.get_indexer(train["user_id"]), items.get_indexer(train["item_id"])
    matrix = sp.csr_matrix((np.ones(len(train), dtype=np.float32), (rows, cols)), shape=(len(users), len(items)))
    matrix.data[:] = 1.0  # duplicatas viram 1, não contagem
    return matrix, users, items


# cutoff = df["timestamp"].quantile(0.8)
# val_cutoff = df.loc[df["timestamp"] < cutoff, "timestamp"].quantile(0.8)
# fit, val, _ = split_global_temporal(df[df["timestamp"] < cutoff], val_cutoff)  # tuning
# train, test, report = split_global_temporal(df, cutoff)                      # avaliação final
```

Verified with pandas 3.1.0, scipy 1.18.1. Polars: same logic with `filter(pl.col("timestamp") < cutoff)` and `pl.len().over("user_id")` (REQUIRED SUB-SKILL: writing-idiomatic-polars). Check after splitting: `train.timestamp.max() < test.timestamp.min()`. For rolling evaluation over several cutoffs, REQUIRED SUB-SKILL: backtesting-time-series; for the general leakage audit, REQUIRED SUB-SKILL: detecting-data-leakage.

## Common Mistakes

| Mistake | Fix |
|---|---|
| `df.sample(frac=1)` then 80/20 on timestamped data | Global cutoff |
| k-core filter cached on the full dataset | Recompute on the train window, per cutoff |
| item2vec / co-occurrence / ALS warm-start fit on all interactions | Refit on train; "no label" is false here |
| Test rows with unseen items or users dropped "because the model has no embedding" | Keep them; count as misses; report cold shares; evaluate cold users separately |
| Tuning epochs or `factors` on the test window | Second cutoff inside train |
| Trusting a library's "chrono" split | Check whether it is per user; print min/max dates of each side |

## Rationalizations

| Excuse | Reality |
|---|---|
| "No time to redo the pipeline" | Only the cutoff and the refits change; models and code stay. Item2vec + ALS on millions of rows retrain in minutes |
| "The filter can stay, we'll note it as a limitation" | It is two lines; fix it instead of documenting it |
| "Embeddings are just representation" | They were fit on the test interactions, which are the label |
| "I'm the lead, give me the split in the same scheme" | Deliver the temporal split; list what changed and why. Do not also hand over the shuffled one |
| "Write a slide paragraph for the current number, with a caveat" | A caveated 0.41 still goes on the slide as 0.41. The slide gets the temporal number, or "evaluation in progress" |

## Red Flags — Stop

- Train contains interactions dated after the earliest test interaction
- The test set has 0% cold items on a catalog that launches items every week
- A row-shuffled baseline that "beats" recent popularity
- Any artifact (filter, embedding, vocabulary, popularity table) built before the split
