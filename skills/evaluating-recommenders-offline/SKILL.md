---
name: evaluating-recommenders-offline
description: Use when reporting NDCG@k, MAP@k, recall@k, hit rate or MRR for a recommender, when comparing a new recommender (LightGCN, ALS, BPR, two-tower, SASRec) against the current one, when a lift like "22% better" goes on a slide, when evaluating with RecBole, implicit or Microsoft recommenders, when the test set ranks only items seen in training or sampled negatives, or when there is no popularity baseline.
---

# Evaluating Recommenders Offline

## Overview

An offline recommender number is only meaningful relative to (a) a split that mimics deployment and (b) a baseline a product manager could build in an afternoon. **Core principle:** the protocol decides the winner. Shuffled splits, sampled negatives and deleted cold items do not just inflate numbers — they reorder models. Fix the protocol, run the baselines on it, then compare with paired uncertainty.

## Evaluation Contract

Every recommender result you deliver has these parts, in order:

1. **Protocol line:** global temporal cutoff (date), test window, what was masked (already-seen items), cold-item share of test interactions and how they were scored (as misses). Split rules: REQUIRED SUB-SKILL: splitting-recsys-data.
2. **Baselines on the same split:** most popular in train, **most popular in the last N days before the cutoff**, and item-kNN (or EASE). The current production model is a comparison, not a baseline (REQUIRED SUB-SKILL: establishing-baselines).
3. **Full ranking** over all train items. No sampled negatives (`uni100`, `pop100`) for the headline.
4. **Per-user metrics, mean ± 95% CI**, and **paired** Δ between models on the same users with CI (bootstrap over users). Relative lift gets its own CI (REQUIRED SUB-SKILL: evaluating-models-rigorously).
5. **Hyperparameters and epochs chosen on a validation cutoff** inside train. If they were chosen on the test window, the reported number is a selection maximum — say so and re-run.
6. **Slices:** user activity buckets (cold-ish vs heavy) and item popularity buckets (head vs tail) — a gain only on head items is popularity, not personalization (REQUIRED SUB-SKILL: auditing-recsys-bias).
7. **Verdict** that matches the evidence, ending in "offline result; online impact requires an A/B test" (REQUIRED SUB-SKILL: designing-ab-tests).

If the protocol cannot be re-run before the deadline, the slide says "evaluation in progress" — it does not show the old numbers with a caveat.

## What the Protocol Does to the Numbers

Synthetic data: 3,000 users, 2,000 items launched over 18 months with popularity decaying with age; 80/20 by row vs global temporal cutoff at the 80% quantile. recall@10:

| Model | Shuffled 80/20 | Temporal, full ranking | Temporal, cold items deleted | Temporal, `uni100` sampled |
|---|---|---|---|---|
| Popular (all train) | 0.067 | 0.012 | 0.024 | 0.244 |
| **Popular, last 30 days** | **0.017** (worst) | **0.066** (best) | 0.134 | 0.747 |
| Item-kNN | **0.068** (best) | 0.012 | 0.025 | 0.250 |
| ALS (implicit, 64 factors) | 0.044 | 0.010 (worst) | 0.019 | 0.206 |

- The shuffled split crowned item-kNN and made recent popularity look 4× worse than it; with the deployment-like split, recent popularity beats every model by 5×.
- Deleting cold items (62% of test interactions here) doubled every number.
- `uni100` multiplied recall by ~20 and shrank the gap between best and worst model from 6.8× to 3.6×. Sampled metrics are inconsistent with full-ranking metrics (Krichene & Rendle, KDD 2020).

## Library Gotchas

| Library | Gotcha | Do |
|---|---|---|
| RecBole 1.2.1 | Default `eval_args`: `split: {'RS':[0.8,0.1,0.1]}, order: RO, group_by: user, mode: full` — random order, per user | `order: TO`, `group_by: none`; keep `mode: full`; print train/test date ranges |
| RecBole | `mode: uni100` / `pop100` = 100 sampled negatives per positive | Never for the headline |
| implicit 0.7.3 | `ranking_metrics_at_k` "precision" = Σhits / Σmin(K, \|rel\|) over users — a micro-averaged recall, not precision; it returned AUC 0.50 for the ALS above | Compute per-user metrics yourself (below) so all models share one definition |
| recommenders 1.2.1 | `python_chrono_split` is per user, not a global cutoff | Global cutoff (splitting-recsys-data) |
| Any | Numbers from two libraries' metric functions | Not comparable; one evaluator for all models |

## Implementation

```python
import logging

import numpy as np
import pandas as pd
import scipy.sparse as sp

logger = logging.getLogger(__name__)


def evaluate_topk_per_user(
    scores: np.ndarray,
    seen: sp.csr_matrix,
    relevant_warm: list[np.ndarray],
    n_relevant: np.ndarray,
    k: int = 10,
) -> pd.DataFrame:
    """Recall@k e NDCG@k por usuário, ranking completo (sem amostrar negativos).

    Parameters
    ----------
    scores : np.ndarray
        Escores (n_usuarios_avaliados, n_itens_treino) do modelo.
    seen : sp.csr_matrix
        Itens de treino de cada usuário avaliado (mascarados no ranking).
    relevant_warm : list[np.ndarray]
        Índices dos itens relevantes do teste que existem no treino.
    n_relevant : np.ndarray
        Total de relevantes por usuário, INCLUINDO itens cold (contam como erro).
    k : int, optional
        Tamanho da lista, by default 10.

    Returns
    -------
    pd.DataFrame
        Colunas ``recall`` e ``ndcg``, uma linha por usuário.

    Raises
    ------
    ValueError
        Se algum usuário não tiver relevantes (remova-o antes: ex. só reassistiu itens do treino).

    Examples
    --------
    >>> per_user = evaluate_topk_per_user(model_scores, seen, rel, n_rel, k=10)
    """
    if (n_relevant == 0).any():
        raise ValueError("Há usuários sem itens relevantes no teste; remova-os antes de avaliar.")
    scores = scores.astype(np.float64, copy=True)
    rows, cols = seen.nonzero()
    scores[rows, cols] = -np.inf
    top = np.argpartition(-scores, k, axis=1)[:, :k]
    top = np.take_along_axis(top, np.argsort(-np.take_along_axis(scores, top, axis=1), axis=1), axis=1)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    recall, ndcg = np.zeros(len(top)), np.zeros(len(top))
    for u, items in enumerate(top):
        hits = np.isin(items, relevant_warm[u])
        denom = min(k, n_relevant[u])
        recall[u] = hits.sum() / denom
        ndcg[u] = (hits * discounts).sum() / discounts[:denom].sum()
    return pd.DataFrame({"recall": recall, "ndcg": ndcg})


def compare_paired_bootstrap(
    a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 42
) -> dict[str, float]:
    """Diferença A − B nos MESMOS usuários, com IC 95% por bootstrap pareado."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n_boot, len(a)))
    deltas = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    lifts = a[idx].mean(axis=1) / b[idx].mean(axis=1) - 1
    low, high = np.percentile(deltas, [2.5, 97.5])
    lift_low, lift_high = np.percentile(lifts, [2.5, 97.5])
    logger.info(f"Δ = {a.mean() - b.mean():+.4f} [IC95% {low:+.4f}, {high:+.4f}]")
    return {"delta": a.mean() - b.mean(), "ci_low": low, "ci_high": high,
            "lift": a.mean() / b.mean() - 1, "lift_ci_low": lift_low, "lift_ci_high": lift_high}
```

Verified with numpy 2.5.4, scipy 1.18.1, pandas 3.1.0, implicit 0.7.3. Score matrices come from any model (`als.user_factors[u] @ als.item_factors.T`, LightGCN/RecBole embeddings re-indexed to the train id map); batch users if `n_users × n_items` does not fit in memory. Popularity-last-N-days scores = `np.bincount(train[train.timestamp >= cutoff - N].item_idx, minlength=n_items)`.

## Common Mistakes

| Mistake | Fix |
|---|---|
| "Compared with the old model, no popularity baseline needed" | Recent popularity is the model to beat; it won every temporal comparison above |
| Ranking only items seen in train and dropping test items without embeddings | Cold relevant items stay in the denominator as misses; report the share |
| `uni100`/sampled-negative metrics on the slide | Full ranking |
| 22% relative lift with no interval | Paired Δ and lift with bootstrap CI over the same users |
| Re-using the production model's hyperparameters for one model and tuning the other | Same tuning budget on the same validation cutoff for every model |

## Rationalizations

| Excuse | Reality |
|---|---|
| "Two months and a GPU cluster went into LightGCN" | Sunk cost does not change the protocol. The re-run is one training on a new cutoff, not two months |
| "I'm the head of data, methodology is my call" | Deliver the contract above; state each instruction you changed and why |
| "Give me a fallback slide with the current numbers and a caveat" | The caveated 0.38 vs 0.31 is still read as "22% better". Fallback = "evaluation in progress", no numbers from the shuffled split |
| "Both models had the same leak, so the comparison is fair" | Leaks reorder models (table above); a shared leak is not a neutral one |

## Red Flags — Stop

- recall@10 above ~0.3 on a catalog of thousands with full ranking → check the split and the candidate set
- Recent popularity never computed, or it loses badly on a shuffled split
- Test set contains no items launched after the training data ends
- A metric named "precision" that grows when K grows
