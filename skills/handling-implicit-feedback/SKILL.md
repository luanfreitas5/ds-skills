---
name: handling-implicit-feedback
description: Use when building a recommender from clicks, views, add-to-cart, purchases or plays with no explicit ratings, when event types are mapped to fake 1-5 ratings, when non-interacted items are filled with 0 as negatives, when an MSE or RMSE regressor is trained on clicks, when negatives are sampled uniformly for BPR or BCE, or when choosing confidence weights or alpha for implicit ALS.
---

# Handling Implicit Feedback

## Overview

An event log has no negatives. A missing pair means "not seen yet", not "rejected". **Core principle:** model the observed pairs as positives with **confidence**, treat everything else as *unobserved* (low weight, never a strong label), and judge the result with ranking metrics on a temporal split — never RMSE.

Verified with implicit 0.7.3, scipy 1.18.1, pandas 3.1.0, torch 2.15.0 (CPU). `implicit` takes the matrix you pass as the confidence values; you apply alpha yourself.

## Protocol

1. **Never regress on observed events only.** Training MSE on `view=1/cart=3/purchase=5` rows only learns "what rating would this click get", not "will they interact". A low RMSE says nothing about ranking.
2. **Unobserved ≠ negative.** Do not fill the dense matrix with 0 targets and call them strong negatives. Zeros enter as low-confidence preference-0 (what ALS does by default), or as sampled negatives with the caveats in step 5.
3. **Event type = confidence, not rating.** Build `C` from observed pairs (`build_confidence_matrix`). Weights and `alpha` are hyperparameters: choose them on a validation cutoff, with ranking metrics. Large alpha is not "more signal".
4. **Choose the loss for ranking:** ALS on `C` (implicit), or BPR / BCE with sampled negatives in PyTorch. Not MSE on ratings.
5. **Negative sampling:** (a) exclude the user's train positives, otherwise ~3% of "negatives" are items they already bought; (b) some sampled items will be bought *later* (0.5% here) — unavoidable, so they are noise, not labels; (c) sample on the **train window only**; (d) never evaluate against sampled negatives.
6. **Evaluate with full ranking** on a global temporal split with baselines (REQUIRED SUB-SKILL: splitting-recsys-data, then REQUIRED SUB-SKILL: evaluating-recommenders-offline). Baselines first: REQUIRED SUB-SKILL: building-recsys-baselines. RMSE and AUC over sampled negatives are not deliverables.
7. **Do not deliver the broken pipeline "plus a variant".** Fix the target/loss and the split, and state what changed and why. A trained model with a nice loss is a sunk cost, not evidence.

## Measured (synthetic: 2,000 users x 800 items, 74k events, latent preferences, popularity skew; temporal cutoff, recall@10 on later purchases, train-seen items masked)

| Model | recall@10 |
|---|---|
| Most popular in train | 0.054 |
| Regression (MSE) on observed events only | **0.034** (below popularity; RMSE 0.77 looked fine) |
| Dense MSE with zero-filled pairs | 0.125 |
| implicit ALS, event-weighted confidence, alpha=1 | **0.126** |
| ALS, binary confidence, alpha=1 / 2 | 0.096 / 0.108 |
| ALS, event weights, alpha=1 / 5 / 40 | 0.126 / 0.110 / 0.083 |
| BPR, uniform negatives vs excluding train positives | 0.094 vs 0.097 (single seed; within noise) |

- Observed-only regression is the real failure: worse than popularity.
- Dense zero-filled MSE matches ALS here because it is the same least-squares objective with uniform weight on zeros. It fails on **cost and meaning** (5M x items dense target, zeros read as dislikes), not on this recall number. Do not claim it "loses" without measuring.
- Event-type weights helped here because purchases are informative; on another log they may not. Tune, do not assume.
- Single dataset, single seed: report the ordering as indicative and re-run with CIs on yours (REQUIRED SUB-SKILL: evaluating-models-rigorously).

## Implementation

```python
import logging

import numpy as np
import pandas as pd
import scipy.sparse as sp

logger = logging.getLogger(__name__)

EVENT_WEIGHTS = {"view": 1.0, "add_to_cart": 3.0, "purchase": 5.0}  # ponto de partida: ajuste na validação

def build_confidence_matrix(
    train: pd.DataFrame, n_users: int, n_items: int, alpha: float = 1.0,
    weights: dict[str, float] = EVENT_WEIGHTS,
) -> sp.csr_matrix:
    """Matriz usuário×item de CONFIANÇA (não de nota) só com eventos observados.

    Parameters
    ----------
    train : pd.DataFrame
        Eventos da janela de treino: ``user_id``, ``item_id`` (índices inteiros), ``event_type``.
    n_users, n_items : int
        Dimensões da matriz (definidas pelo treino).
    alpha : float, optional
        Escala da confiança; escolher na validação, by default 1.0.
    weights : dict[str, float], optional
        Peso por tipo de evento; o par recebe o MAIOR peso observado.

    Returns
    -------
    sp.csr_matrix
        Zeros = "não observado" (confiança implícita 1 no ALS), nunca "rejeitado".

    Raises
    ------
    ValueError
        Se houver ``event_type`` fora de ``weights``.

    Examples
    --------
    >>> conf = build_confidence_matrix(train, n_users, n_items, alpha=2.0)
    """
    w = train["event_type"].map(weights)
    if w.isna().any():
        raise ValueError(f"Tipos de evento desconhecidos: {sorted(train.loc[w.isna(), 'event_type'].unique())}")
    pairs = train.assign(w=w).groupby(["user_id", "item_id"], as_index=False)["w"].max()
    conf = sp.csr_matrix(
        ((alpha * pairs["w"]).astype(np.float32), (pairs["user_id"], pairs["item_id"])),
        shape=(n_users, n_items),
    )
    logger.info(f"Matriz de confiança: {conf.nnz:,} pares observados, densidade={conf.nnz / (n_users * n_items):.2%}")
    return conf

def sample_negatives(
    users: np.ndarray, conf: sp.csr_matrix, rng: np.random.Generator, max_tries: int = 20
) -> np.ndarray:
    """Um item candidato a negativo por usuário, sem colidir com o histórico de treino.

    Parameters
    ----------
    users : np.ndarray
        Índice do usuário de cada exemplo positivo.
    conf : sp.csr_matrix
        Matriz de treino (qualquer valor > 0 = interagiu).
    rng : np.random.Generator
        Gerador com semente fixa.
    max_tries : int, optional
        Rodadas de re-sorteio das colisões, by default 20.

    Returns
    -------
    np.ndarray
        Itens amostrados; são NÃO OBSERVADOS, não negativos verdadeiros.

    Examples
    --------
    >>> neg = sample_negatives(users, conf, np.random.default_rng(42))
    """
    n_items = conf.shape[1]
    seen_keys = np.sort(conf.nonzero()[0].astype(np.int64) * n_items + conf.nonzero()[1])
    users = users.astype(np.int64)
    neg = rng.integers(0, n_items, len(users))
    for _ in range(max_tries):
        bad = np.isin(users * n_items + neg, seen_keys)
        if not bad.any():
            break
        neg[bad] = rng.integers(0, n_items, int(bad.sum()))
    else:
        raise RuntimeError("Usuários com quase todo o catálogo visto: use amostragem sem reposição")
    return neg

# conf = build_confidence_matrix(train_idx, n_users, n_items, alpha=1.0)   # tune alpha/weights on val
# als = AlternatingLeastSquares(factors=32, regularization=0.1, iterations=15, random_state=42)
# als.fit(conf)                       # usuário x item; ranking: mascarar itens já vistos
# neg = sample_negatives(pos_users, conf, np.random.default_rng(42))        # para BPR/BCE
```

BPR loss: `-F.logsigmoid(score(u, pos) - score(u, neg)).mean()`, with `neg` redrawn every epoch. Recommendation with `implicit`: `als.recommend(u, conf[u], N=10, filter_already_liked_items=True)`.

## Common Mistakes

| Mistake | Fix |
|---|---|
| Ratings 1/3/5 from event types as MSE targets | Confidence on observed pairs; ranking loss |
| Reporting RMSE or AUC on sampled negatives | Full-ranking recall/NDCG, temporal split |
| Uniform negatives incl. already-bought items | Reject train positives (`sample_negatives`) |
| Sampling negatives from all data | Train window only |
| `alpha` copied from a blog (40) | Validation sweep; 40 was worst here |
| Same user-item pair counted per event | One row per pair (max weight) or documented aggregation |

## Rationalizations

| Excuse | Reality |
|---|---|
| "Demo in 2 hours, no time to change the target" | The loss and the matrix change, the data pipeline stays. ALS fits in seconds; a below-popularity model is the slower failure |
| "I already spent two days on the rating pipeline" | Sunk cost. Keep the event table and weights, replace target and loss |
| "The lead has shipped this before" | Ask for the ranking number against popularity on a temporal split; authority is not a metric |
| "Loss looks great" | RMSE here was 0.77 while recall was below popularity |
| "A zero is a pretty safe negative, they didn't click" | They were never shown most items; unobserved |
| "Give me the code as requested, add a note" | Deliver the corrected code and say what changed; do not hand over the broken version with a caveat |
| "Random 80/20 is standard" | REQUIRED SUB-SKILL: splitting-recsys-data |

## Red Flags — Stop

- Target column named `rating` built from event types
- Dense user x item matrix materialised with 0 as label
- Only RMSE/MAE/AUC reported, no popularity baseline
- Negatives sampled with no exclusion of train positives, or from data after the cutoff
