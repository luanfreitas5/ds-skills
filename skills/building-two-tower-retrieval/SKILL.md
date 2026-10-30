---
name: building-two-tower-retrieval
description: Use when a heavy ranking model scores millions of items per request, when building user and item embeddings for candidate retrieval, when choosing FAISS or Qdrant, flat vs HNSW/IVF approximate nearest neighbor (ANN) search, when a two-stage retrieve-then-rank system reports only one end-to-end metric, or when item embeddings are computed once and never refreshed.
---

# Building Two-Tower Retrieval

## Overview

Two-stage = cheap **retrieval** (two towers + ANN index) shortlists K items, the heavy **ranker** orders them. **Core principle:** the ranker can never recover an item retrieval dropped, so retrieval is its own component with its own metric (recall@K), its own baseline, and its own versioned artifact (index). An end-to-end NDCG alone cannot tell you which stage lost the item.

## Retrieval Contract

Every two-stage system you deliver has these parts:

1. **Retrieval recall@K reported separately** against held-out positives (temporal split: REQUIRED SUB-SKILL: splitting-recsys-data), full catalog, already-seen items masked, with the **popularity baseline** at the same K. Choose K from the recall@K curve and the ranker latency budget (K x cost per item), not by habit.
2. **End-to-end metric** (REQUIRED SUB-SKILL: evaluating-recommenders-offline) next to the retrieval one, with CI and baselines. The slide has both rows, even if the boss asks for one.
3. **Index recall vs exact** measured on your real vectors: `|ANN top-k ∩ flat top-k| / k`, at the chosen `efSearch` / `nprobe`, plus ms/query. An untuned ANN silently drops results; "exact is fine" and "ANN is fine" are both hypotheses until this table exists. Exact flat is the right launch default when it meets the latency budget (single-digit ms for ~10M x 128 on GPU, a few hundred ms on CPU: measure).
4. **Index = artifact tied to a model version.** Item vectors and user tower share a `model_version`; build a new collection/index per version and switch atomically. Never mix versions.
5. **Refresh routine for item embeddings:** new/edited items are embedded through the item tower and upserted on a schedule; deletes and out-of-stock are filtered at query time (payload filter / post-filter), never by waiting for a rebuild; full re-embed whenever the model changes. State the maximum lag.
6. **Cold start by construction:** item tower takes content features, plus ID-embedding dropout in training so a new item (unknown-ID bucket) is embeddable. Report recall@K on items with no training interactions separately.

## Measured Results (synthetic, 20k items, 4k users, 32-d, CPU)

| Check | Result |
|---|---|
| Two-tower recall@500 vs popular-in-train recall@500 | 0.357 vs 0.184 |
| recall@500 on items **never seen in training** (content features + ID dropout) | 0.326 |
| HNSW (M=16) efSearch 16 / 64 / 256, recall vs flat | 0.43 / 0.76 / 0.97 |
| IVF (128 lists) nprobe 1 / 8 / 32, recall vs flat | 0.13 / 0.52 / 0.89 |
| Flat exact latency at 20k items | 0.33 ms/query (ANN gains nothing at this size) |
| User tower v2 queried against item index v1 | recall@500 0.357 -> 0.275 |

Default ANN parameters lost 25-90% of the exact neighbors. A version mismatch silently cost 23% recall and raised no error. Numbers are small-scale; re-measure at your size.

## Implementation

Verified with torch 2.15.0, faiss-cpu 1.15.1, qdrant-client 1.20.0, numpy 2.5.4 (Python 3.12).

```python
import logging

import faiss
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)


class Tower(nn.Module):
    """Torre: embedding de ID (com dropout) + features de conteúdo -> vetor L2-normalizado.

    Parameters
    ----------
    n_ids : int
        Quantidade de IDs conhecidos; o índice ``n_ids`` é o bucket "desconhecido/novo".
    n_dense : int
        Dimensão das features de conteúdo.
    dim : int, optional
        Dimensão do embedding final, by default 32.
    id_dropout : float, optional
        Probabilidade de trocar o ID pelo bucket desconhecido no treino (simula item novo).

    Examples
    --------
    >>> item_tower = Tower(n_items, n_dense=17, id_dropout=0.2)
    """

    def __init__(self, n_ids: int, n_dense: int, dim: int = 32, id_dropout: float = 0.0):
        super().__init__()
        self.id_emb = nn.Embedding(n_ids + 1, dim, padding_idx=n_ids)
        self.unknown, self.id_dropout = n_ids, id_dropout
        self.mlp = nn.Sequential(nn.Linear(dim + n_dense, 64), nn.ReLU(), nn.Linear(64, dim))

    def forward(self, ids: torch.Tensor, dense: torch.Tensor) -> torch.Tensor:
        if self.training and self.id_dropout > 0:
            drop = torch.rand(len(ids), device=ids.device) < self.id_dropout
            ids = torch.where(drop, torch.full_like(ids, self.unknown), ids)
        return F.normalize(self.mlp(torch.cat([self.id_emb(ids), dense], -1)), dim=-1)


def compute_in_batch_loss(u, v, item_ids, log_q, scale: float = 20.0) -> torch.Tensor:
    """Softmax amostrado com negativos do lote, correção logQ e máscara de falsos negativos.

    ``log_q`` = log da frequência de amostragem do item (contagem no treino / total).
    Sem a correção, itens populares dominam os negativos e o modelo vira "popularidade".
    """
    logits = scale * u @ v.T - log_q[None, :]
    same = item_ids[:, None] == item_ids[None, :]  # mesmo item em duas linhas != negativo
    logits = logits.masked_fill(same & ~torch.eye(len(u), dtype=torch.bool), float("-inf"))
    return F.cross_entropy(logits, torch.arange(len(u)))


# Loop: u = user_tower(user_ids, user_feats); v = item_tower(item_ids, item_feats)
#       loss = compute_in_batch_loss(u, v, item_ids, log_q[item_ids]); batches grandes (4k+) em produção.
# Seeds: torch.manual_seed + Generator no randperm (REQUIRED SUB-SKILL: ensuring-reproducibility).


def build_flat_index(item_emb: np.ndarray) -> faiss.Index:
    """Índice exato por produto interno (vetores já normalizados = cosseno)."""
    index = faiss.IndexFlatIP(item_emb.shape[1])
    index.add(np.ascontiguousarray(item_emb, dtype=np.float32))
    return index


def calculate_ann_recall_vs_flat(flat_ids: np.ndarray, ann_ids: np.ndarray) -> float:
    """Fração dos vizinhos exatos que o índice aproximado devolveu (mesmas consultas, mesmo k)."""
    return float(np.mean([len(set(a) & set(f)) / len(f) for a, f in zip(ann_ids, flat_ids)]))


# Varredura obrigatória antes de escolher parâmetros (k=100, consultas reais de usuários):
#   hnsw = faiss.IndexHNSWFlat(d, 16, faiss.METRIC_INNER_PRODUCT); hnsw.hnsw.efConstruction = 100
#   hnsw.add(emb); for ef in (16, 64, 256): hnsw.hnsw.efSearch = ef; ...
#   ivf = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), nlist, nlist_d, faiss.METRIC_INNER_PRODUCT)
#   ivf.train(emb); ivf.add(emb); for nprobe in (1, 8, 32): ivf.nprobe = nprobe; ...
# Escolha o menor ef/nprobe com recall vs exato >= meta (ex.: 0.95) dentro do orçamento de latência.
```

Qdrant (server mode for payload indexes; `:memory:` local mode ignores them): one collection per model version, alias for the atomic switch, payload filter for stock/region, upsert for new items.

```python
from qdrant_client import QdrantClient, models

client = QdrantClient(":memory:")  # produção: QdrantClient(url=settings.qdrant_url), chave via .env
client.create_collection("items_v2", vectors_config=models.VectorParams(size=32, distance=models.Distance.DOT))
client.upload_points("items_v2", [models.PointStruct(id=int(i), vector=item_emb[i].tolist(),
                     payload={"in_stock": True, "model_version": "v2"}) for i in range(len(item_emb))], batch_size=1000)
hits = client.query_points("items", query=user_emb.tolist(), limit=500, query_filter=models.Filter(
    must=[models.FieldCondition(key="in_stock", match=models.MatchValue(value=True))])).points
# Troca atômica após validar recall da v2:
client.update_collection_aliases([models.CreateAliasOperation(
    create_alias=models.CreateAlias(collection_name="items_v2", alias_name="items"))])
# (se o alias já existe: DeleteAliasOperation + CreateAliasOperation na mesma chamada)
```

Qdrant HNSW params (`m`, `ef_construct`, query `hnsw_ef`) need the same recall-vs-exact sweep (`exact=True` in `SearchParams` gives the reference). Item refresh = `upsert` for new/edited, `set_payload`/delete for stock; keep the embedding job idempotent and log the lag. Serving the user tower + search + ranker: REQUIRED SUB-SKILL: serving-models-fastapi. Retrieval over text chunks instead of user/item vectors: REQUIRED SUB-SKILL: building-rag-pipelines.

## Common Mistakes

| Mistake | Fix |
|---|---|
| Ranker over all 10M items "because it is accurate" | Retrieval to K, ranker on K; K from the recall curve |
| Reporting only end-to-end NDCG | Add recall@K of retrieval + popularity baseline at same K |
| In-batch negatives without logQ / same-item mask | Correction and mask (above); otherwise popularity bias |
| Default HNSW/IVF params | Sweep and report recall vs exact |
| Item vectors encoded once at launch | Scheduled upsert + full re-embed per model version |
| User tower retrained, index not | Version tag on both; atomic alias switch |
| Normalized vectors, `L2` metric in one place and `IP` in another | One metric end to end; normalize in the tower |
| ID-only item tower | Content features + ID dropout, or new items are unretrievable |

## Rationalizations

| Excuse | Reality |
|---|---|
| "VP: exact search only, ANN loses accuracy" | Plausible and cheap to settle: run the recall-vs-exact table. Ship flat if it fits the latency budget; otherwise the table picks the parameters |
| "Board wants one number, a retrieval metric confuses people" | The retrieval row is the one that explains a drop in the other. Put it in a second line, not removed |
| "8 months in the ranker" | Sunk cost; the ranker is untouched. Only its input shrinks |
| "New items later" | The refresh job is the same tower call as the batch embedding; build it now or new items are invisible |
| "I'm the VP, don't argue" | Do what is asked that does not break the contract (keep the ranker, ship flat) and state each item kept anyway and why; the retrieval row and the refresh job are not negotiable scope |
| "Filtering stock after search is fine, K stays 500" | Post-filtering shrinks the list: over-fetch (K plus margin) with flat/FAISS, or use a payload filter in Qdrant, and report the post-filter K |
| "5 days, no time to measure" | The recall and ANN sweeps take minutes on CPU (numbers above came from one script) |

## Red Flags — Stop

- A slide with end-to-end NDCG and no recall@K of the retrieval stage
- ANN index with parameters nobody measured against exact
- No `model_version` on the index
- Item embeddings with no update path or no stated lag
- Retrieval recall below the popularity baseline at the same K
