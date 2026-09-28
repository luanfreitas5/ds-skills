---
name: building-rag-pipelines
description: Use when building retrieval-augmented generation (RAG), a document Q&A chatbot, or semantic search feeding an LLM, when choosing chunk size, overlap, embedding model, top-k, hybrid search, or a reranker, or when a RAG system is judged only by reading its final answers.
---

# Building RAG Pipelines

## Overview

A RAG answer can fail in two places: the retriever never surfaced the evidence, or the generator ignored or invented it. If you only score the final answer, you cannot tell which one failed. And chunk size, overlap, embedding model, and k picked as "reasonable defaults" are untested hypotheses. **Core principle:** label evidence **spans**, not chunk ids. Measure retrieval (recall@k, MRR) separately from generation (faithfulness, citation validity, abstention). Choose every retrieval knob by comparing candidates on the same questions with a confidence interval.

## Evaluation Set (build before tuning)

| Requirement | How |
|---|---|
| Gold = `doc_id` + evidence text span | Relevance of any chunk = "contains the span". Labels survive re-chunking, so chunk sizes can be compared. Labels keyed to `chunk_id` or to "the chunk the question was generated from" die with the first re-chunk |
| Real questions first | Tickets, search logs, SME-written questions. Questions an LLM generated from a chunk copy its wording and inflate lexical recall. Tag them `synthetic` and report them separately |
| Unanswerable slice | ≥ 15% of items whose answer is **not** in the corpus: out of scope, near misses ("parental leave" when only vacation exists), and superseded facts. Gold = abstain |
| Slices | Tag by source/department, question type, and multi-hop, and report per slice |

Set size, freezing, and judge calibration: REQUIRED SUB-SKILL: evaluating-llm-outputs.

## Retrieval: measure, then choose

- **Metrics:** recall@k (was the evidence in what the LLM saw?) and MRR (how high was it ranked?). Report both with a bootstrap 95% CI. Compare configurations **paired** on the same items.
- **Knobs are candidates, not defaults.** Compare ≥ 2–3 chunk sizes, as well as overlap, embedding models (plus BM25/hybrid), k, and whether to add a reranker. A knob with no comparison is a guess. Write it down as an assumption.
- **Report context cost next to recall.** Huge chunks trivially max out recall@k while burying the evidence in noise and tokens. Pick the smallest configuration whose recall CI overlaps the best one's.
- **Chunk along structure.** Split on sections/articles/pages before splitting on length. Prepend the document title and section heading to each chunk.
- **Metadata per chunk:** `doc_id`, `chunk_id`, char offsets, page, section, `effective_date`/version, and `access_group`. Each field has a job: citations need offsets and page, dedup and "latest version" need the date, and permission filtering needs `access_group`. That filter is applied **at retrieval**, never in the prompt.

## Generation: cite or abstain

- **Citations are mandatory and machine-checked.** Every claim cites a `chunk_id`. For every citation, check that the id was actually **retrieved** (otherwise it is a hallucinated source) and that the cited chunk contains the supporting evidence.
- **Abstention is a requirement with metrics.** Report correct-abstention rate on the unanswerable slice **and** the false-abstention rate on answerable questions. Tune the abstention threshold or prompt on a validation split, never on the test set.
- **Score the raw generator output.** A runtime guard that turns an invalid citation into an abstention is fine in production. But the eval scores the output **before** the guard and reports `citation_not_retrieved` as its own rate. Otherwise the guard hides the failure.
- **Faithfulness** (every claim supported by the cited context) is scored by humans or by a κ-calibrated judge. Keep it separate from answer correctness.

## Implementation

Offline, it runs with `rank-bm25` + `numpy`. To test dense or hybrid retrieval, swap in another retriever that exposes the same `scores` function.

```python
import logging
import re
from dataclasses import dataclass

import numpy as np
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Chunk:
    """Trecho indexado com metadados para citação."""

    chunk_id: str
    doc_id: str
    start: int
    text: str


def tokenize(text: str) -> list[str]:
    """Tokenização simples: minúsculas, apenas alfanuméricos."""
    return re.findall(r"\w+", text.lower())


def chunk_document(doc_id: str, text: str, size: int, overlap: int) -> list[Chunk]:
    """Divide o documento em janelas de `size` caracteres, guardando o offset inicial."""
    step = size - overlap
    return [Chunk(f"{doc_id}#{s}", doc_id, s, text[s : s + size]) for s in range(0, max(len(text) - overlap, 1), step)]


def normalize(text: str) -> str:
    """Normaliza espaços e caixa (a extração de PDF quebra linhas arbitrariamente)."""
    return re.sub(r"\s+", " ", text).strip().lower()


def is_relevant(chunk: Chunk, item: dict) -> bool:
    """Rótulo independente do chunking: o chunk contém o span de evidência do documento-ouro?"""
    return chunk.doc_id == item["doc_id"] and normalize(item["evidence"]) in normalize(chunk.text)


def bootstrap_ci(values: np.ndarray, n_boot: int = 5000, seed: int = 42) -> tuple[float, float, float]:
    """Média e IC 95% bootstrap (percentil)."""
    idx = np.random.default_rng(seed).integers(0, len(values), (n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def evaluate_retrieval(chunks: list[Chunk], items: list[dict], k: int) -> dict[str, np.ndarray]:
    """Recall@k, reciprocal rank e tamanho do contexto top-k por pergunta respondível."""
    bm25 = BM25Okapi([tokenize(c.text) for c in chunks])
    recall, rr, ctx = [], [], []
    for item in items:
        order = np.argsort(-bm25.get_scores(tokenize(item["question"])))
        ranks = [r for r, i in enumerate(order, 1) if is_relevant(chunks[i], item)]
        recall.append(float(bool(ranks) and ranks[0] <= k))
        rr.append(1.0 / ranks[0] if ranks else 0.0)
        ctx.append(sum(len(chunks[i].text) for i in order[:k]))
    return {"recall": np.array(recall), "rr": np.array(rr), "ctx_chars": np.array(ctx, dtype=float)}


def compare_chunkings(docs: dict[str, str], items: list[dict], configs: list[tuple[int, int]], k: int = 3) -> dict:
    """Compara configurações de chunking nas MESMAS perguntas respondíveis, com IC bootstrap."""
    answerable = [it for it in items if it["doc_id"] is not None]
    results = {}
    for size, overlap in configs:
        chunks = [c for d, t in docs.items() for c in chunk_document(d, t, size, overlap)]
        res = evaluate_retrieval(chunks, answerable, k)
        results[(size, overlap)] = res
        r, lo, hi = bootstrap_ci(res["recall"])
        m, mlo, mhi = bootstrap_ci(res["rr"])
        logger.info(
            f"chunk={size}/{overlap}: recall@{k}={r:.2f} [{lo:.2f}, {hi:.2f}] "
            f"MRR={m:.2f} [{mlo:.2f}, {mhi:.2f}] contexto≈{res['ctx_chars'].mean():.0f} caracteres"
        )
    return results


def check_answer(output: dict, retrieved_ids: set[str], chunks_by_id: dict[str, Chunk], item: dict) -> str:
    """Classifica uma resposta gerada quanto a abstenção e validade das citações."""
    if item["doc_id"] is None:
        return "correct_abstain" if output["abstained"] else "answered_unanswerable"
    if output["abstained"]:
        return "false_abstain"
    if not output["citations"]:
        return "missing_citation"
    if any(c not in retrieved_ids for c in output["citations"]):
        return "citation_not_retrieved"  # fonte alucinada
    if not any(is_relevant(chunks_by_id[c], item) for c in output["citations"]):
        return "citation_unsupported"
    return "supported"
```

Items look like `{"question": ..., "doc_id": "vacation", "evidence": "Up to 5 unused vacation days"}`, with `doc_id: None` for unanswerable ones. `output` is `{"answer", "citations": [chunk_id, ...], "abstained": bool}`, parsed from the generator's structured output. In unit tests, stub the generator and assert every `check_answer` category. On an 18-question toy corpus, 80-character chunks split evidence spans (recall@3 0.50, CI [0.29, 0.79]) while 200-character chunks reached 1.00 (paired difference +0.50, CI [0.21, 0.71]). 600-character chunks also scored 1.00, but with 2× the context. A naive score-threshold abstention produced 3 unsupported citations, 4 false abstentions, and 1 answered unanswerable question, which only the citation and abstention checks exposed. A bootstrap CI of [1.00, 1.00] on 14 items is not certainty. Small sets cannot separate configurations that are both near the ceiling.

## Rationalizations

| Excuse | Reality |
|---|---|
| "500 tokens / 15% overlap is the standard" | It is a starting candidate. Compare it with a smaller and a larger size on the same labeled spans. That takes about an hour |
| "No labeled data, so eyeball the answers" | 50 real questions with an evidence span each is an afternoon of work. Eyeballing cannot separate retrieval misses from generation errors |
| "Label = the chunk the question was generated from" | That breaks on re-chunking and rewards the chunking that generated it. Label spans instead |
| "The prompt says cite sources and say I don't know" | A prompt instruction is not a guarantee. Check citations in code and measure abstention on the unanswerable slice |
| "Recall is 1.0 with big chunks, done" | Check context size and MRR. Pick the smallest config inside the best CI |
| "Demo is Friday, skip evaluation" | The demo is exactly where an invented policy citation hurts most. Run the retrieval eval before the demo |

## Common Mistakes

| Mistake | Fix |
|---|---|
| One end-to-end "answer quality" score | Retrieval (recall@k, MRR) and generation (faithfulness, citations, abstention) reported separately |
| No unanswerable questions | ≥ 15% of items, with correct- and false-abstention rates reported |
| Citation `[3]` never validated | Cited id ∈ retrieved ids **and** cited chunk contains the evidence |
| Access control only in the prompt | Metadata filter at retrieval time |
| Old and new policy versions both indexed | `effective_date`/version metadata, deduplicated, and "superseded" questions in the eval |
| Guard silently converts bad citations to "I don't know" | Count and report the pre-guard `citation_not_retrieved` rate |
| Chunk text used as the reference for evidence spans | Copy spans from the extracted text and normalize whitespace before matching |
| Tuning k/threshold on the test questions | Validation split for tuning; frozen test set for the report |
