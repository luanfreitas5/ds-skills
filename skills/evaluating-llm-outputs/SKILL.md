---
name: evaluating-llm-outputs
description: Use when measuring the quality of a prompt, agent, RAG system, or LLM pipeline, when comparing two prompt or model versions before shipping, when using an LLM as a judge, or when an evaluation rests on a handful of hand-picked examples.
---

# Evaluating LLM Outputs

## Overview

Five hand-picked examples measure the author's taste, not the system. An LLM judge whose scores were never compared with humans measures the judge. **Core principle:** a versioned evaluation set that includes hard cases, metrics that were validated against human labels, and a **paired** comparison between versions with a confidence interval.

## Evaluation Set

| Requirement | How |
|---|---|
| Size | ≥ 200 items for a decision; with n items a 95% CI on accuracy is about ±1/√n (n = 100 → ±10 pp) |
| Sampling | Random sample of real traffic **plus** a hard-case slice (ambiguous, long, multilingual, adversarial, past incidents), tagged |
| Labels | Human reference where the task has one (class, extracted field); rubric labels where it does not (summary quality) |
| Frozen and versioned | `evals/<task>/v3.jsonl` in DVC/Git with a hash; never edited after seeing a candidate's results — add a new version |
| No leakage | Items not used as few-shot examples in any prompt under test |

## Metrics

1. **Deterministic first.** Classification → accuracy / macro-F1 vs human labels. Extraction → exact/field match. Format → schema validation rate.
2. **LLM judge only for what cannot be computed**, and only after calibration:
   - Humans and judge label the same ≥ 100 items with the same rubric (binary or 3-point per criterion beats a 1–10 score).
   - Agreement: Cohen's κ (weighted κ for ordinal). κ ≥ 0.6 → usable; below → fix rubric/judge, do not report.
   - Pairwise judging (A vs B) with order randomized and swapped; a judge that flips with order counts as a tie.
3. **Slices**: report per tag (hard cases, category, language), not only overall.

## Comparing Versions

Same items, both versions → paired analysis: bootstrap CI of the difference, and McNemar for binary correctness. Ship only if the CI lower bound clears the agreed minimum and no hard-case slice regresses. REQUIRED SUB-SKILL: evaluating-models-rigorously.

## Implementation

```python
import numpy as np
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.contingency_tables import mcnemar


def calibrate_judge(human: list[int], judge: list[int], ordinal: bool = False) -> float:
    """Concordância juiz × humanos (kappa de Cohen; ponderado se a escala for ordinal)."""
    return float(cohen_kappa_score(human, judge, weights="quadratic" if ordinal else None))


def compare_versions_paired(correct_v1: np.ndarray, correct_v2: np.ndarray, n_boot: int = 5000, seed: int = 42) -> dict[str, float]:
    """Diferença de acurácia v2 − v1 nos mesmos itens, IC 95% bootstrap e p-valor de McNemar."""
    rng = np.random.default_rng(seed)
    n = len(correct_v1)
    idx = rng.integers(0, n, (n_boot, n))
    diffs = correct_v2[idx].mean(axis=1) - correct_v1[idx].mean(axis=1)
    table = [
        [np.sum(correct_v1 & correct_v2), np.sum(correct_v1 & ~correct_v2)],
        [np.sum(~correct_v1 & correct_v2), np.sum(~correct_v1 & ~correct_v2)],
    ]
    return {
        "diff": float(correct_v2.mean() - correct_v1.mean()),
        "ci_low": float(np.percentile(diffs, 2.5)),
        "ci_high": float(np.percentile(diffs, 97.5)),
        "mcnemar_p": float(mcnemar(table, exact=True).pvalue),
    }
```

`correct_v1` / `correct_v2` are boolean arrays aligned by item id (urgency matches the human label; or judge verdict "acceptable" once κ is acceptable). Log eval set version, prompt versions, model name, temperature, and seed with the results.

## Common Mistakes

| Mistake | Fix |
|---|---|
| 5 curated "good" tickets | ≥ 200 sampled items + tagged hard cases |
| GPT-4o 1–10 score, never checked | Rubric + κ against ≥ 100 human labels; binary/3-point |
| Judge = same model family as the system, same prompt style | Different judge or human spot checks; watch self-preference |
| Comparing averages from different item sets | Same items, paired test |
| Editing the eval set after seeing failures of v2 | New eval version; rerun both versions |
| One overall number | Per-slice metrics, hard cases shown separately |
| Human-labeled set used as few-shot examples | Keep eval items out of prompts |
