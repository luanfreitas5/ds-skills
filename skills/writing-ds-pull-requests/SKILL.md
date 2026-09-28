---
name: writing-ds-pull-requests
description: Use when opening, writing, or describing a pull request that changes data cleaning, filters, deduplication, schemas, features, labels, hyperparameters, or a trained model, or when asked to "just describe the code" and leave metrics for later.
---

# Writing Data Science Pull Requests

## Overview

In a DS pull request, the diff shows the code but hides what matters: which rows the model now sees and what happened to the metric. **Core principle:** a reviewer must be able to approve or reject the PR **from the description alone**. That requires the before/after metric with a CI, the data and schema changes, and links to the exact MLflow runs and data hashes.

## When to Use

- The diff touches `cleaning`, `filtering`, `features`, `schemas`, `labels`, `configs/model_params.yaml`, or `dvc.lock`
- A model was retrained, or will be retrained, because of the change

Not for: PRs that only touch docs, CI, or refactors that provably leave the output unchanged. For those, state the proof in the description: the output hash is the same before and after.

## Required PR Body

Fill in `pull_request_template.md` from this folder. The sections below are **required and go in this order**. A section with missing numbers is a blocker, not a TODO.

1. **Resumo**: one or two sentences with the change and the hypothesis behind it.
2. **Efeito na métrica**: main vs branch, with a point estimate, a 95% CI, and a **paired Δ with its CI and p-value** from a correct test (REQUIRED SUB-SKILL: evaluating-models-rigorously). Add a verdict whose wording matches the test, and any slice where the metric gets worse.
3. **Mudanças nos dados e no schema**: row count, distinct entities, target prevalence, and SHA-256 for main vs branch. Then **one row per new data rule**: before, after, rows affected, and the reason. Also state whether serving applies the same rule.
4. **Rastreabilidade**: full MLflow run URLs for **both** runs, the Git SHA, and full data hashes. Say whether `dvc.lock` was updated and `dvc push` was run (REQUIRED SUB-SKILL: tracking-experiments-mlflow).
5. **Mudanças de código**: keep it short, because the diff already shows the code.

## Comparable Before/After

When the PR changes cleaning or filters, main and the branch produce **different datasets**. The metric logged in each run is then measured on different rows and **is not comparable**. So:

- Score both models on the **same evaluation rows**: a frozen holdout, or the entity keys present in both datasets. Use the same folds and seed.
- Compare with a paired Δ and its CI. "The CIs overlap" is not a test. Overlapping intervals can still hide a significant paired difference.
- If data, features, and hyperparameters all change in one PR, add an ablation row for each change, or split the PR.
- When one entity now has several rows (e.g. `customer_id` × `snapshot_date`), the CV must group by that entity. Otherwise leakage inflates the Δ.

## Implementation

`summarize_data_change.py` (in this folder) builds the data section from the parquet files of both branches. It reports rows, lost entities, prevalence, full SHA-256 hashes, and columns whose dtype or null count changed:

```bash
git worktree add ../main-wt main && (cd ../main-wt && python -m src.main --stage process)
python -m src.main --stage process
python summarize_data_change.py ../main-wt/data/processed/train.parquet data/processed/train.parquet \
    --target churn --key customer_id 2>/dev/null >> pr_body.md
```

Split the "rows removed" count by rule, e.g. `filter(monthly_charges <= 0).height` on the main dataset. A single net number hides which rule removed which rows and whether they had a different churn rate.

## Common Mistakes

| Mistake | Fix |
|---|---|
| Body lists changed files and functions only | Metric and data sections come first; the code section is last |
| "Metrics on Monday" / "looked a bit better" | Blocker: run both evaluations before requesting review, or open as **Draft** |
| Main metric taken from an old run on the old dataset | Re-score both models on the same evaluation rows |
| "ICs overlap, so no claim" | Paired Δ with CI + p-value |
| Schema change (`nullable=True`) buried in a bullet | Its own row: before, after, rows affected, reason, serving impact |
| Hash or run ID truncated (`9f3c1a…e71b`) | Full hash and full clickable run URL |
| Net row delta only | One row per rule, with prevalence of the removed rows |

## Red Flags: Stop

- "Just describe the code, metrics later": a DS PR without metrics is not ready for review. Open it as a **Draft**.
- "The run is somewhere in MLflow": find it and link it, or re-run it.
- Any metric in the body without a CI, or any "improves" without a paired test.
- Filling a blank with an estimate, "~0.6", or the old run's number. Mark it as a blocker instead.
