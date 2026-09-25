---
name: reviewing-ds-code
description: Use when reviewing a pull request, diff, notebook, or snippet that loads data, preprocesses, engineers features, trains, tunes, or evaluates a model, especially when asked for a "quick look" or whether it "can be approved".
---

# Reviewing Data Science Code

## Overview

Data science bugs do not crash. A leaky pipeline runs, prints a great number, and passes every linter. **Core principle:** review the correctness of the *number* before the correctness of the *code*. Every finding names a location, the defect, its consequence, and the fix. Findings are ordered by severity; style is the linter's job.

## Output Contract

The review has exactly these parts, in this order:

1. **Verdict** — one line: `Request changes` or `Approve`, plus the single most important reason.
2. **Findings table**, most severe first:

   | Sev | Where | Defect → consequence | Fix |
   |---|---|---|---|
   | S1 | `train.py:14` | `scaler.fit_transform(X)` before split → test stats leak into train | Split first; `Pipeline([("scaler", ...), ("model", ...)])` |

   `Where` = `path:line`. If the diff has no line numbers, quote the offending line.
3. **Routing** — for each S1/S2 class found, the skill that covers the fix: leakage → REQUIRED SUB-SKILL: detecting-data-leakage; metrics, CIs, model comparison → REQUIRED SUB-SKILL: evaluating-models-rigorously.
4. **Style** — at most one line grouping S4 items ("naming, type hints, docstrings: run `ruff` + `basedpyright`").

The review contains no praise, no restatement of what the PR does, and no findings without a fix.

## Checklist (scan in this order)

**S1 — invalidates the reported result**
- Any `fit` / `fit_transform` (scaler, imputer, encoder, selector, SMOTE, PCA, target encoder) on data that includes the test rows.
- Metric computed on training data; test set used to pick hyperparameters, threshold, features, or model.
- Split does not match deployment: random split on time-ordered data; the same entity (customer, patient) in train and test.
- Feature built after the prediction moment (current-state table, post-outcome field, ID that encodes the target).

**S2 — wrong or not reproducible**
- No seed on split, model, CV, or sampler.
- Returned/saved object is not the evaluated one (search result discarded; transformers not persisted with the model).
- Silent row loss (`dropna`, inner join, filter) without logging how many rows left.
- Imbalanced target without `stratify`, or accuracy as the headline metric.
- Single point metric, no CI, no baseline.

**S3 — maintainability**: hardcoded paths, `print` instead of logging, new transform without tests, no schema check at stage boundary.

**S4 — style**: naming, docstrings, type hints.

## Common Mistakes

| Mistake | Fix |
|---|---|
| Twelve bullets of equal weight | Severity column; S1 first; S4 collapsed to one line |
| "Consider using a Pipeline" | Name the line, the consequence, and the exact fix |
| Reviewing only the diff lines | Check what the diff *feeds*: where `X_test` goes, what gets returned/saved |
| Approving "with nits" when a metric is invalid | Any S1 → `Request changes` |
| Rewriting the whole PR in the review | Findings + fixes; a full rewrite only if asked |

## Red Flags — you are reviewing style, not science

- First finding is about naming or PEP 8
- No finding mentions where `fit` is called relative to the split
- Review never asks which data the printed metric was computed on
