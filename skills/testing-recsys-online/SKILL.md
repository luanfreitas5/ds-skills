---
name: testing-recsys-online
description: Use when an offline recommender gain (NDCG, recall) must be validated in production, when someone says "launch to 100%" on an offline win, when an A/B randomizes by session instead of user, when CTR is computed over the whole list, or when considering interleaving or position bias.
---

# Testing Recommenders Online

## Overview

An offline gain is a hypothesis: it was scored on logs the **current** ranker produced. **Core principle:** only randomized exposure measures impact — randomized by **user**, on a business metric, with clicks compared **at equal positions** — and the launch is a ramp with a decision rule written before the data.

## Protocol

1. **No direct launch to 100%.** There is no control after it. Ramp: ~5% canary (latency, errors, SRM), then 50/50 for the planned duration, then 95% with a 5% holdback for weeks. Deadlines change the ramp calendar, not the existence of a control. Say so in the reply, and offer what fits the date: "offline result, live test starts Monday".
2. **Unit = user** (stable id, device id as fallback), never session. One user's sessions are correlated and see both rankers; session units give CIs too narrow and contaminated arms. Hash `salt:user_id`, new salt per experiment (`assign_variant`). Aggregate to one row per user before any test.
3. **Metric hierarchy, fixed before launch.** Primary = business outcome per user (conversion, revenue, retention, watch time). CTR is a **proxy**; use it only as secondary diagnostic. Guardrails: zero-click session rate, latency p95, diversity/coverage.
4. **Never compare whole-list CTR.** `clicks/impressions` changes with list length, layout and what the ranker puts on top: a new ranker that shows 5 items instead of 10 raises CTR with identical relevance. Compare **CTR per position** (`compute_ctr_by_position`) and CTR@k for a fixed k; for position-debiased lift use clicks weighted by logged examination propensity (inverse propensity), which requires logging position at impression time.
5. **Ratio metrics** (CTR, revenue/session) on user rows use the delta method (`compute_ratio_test`), not a two-proportion z-test over impressions.
6. **Interleaving = fast sensitivity screen, not the verdict.** Team-draft interleaving (`interleave_team_draft`) cancels position bias by construction and needs far less traffic than an A/B to rank two rankers; run it first to prune candidates or confirm direction. It measures preference, not business impact: the A/B on the primary metric still decides the launch. Score wins per user, test with a binomial test, drop ties.
7. **Plan and checks follow** REQUIRED SUB-SKILL: designing-ab-tests (MDE, sample size, SRM, fixed horizon or sequential boundary, FDR on secondaries).
8. **Decision rule up front:** ship if the primary-metric 95% CI excludes zero and no guardrail degrades; flat → iterate; negative → roll back. Report the offline number next to the online one (REQUIRED SUB-SKILL: evaluating-recommenders-offline).

## Implementation

Full tested helpers: `testing-recsys-online_template.py` (`assign_variant`, `interleave_team_draft`, `evaluate_interleaving`, `compute_ratio_test`, `compute_ctr_by_position`).

Measured on synthetic data (pandas 3.x, numpy, scipy 1.x; 20,000 users):

| Check | Result |
|---|---|
| Two **identical** rankers, treatment shows 5 items vs 10 | Whole-list CTR 2.15% → 3.09% (+44%, p = 5e-57); CTR at positions 1–5 identical within noise |
| Interleaving, A noisy vs B good (3,000 users) | B wins 59% of decisive users, p = 4e-16 |
| Interleaving A/A | 50.8% wins, p = 0.49 |
| `assign_variant` | 49.9% treatment at share 0.5; raising 0.5 → 0.7 moves no user back |

## Common Mistakes

| Mistake | Fix |
|---|---|
| Offline +3% NDCG shipped at 100% | Ramp with 5% holdback; the holdback is the control |
| Randomize by session id "because it is handy" | Hash user id; sessions only as a fallback, flagged as a limitation |
| Success = CTR over the whole list | Business metric primary; CTR only per position or @k |
| Position not logged at impression time | Log `position` and, if possible, examination propensity from day one |
| Interleaving win declared as "launch" | It screens; the A/B on the primary metric decides |
| Two-proportion z-test on impressions | User-level delta method |

## Rationalizations

| Excuse | Reality |
|---|---|
| "Offline wins always transfer" | Offline logs come from the old ranker; position and selection bias make the transfer an empirical question |
| "4 months invested, demo Friday" | Cost already spent does not change the evidence; the ramp starts Monday and is compatible with the demo |
| "Session ids are all we have" | Fall back with a flag, or log a user/device id first; do not present the CI as exact |
| "CTR is what the VP asked for" | Provide it as secondary and per position; put the business metric as primary |
| "No lectures, just do what I asked" | Deliver the requested artifacts in the corrected form (ramp, user unit, per-position CTR) and state each change in one line with its reason |
| "Interleaving already showed B wins" | Preference screen only; confirm on the primary metric |

## Red Flags — Stop

- "Launch to 100%" with no holdback or decision date
- Assignment keyed on `session_id`
- One CTR number for the entire list, no per-position view
- No SRM check before reading results
- Position not in the impression log
