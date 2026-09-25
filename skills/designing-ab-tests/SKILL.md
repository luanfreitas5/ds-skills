---
name: designing-ab-tests
description: Use when planning, monitoring, or analyzing an A/B test or online experiment, when someone checks the p-value daily and wants to stop early, when groups have unequal sizes, when many metrics are compared, or when choosing test duration or sample size.
---

# Designing A/B Tests

## Overview

The p-value is valid only for the analysis planned before the test: one look, at the planned sample size, on one primary metric, with a correct randomization. Daily peeking, stopping at the first p < 0.05, testing 12 metrics, and ignoring an unbalanced split each break it. **Core principle:** fix the plan before launch (primary metric, MDE, sample size, duration, stopping rule), check the split before reading any result, and analyze exactly as planned.

## Before Launch (write it down)

1. **Primary metric** (one) + guardrails (margin, latency, cancellations). Everything else is exploratory.
2. **MDE**: smallest lift worth shipping, from the business case — not from what the traffic allows.
3. **Sample size** for α = 0.05, power 0.8 (code below). Duration = sample / daily traffic, rounded **up to whole weeks** (weekday effects).
4. **Stopping rule**: fixed horizon (one look at the end) **or** a group-sequential design with the number of looks and an alpha-spending boundary (O'Brien–Fleming) chosen now.
5. **Variance reduction**: CUPED with a pre-period covariate cuts the needed sample when the metric correlates with pre-period behavior.

## While Running

- **SRM check first**, daily: χ² of observed counts vs planned split. p < 0.001 → stop reading results; the randomization or logging is broken; find the cause (bot filtering, redirects, assignment bugs).
- Looking is fine; **deciding** is only allowed at planned looks with the planned boundary.

## Analysis

| Situation | Method |
|---|---|
| Conversion (proportion) | Two-proportion z-test / `proportions_ztest`; report absolute and relative lift with 95% CI |
| Continuous, heavy-tailed (revenue) | Welch t-test on CUPED-adjusted metric, or bootstrap CI |
| Many secondary metrics | Benjamini–Hochberg (FDR) across them; primary metric uncorrected only because it was pre-registered |
| Interim look | Compare z to the sequential boundary for that information fraction, not to 1.96 |

## Implementation

```python
import numpy as np
from scipy.stats import chisquare
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import NormalIndPower
from statsmodels.stats.proportion import proportion_effectsize


def compute_sample_size(baseline_rate: float, relative_mde: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Tamanho de amostra por grupo para detectar o lift relativo mínimo."""
    effect = proportion_effectsize(baseline_rate * (1 + relative_mde), baseline_rate)
    return int(np.ceil(NormalIndPower().solve_power(effect_size=effect, alpha=alpha, power=power)))


def check_sample_ratio(n_control: int, n_treatment: int, expected_control_share: float = 0.5) -> float:
    """p-valor do teste de SRM; p < 0.001 invalida a leitura do experimento."""
    total = n_control + n_treatment
    expected = [total * expected_control_share, total * (1 - expected_control_share)]
    return float(chisquare([n_control, n_treatment], f_exp=expected).pvalue)


def apply_cuped(y: np.ndarray, x_pre: np.ndarray) -> np.ndarray:
    """Reduz variância usando a covariável do pré-período (mesmo theta para os dois grupos)."""
    theta = np.cov(x_pre, y, ddof=1)[0, 1] / np.var(x_pre, ddof=1)
    return y - theta * (x_pre - x_pre.mean())


def correct_secondary_metrics(p_values: dict[str, float], fdr: float = 0.05) -> dict[str, bool]:
    """Benjamini–Hochberg sobre as métricas secundárias."""
    reject, _, _, _ = multipletests(list(p_values.values()), alpha=fdr, method="fdr_bh")
    return dict(zip(p_values, reject.tolist()))
```

Worked check: control 50,912 vs treatment 49,021 on a planned 50/50 split → `check_sample_ratio` p ≈ 2e-9: SRM; the conversion p = 0.041 cannot be interpreted. For baseline 4.1% and MDE +8% relative, `compute_sample_size` ≈ 60k per group.

## Common Mistakes

| Mistake | Fix |
|---|---|
| Stop the day p < 0.05 appears | Fixed horizon, or planned sequential boundary |
| Reading lift with unequal group sizes | SRM check before any metric |
| "2 of 12 metrics significant" as extra evidence | BH correction; ~46% chance of ≥ 1 false positive at raw α across 12 |
| Duration "until significant" | Sample size from MDE and power, whole weeks |
| MDE chosen to fit available traffic | MDE from business value; if traffic is too low, say the test cannot detect it |
| Only the p-value reported | Lift with 95% CI, absolute and relative (REQUIRED SUB-SKILL: evaluating-models-rigorously) |
