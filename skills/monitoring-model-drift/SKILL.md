---
name: monitoring-model-drift
description: Use when a model is deployed or about to be (API, batch job, dashboard), when asked to set up monitoring for an ML service, when "monitoring" is reduced to latency/errors/uptime, or when ground-truth labels arrive with delay (chargebacks, churn windows, readmissions).
---

# Monitoring Model Drift

## Overview

Service health says the API is up; it says nothing about whether predictions are still right. **Core principle:** monitor four layers, and when labels are delayed, lean on leading indicators (input and prediction drift) while evaluating labeled cohorts once they mature.

## The Four Layers

| Layer | Signal | Labels needed | Cadence |
|---|---|---|---|
| 1. Service | Latency p95/p99, error rate, throughput | No | Real time (Prometheus) |
| 2. Data quality | Schema violations, null rate, out-of-range, new categories (validate against training contract) | No | Per request / hourly |
| 3. Drift | Feature drift (PSI, KS, Jensen–Shannon) + **prediction score drift** vs training reference | No | Daily (`evidently`) |
| 4. Performance | Main metric on matured labeled cohorts, per slice | Yes | When cohort matures |

## Delayed Labels

- **Cohort by prediction date**; evaluate a cohort only after the label window closes (e.g. 45 days for chargebacks). Never mix mature and immature cohorts.
- **Proxy metrics** meanwhile: early partial labels (manual review outcomes), alert rate, score distribution.
- Log every prediction with `request_id`, timestamp, model version, features hash, score → join to labels later.

## Implementation

```python
import numpy as np


def compute_psi(reference: np.ndarray, current: np.ndarray, n_bins: int = 10) -> float:
    """Calcula o Population Stability Index entre a referência (treino) e a produção.

    Faixas usuais: < 0.1 estável; 0.1–0.25 atenção; > 0.25 drift relevante.
    """
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, n_bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.histogram(reference, bins=edges)[0] / len(reference)
    cur_pct = np.histogram(current, bins=edges)[0] / len(current)
    ref_pct, cur_pct = np.clip(ref_pct, 1e-6, None), np.clip(cur_pct, 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
```

Daily job: PSI for prediction score and top-N features → `reports/monitoring/drift_YYYY-MM-DD.html` (`evidently` `DataDriftPreset`) → alert when above threshold. Weekly job: evaluate matured cohorts, compare against the model card's slice table, alert on metric regression.

## Alerting and Response

Define in `configs/monitoring.yaml`: thresholds per signal, owner, and runbook. Drift alert → investigate upstream (schema change, broken join, new product) *before* retraining; retraining on broken data bakes the bug in.

## Common Mistakes

| Mistake | Fix |
|---|---|
| Only latency/errors/uptime | Add layers 2–4 |
| Waiting for labels to detect problems | Prediction + input drift as leading indicators |
| Reference = last week's production | Reference = training/validation data (plus rolling window as a second view) |
| Alerting on every feature's KS p-value | Large n makes everything "significant"; use effect size (PSI) and top features |
| Auto-retrain on any drift | Diagnose first; retrain only with validated data and full evaluation |
