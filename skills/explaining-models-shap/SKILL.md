---
name: explaining-models-shap
description: Use when explaining model predictions or feature importance with SHAP, LIME, permutation importance, or feature_importances_, when stakeholders ask "why" customers churn/default/buy based on a model, or when features are correlated.
---

# Explaining Models with SHAP

## Overview

SHAP answers "how did **this model** use each feature to produce **this prediction**". It does not answer "what **causes** the outcome" or "what happens if we change X". **Core principle:** report explanations as model behavior, cross-checked with permutation importance on held-out data, with correlated features grouped, and never turn them into causal claims or invented numbers.

## Which Tool

| Question | Tool | Cost / caveat |
|---|---|---|
| Per-prediction attribution, tree model | `shap.TreeExplainer` | Exact, fast (polynomial in tree depth); default output in log-odds for XGBoost/LightGBM |
| Per-prediction attribution, any model | `shap.KernelExplainer` / `shap.Explainer(f, masker)` | Model-agnostic, ~`n_samples × n_background` model calls: background ≤ 100 rows (`shap.kmeans`), explain a sample |
| Linear model | `shap.LinearExplainer` | Exact; equals coefficient × centered value |
| Global importance for prediction quality | `sklearn.inspection.permutation_importance` on **validation** | Drop in the metric you care about; uses unseen data |
| Do **not** use | `feature_importances_` (impurity / gain) | Computed on training data; inflated for high-cardinality and continuous features |

## Rules

1. **Explain on validation/test data**, never on training rows only.
2. **Group correlated features** (|ρ| > ~0.7, or one derived from another, e.g. `gasto_total ≈ meses_cliente × mensalidade`): SHAP splits credit between them arbitrarily and per-feature permutation underestimates both. Sum SHAP values of the group and permute the group jointly.
3. **Language:** "the model associates longer tenure with lower predicted churn", never "tenure reduces churn" or "support calls cause cancellation". Causal claims need an experiment (A/B test) or a causal design.
4. **No numbers without output.** If the analysis did not run on real data, the stakeholder text contains `TODO` placeholders, not illustrative figures.
5. **State the limitations** in the same deliverable (template below). Record them in the model card (REQUIRED SUB-SKILL: writing-model-cards-datasheets); check whether sensitive attributes or proxies (region, age) drive predictions (REQUIRED SUB-SKILL: auditing-fairness).

## Implementation

```python
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score


def compute_grouped_importance(
    model: object, X_val: pd.DataFrame, y_val: pd.Series, groups: dict[str, list[str]], seed: int = 42
) -> pd.DataFrame:
    """Compara |SHAP| médio e queda de AUC por permutação, somando grupos correlacionados."""
    shap_values = shap.TreeExplainer(model)(X_val).values  # validação, não treino
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]
    mean_abs = pd.Series(np.abs(shap_values).mean(axis=0), index=X_val.columns)

    rng = np.random.default_rng(seed)
    base = permutation_importance(model, X_val, y_val, scoring="roc_auc", n_repeats=10, random_state=seed)
    perm = pd.Series(base.importances_mean, index=X_val.columns)

    rows = []
    grouped = {c for cols in groups.values() for c in cols}
    for name, cols in groups.items():  # permutação conjunta do grupo
        base_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
        drops = []
        for _ in range(10):
            X_perm = X_val.copy()
            X_perm[cols] = X_val[cols].to_numpy()[rng.permutation(len(X_val))]
            drops.append(base_auc - roc_auc_score(y_val, model.predict_proba(X_perm)[:, 1]))
        rows.append({"feature": name, "mean_abs_shap": mean_abs[cols].sum(), "perm_auc_drop": float(np.mean(drops))})
    for col in X_val.columns.difference(list(grouped)):
        rows.append({"feature": col, "mean_abs_shap": mean_abs[col], "perm_auc_drop": perm[col]})
    return pd.DataFrame(rows).sort_values("perm_auc_drop", ascending=False, ignore_index=True)
```

## Stakeholder Text Template

> O modelo usa principalmente **[grupo 1]** e **[feature 2]** para estimar o risco de cancelamento (queda de AUC de [TODO] e [TODO] quando embaralhadas). Isso descreve **como o modelo prevê**, não **o que causa** o cancelamento: clientes com mais chamados podem cancelar por um problema que também gera chamados. Tempo de casa e gasto total medem quase a mesma coisa e foram analisados juntos. Para saber se uma ação (ex.: reduzir tempo de atendimento) reduz cancelamentos, a próxima etapa é um teste controlado.

## Common Mistakes

| Mistake | Fix |
|---|---|
| "Factors that cause churn" from SHAP | "Factors the model uses"; recommend an experiment for causal questions |
| `model.feature_importances_` as the ranking | Permutation importance on validation + SHAP |
| Ranking two correlated features separately | Group them; sum SHAP; permute jointly |
| Illustrative numbers in the executive text | `TODO` until the analysis runs on real data |
| `KernelExplainer` on full dataset | Background ≤ 100 rows, explain a sample, or use `TreeExplainer` |
| SHAP on training data | Validation/test rows |
