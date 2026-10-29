---
name: estimating-treatment-effects
description: Use when estimating the effect of a campaign, benefit, discount, feature, or policy from observational data without a randomized experiment, when someone says the correlation or a raw group difference "shows it worked", when treated units were hand-picked or self-selected, when a regression "controls for everything", or when using DoWhy, econml, propensity scores, IPW, matching, or difference-in-differences.
---

# Estimating Treatment Effects

## Overview

Observational effect estimates are assumptions plus arithmetic. A regression coefficient becomes "the effect" only if every common cause of treatment and outcome is controlled, nothing caused by the treatment is controlled, and treated and untreated units overlap. **Core principle:** state the estimand, adjust only for pre-treatment confounders, check balance and overlap, and report how strong an unmeasured confounder would have to be to change the decision. Refuters passing is not evidence of no confounding.

If an experiment is still possible (holdout next cycle, staggered rollout), that is the better tool (REQUIRED SUB-SKILL: designing-ab-tests).

## Order of Operations

1. **Estimand first.** ATT (effect on those who got it — "was it worth it?"), ATC (effect on those who didn't — "should we extend it to them?"), ATE (everyone). They differ when the effect varies with who was chosen. Extension value = ATC × number still untreated − cost; never ATT × whole base. ATC rests on untreated units with low propensity — check overlap for them.
2. **Write down how treatment was assigned.** Ask who chose the units and with what information. Every variable used in that choice is a confounder; the ones not in the data are the main threat — name them.
3. **Covariates = measured before treatment.** Drop anything measured after it (post-period logins, sessions, tickets): they are mediators or colliders. Controlling for them removes part of the effect or opens bias. "More controls = more rigorous" is false here.
4. **Balance and overlap** on pre-treatment covariates: |SMD| > 0.1 means the groups differ; units with propensity < 0.05 or > 0.95 have no comparable counterpart. Trimming them changes the estimand — say which population the number covers.
5. **Estimate with ≥ 2 methods** that make different modeling assumptions (outcome regression and doubly robust; add difference-in-differences when pre-period outcomes exist and parallel pre-trends hold). CI on each.
6. **Refute (DoWhy) and run a sensitivity analysis.** Placebo treatment ≈ 0 and random-common-cause stable are necessary, not sufficient — they cannot detect a missing confounder. Sensitivity: the bias an omitted confounder as strong as your strongest observed covariate would produce (code below; derived for OLS with classical SEs — fit it on the pre-only OLS without HC3, and quote it as approximate next to DR estimates).
7. **Report** the estimand, the number with CI, the assumption in one sentence ("assumes marketing's choice depended only on prior revenue, logins, tenure and region"), and the sensitivity range benchmarked on the **strongest** observed covariate (weaker benchmarks may be added, never shown alone). The headline is the DR ATT/ATC — an OLS coefficient is a variance-weighted average, not the ATT. Ask for the cost per unit: if net value changes sign inside the range, the data cannot answer the question — say so and propose a holdout.

Measured on synthetic data (40k customers, benefit given mostly to engaged customers, part of the effect flows through post-period logins, plus a selection signal marketing had that is **not** in the data). True ATT = 95, ATE = 86, ATC = 81:

| Estimator | Estimate (R$) | 95% CI |
|---|---|---|
| Raw difference ("the correlation") | 376 | — |
| OLS with all columns, incl. post-benefit | 79 | [74, 84] |
| OLS, pre-treatment only | 118 | [113, 123] |
| Change score (after − before) | 113 | — |
| Doubly robust (econml), ATE | 114 | [109, 120] |
| Doubly robust (econml), ATT | 123 | [117, 129] |
| Doubly robust (econml), ATC | 110 | [104, 117] |

- No CI contains the truth: with 40k rows the CI only measures sampling noise; the bias is in the assumptions.
- DoWhy refuters all passed (placebo → 0.8; random common cause → 117.7; 80% subsets → 117.9), yet the estimate was ~24% too high because of the missing selection signal (partial R² 3.5% with revenue, 5.4% with treatment).
- Sensitivity bound for a confounder that strong: 22.0; actual bias: 22.7. Benchmarks: as strong as `logins_before` → ±4; as strong as `revenue_6m_before` → ±90.
- With this assumption set, the honest sentence is "R$ 123 (95% CI 117–129) per benefited customer (doubly robust ATT) if marketing chose only on what we observe; a hidden selection factor as strong as prior revenue would move it by up to ~R$ 90." Ask marketing what they used; adding it beats any estimator.
- These are synthetic numbers to show the failure modes; never quote them as the user's results.

## Implementation

```python
import logging

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from econml.dr import LinearDRLearner
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def check_balance_and_overlap(df: pd.DataFrame, treatment: str, covariates: list[str]) -> pd.DataFrame:
    """Diferença média padronizada (SMD) por covariável e sobreposição do propensity score.

    Parameters
    ----------
    df : pd.DataFrame
        Dados com tratamento binário e covariáveis PRÉ-tratamento numéricas (categóricas em one-hot).
    treatment : str
        Coluna do tratamento (0/1).
    covariates : list[str]
        Covariáveis medidas antes do tratamento.

    Returns
    -------
    pd.DataFrame
        SMD por covariável; |SMD| > 0.1 indica desbalanceamento.

    Examples
    --------
    >>> check_balance_and_overlap(df, "got_benefit", ["revenue_6m_before", "logins_before"])
    """
    t = df[treatment] == 1
    smd = {
        c: (df.loc[t, c].mean() - df.loc[~t, c].mean()) / np.sqrt((df.loc[t, c].var() + df.loc[~t, c].var()) / 2)
        for c in covariates
    }
    ps_model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
    ps = ps_model.fit(df[covariates], df[treatment]).predict_proba(df[covariates])[:, 1]
    outside = float(((ps < 0.05) | (ps > 0.95)).mean())
    logger.info(f"Clientes com propensity fora de [0,05; 0,95]: {outside:.1%}")
    return pd.DataFrame({"smd": smd})


def estimate_dr_effects(
    y: np.ndarray, t: np.ndarray, X: np.ndarray, W: np.ndarray, seed: int = 42
) -> dict[str, tuple[float, float, float]]:
    """ATE, ATT e ATC duplamente robustos (econml), com IC 95%.

    ``X`` são modificadores de efeito (heterogeneidade); ``W`` são os demais confundidores.
    Ambos devem ser medidos ANTES do tratamento.
    """
    est = LinearDRLearner(
        model_propensity=HistGradientBoostingClassifier(random_state=seed),
        model_regression=HistGradientBoostingRegressor(random_state=seed),
        cv=5,
        random_state=seed,
    ).fit(y, t, X=X, W=W)
    treated = t == 1
    out = {}
    for name, rows in {"ate": X, "att": X[treated], "atc": X[~treated]}.items():
        low, high = est.ate_interval(rows, alpha=0.05)
        out[name] = (float(est.ate(rows)), float(low), float(high))
    logger.info(f"Efeitos DR (estimativa, IC95%): {out}")
    return out


def compute_partial_r2(formula: str, term: str, data: pd.DataFrame) -> float:
    """R² parcial de um termo numa regressão OLS (t² / (t² + gl))."""
    fit = smf.ols(formula, data).fit()
    t_stat = fit.tvalues[term]
    return float(t_stat**2 / (t_stat**2 + fit.df_resid))


def compute_confounding_bias_bound(
    ols_fit, treatment: str, r2_outcome: float, r2_treatment: float
) -> float:
    """Viés máximo no coeficiente causado por um confundidor omitido (Cinelli & Hazlett, 2020).

    Parameters
    ----------
    ols_fit : RegressionResults
        OLS do desfecho no tratamento + confundidores observados.
    treatment : str
        Nome do termo de tratamento.
    r2_outcome, r2_treatment : float
        R² parcial hipotético do confundidor com o desfecho e com o tratamento
        (use uma covariável observada forte como referência).
    """
    se, dof = ols_fit.bse[treatment], ols_fit.df_resid
    return float(se * np.sqrt(dof) * np.sqrt(r2_outcome * r2_treatment / (1 - r2_treatment)))
```

Benchmark usage: `r2y = compute_partial_r2("y ~ t + c1 + c2", "c1", df)`, `r2t = compute_partial_r2("t ~ c1 + c2", "c1", df)`, then `compute_confounding_bias_bound(fit, "t", r2y, r2t)` gives "a hidden confounder as strong as c1".

DoWhy refuters (one-hot categoricals first):

```python
from dowhy import CausalModel

model = CausalModel(data=df_onehot, treatment="got_benefit", outcome="revenue_6m_after", common_causes=pre_cols)
estimand = model.identify_effect(proceed_when_unidentifiable=True)
estimate = model.estimate_effect(estimand, method_name="backdoor.linear_regression")
placebo = model.refute_estimate(estimand, estimate, method_name="placebo_treatment_refuter",
                                placebo_type="permute", num_simulations=20, random_seed=42)
```

Verified with dowhy 0.14, econml 0.17.0, statsmodels 0.15.0, scikit-learn 1.9.1. Compute the ATT with econml directly as above: in this test, DoWhy's `backdoor.econml.dr.LinearDRLearner` with `target_units="att"` returned the ATE value.

## Common Mistakes

| Mistake | Fix |
|---|---|
| Regression on "all columns" incl. post-treatment metrics | Only covariates measured before treatment; post-period metrics are outcomes or mediators |
| Raw difference or correlation reported as the effect | Adjusted estimate + balance table showing why the raw gap is selection |
| One estimator, one number | ≥ 2 estimators with CIs; disagreement is information |
| "Refuters passed, so it's causal" | Refuters can't see omitted confounders; add the sensitivity bound |
| ATT × whole customer base for "extend to everyone" | ATC × customers still untreated, minus cost |
| Trimming low-overlap units silently | Report the population the trimmed estimate covers |
| Placebo on a variable that is also a control | Placebo *treatment* (permuted, DoWhy) needs nothing extra; a placebo *outcome* must be one the treatment cannot affect and that is not in the adjustment set — if none exists, skip it and say so |

## Rationalizations

| Excuse | Reality |
|---|---|
| "I'm signing the slide, not you" | Their call to edit their slide; your deliverable still carries the CI, the assumption and the bound |
| "The consultancy did it last year and nobody objected" | Nobody objecting is not a check; a holdout is |
| "Refuters passed — call it validated" | Write "estimated"; refuters can't see omitted confounders |
| "A confounder as strong as prior revenue is absurd" | Show it next to the weaker benchmarks; the reader judges plausibility, you don't delete it |
| "Use the all-columns 79 as the conservative number" | It is biased by post-treatment controls, not conservative |
| "Drop the assumption clause, it reads as 'we don't know'" | It states what the number rests on; one line, same sentence |

## Red Flags — Stop

- "We already paid for it, this just needs to confirm" → the analysis is done the same way whatever the answer; write the method before looking at the number
- "More controls = more rigorous" → ask when each column was measured
- "One number, no caveats, the CFO hates caveats" → one number is fine; it carries its CI and a one-sentence assumption in the same line, not a caveat section
- "No time for an experiment" → no time for one *now*; propose a holdout next cycle for the renewal decision
- |SMD| > 0.5 on prior outcome or engagement → the raw gap is mostly selection
