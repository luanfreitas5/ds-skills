---
name: labeling-images-with-quality-control
description: Use when images are labeled by hand (Label Studio) or by a model, when one annotator labels a whole dataset, when there is no labeling guide or double-labeled sample, when asked about inter-annotator agreement or label noise, or when a classifier is trained and tested on the same unverified labels.
---

# Labeling Images with Quality Control

## Overview

Labels are measurements, and a measurement with unknown error cannot grade a model. One annotator, no guide, no overlap means the noise rate is unknown, and a holdout carved from the same labels cannot expose it: the model is scored on how well it copies the annotator's mistakes. **Core principle:** measure label quality before training, clean the evaluation set, and report accuracy against adjudicated labels, not raw ones.

**Violating the letter is violating the spirit.** "Accuracy vs. Marie's labels" in a slide does not replace measuring Marie.

Verified with cleanlab 2.9.0, scikit-learn 1.9.1 on synthetic data. Label Studio config is not run here; check its docs for the current export schema.

## Gate Before Training

1. **Write the label guide first** (1 page): per class a definition, 2 positive and 2 borderline example image IDs, and an explicit `unsure`/`none_of_the_above` option. Version it, e.g. `guide_v1`.
2. **Double-label a sample.** A second person labels 300 random images (stratified by class), blind to the first. Cost: about 2-3 hours. Never skip because there is "one annotator"; the same person relabeling blind after 2+ weeks is the minimum fallback and gives intra-annotator kappa only.
3. **Measure.** Cohen's kappa with a 95% bootstrap CI and per-class confusions (`evaluate_annotator_agreement` below). Gate: kappa >= 0.7 for the training labels, and no class pair dominating disagreements. Below that, fix the guide, relabel the disputed classes, and measure again.
4. **Adjudicate** disagreements and cleanlab-flagged items; a senior reviewer decides, with the guide. This produces the evaluation set.
5. **Split by group** (REQUIRED SUB-SKILL: splitting-image-datasets), then train on the cleaned or raw training labels, but evaluate only on the adjudicated test set.

## Finding Label Errors (confident learning)

Use out-of-fold predicted probabilities from any model (embeddings from a pretrained backbone plus logistic regression is cheap). Never use in-sample probabilities: the model memorizes the noise and flags nothing.

| Step | Detail |
|---|---|
| Probabilities | `cross_val_predict(..., cv=5, method="predict_proba")`, grouped folds if images share a group |
| Flag | `cleanlab.filter.find_label_issues(labels, probs, return_indices_ranked_by="self_confidence")` |
| Review | Humans review the top-ranked flags (the ranked order lets a budget cut off). Flags are suspects, not verdicts |
| Report | Estimated noise rate, flags reviewed, fraction confirmed |

## Core Code

```python
import logging

import numpy as np
from cleanlab.filter import find_label_issues
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, recall_score
from sklearn.model_selection import cross_val_predict

logger = logging.getLogger(__name__)


def select_double_label_sample(ids: np.ndarray, labels: np.ndarray, n: int = 300, seed: int = 42) -> np.ndarray:
    """Seleciona amostra estratificada por classe para dupla anotação cega.

    Parameters
    ----------
    ids : np.ndarray
        Identificadores das imagens.
    labels : np.ndarray
        Rótulos do primeiro anotador (usados só para estratificar).
    n : int, optional
        Tamanho total da amostra, by default 300.
    seed : int, optional
        Semente aleatória, by default 42.

    Returns
    -------
    np.ndarray
        Identificadores a enviar ao segundo anotador, sem mostrar o rótulo do primeiro.
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(labels)
    per_class = max(n // len(classes), 1)
    picked: list[int] = []
    for c in classes:
        pool = np.flatnonzero(labels == c)
        picked.extend(rng.choice(pool, min(per_class, len(pool)), replace=False))
    return np.asarray(ids)[np.sort(picked)]


def evaluate_annotator_agreement(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 42) -> dict[str, float]:
    """Kappa de Cohen entre dois anotadores, com IC 95% bootstrap."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n_boot, len(a)))
    boots = [cohen_kappa_score(a[i], b[i]) for i in idx]
    kappa = float(cohen_kappa_score(a, b))
    low, high = np.nanpercentile(boots, [2.5, 97.5])
    logger.info(f"Kappa entre anotadores: {kappa:.3f} (IC 95% {low:.3f}-{high:.3f}, n={len(a)})")
    return {"kappa": kappa, "ci_low": float(low), "ci_high": float(high), "accuracy": float((a == b).mean())}


def find_suspect_labels(features: np.ndarray, noisy_labels: np.ndarray, seed: int = 42) -> np.ndarray:
    """Retorna índices de rótulos suspeitos, do mais para o menos provável (confident learning)."""
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    probs = cross_val_predict(clf, features, noisy_labels, cv=5, method="predict_proba")
    issues = find_label_issues(noisy_labels, probs, return_indices_ranked_by="self_confidence")
    logger.info(f"{len(issues)} rótulos suspeitos de {len(noisy_labels)} ({len(issues) / len(noisy_labels):.1%})")
    return issues


def evaluate_on_adjudicated(y_pred: np.ndarray, y_raw: np.ndarray, y_adjudicated: np.ndarray) -> dict[str, object]:
    """Compara acurácia contra rótulos brutos e adjudicados; reporte os dois."""
    return {
        "acc_vs_raw": float((y_pred == y_raw).mean()),
        "acc_vs_adjudicated": float((y_pred == y_adjudicated).mean()),
        "recall_per_class": recall_score(y_adjudicated, y_pred, average=None).tolist(),
    }
```

Synthetic check (12% injected noise, 4 classes): kappa 0.76 (CI 0.71-0.82) on the 300-item overlap; cleanlab flagged 354 of 3000 with precision 1.00 and recall 0.99; a perfect model scored 0.88 against raw labels and 1.00 against adjudicated. That 12-point ceiling is invisible without the second column.

Label Studio: put the guide text in the project instructions, include an `unsure` choice, give each annotator their own account so overlap tasks can be exported per user, and keep `task id` as the join key. Store exports under `data/bronze/` with a dated snapshot and hash. Model pre-labels shown to annotators bias them (anchoring): hide them for the double-labeled sample. For model-generated labels, REQUIRED SUB-SKILL: labeling-with-llms. Document guide, kappa, noise estimate in the datasheet: REQUIRED SUB-SKILL: writing-model-cards-datasheets.

## Rationalizations

| Excuse | Reality |
|---|---|
| "The VP says the labels are ground truth" | Authority does not measure quality. Report kappa and noise estimate next to the instruction |
| "120 hours are already spent" | Sunk cost. 3 more hours of double-labeling protects all 120 |
| "Demo is tomorrow" | The overlap sample and cleanlab run fit in an afternoon; ship the script with the audit, report `pending_adjudication` if needed |
| "No budget for more annotators" | One person for 300 images, or the same annotator blind after 2 weeks. Free alternative: cleanlab review of the top flags |
| "No guide, she used judgment" | Judgment is unreproducible. A 1-page guide written now fixes the next batch and explains disagreements |
| "Hold out 20% from the same labels is standard" | Standard only if test labels are trustworthy. Adjudicate the test set |
| "I will present it as agreement with her labels" | Disclosure is not measurement. Still measure |
| "Train on all, flag with in-sample probs" | Memorized noise flags nothing. Out-of-fold only |
| "There is no senior reviewer to adjudicate" | The second annotator plus the guide decide ties; if still unresolved, the number is labeled `pending_adjudication`, never "accuracy" |
| "Model pre-labels speed it up" | Anchoring. Blind for the audit sample |

## Red Flags

- One annotator, no overlap, no kappa with CI
- No label guide or no `unsure` class
- Test labels never reviewed by a second person
- cleanlab run on in-sample probabilities
- Accuracy reported against raw labels only
- Pre-labels visible during the double-label sample
