---
name: building-multimodal-pipelines
description: Use when using CLIP or SigLIP for zero-shot image classification or text-to-image search, when a vision-language model (GPT-4o vision, Gemini, Qwen-VL) labels images, when class names are used directly as prompts, or when zero-shot is declared "validated" after a handful of hand-picked examples.
---

# Building Multimodal Pipelines

## Overview

Zero-shot image-text models are classifiers whose accuracy you have not measured. A few hand-picked photos prove the demo works, not that 300k photos are right: 10/10 correct still allows true accuracy near 72% (Wilson 95%). A VLM that labels images produces predictions, not ground truth, and the hardest images (low zero-shot confidence) are exactly where it errs most. **Core principle:** nothing is "validated" until a random, human-labeled sample reports accuracy with a CI per class. Prompts are hyperparameters chosen on a dev split and frozen before the test split. VLM labels stay `unverified` until a blind audit passes a gate.

**Violating the letter is violating the spirit.** "CEO approved the labels" is a decision, not a measurement. Add the audit next to the decision.

Verified on: transformers 5.22.1, faiss-cpu 1.15.1, torch 2.15 (SigLIP API exercised on a randomly initialized tiny config; no weights downloaded).

## Evaluation Sets (build before tuning)

| Set | Rule |
|---|---|
| Zero-shot test | Random, labeled by humans blind to the model, ≥ 50 per class. Labels do not exist yet, so stratify the draw by zero-shot predicted class (plus a plain random slice) or a plain random 600 leaves rare classes with n < 10 Never hand-picked, never reused to choose prompts or thresholds |
| Dev split | Separate random sample, used only to pick templates, class synonyms, and the confidence threshold. Report test only after freezing |
| Retrieval gold | Real search queries (logs, SME-written) paired with the correct image ids. Tag generated queries `synthetic` and report apart |
| Out-of-catalog slice | ≥ 10% images belonging to **no** class (accessories, packaging, blurry). Gold = abstain |
| Slices | Report accuracy per class, per source/seller, per image quality. Aggregates hide dead classes |

## Zero-shot Rules

- **Class prompts are hypotheses.** Compare ≥ 2 template sets and synonyms on the dev split (`"a photo of a {}"`, `"a product photo of {} on white background"`, translated vs original names). Freeze the winner, then test once. Portuguese names on an English-trained CLIP often fail, so test translation both ways. For multilingual needs prefer a multilingual SigLIP checkpoint.
- **SigLIP is sigmoid, CLIP is softmax.** Softmax forces a winner among 12 classes, so out-of-catalog photos get a confident wrong label. SigLIP probabilities are independent and allow abstention. Use `logit_scale` and `logit_bias` from the model. Tokenize text with `padding="max_length", max_length=64` (as trained).
- **Confidence needs calibration.** Report precision per confidence bin on the dev split and pick the abstain threshold there. REQUIRED SUB-SKILL: checking-probability-calibration.
- **Compare to a baseline** (linear probe on 50–100 labeled images/class, or majority class) with a paired test on the same items. REQUIRED SUB-SKILL: establishing-baselines.

## VLM as Labeler

- **Treat output as predictions.** Write `label_pseudo`, `label_source`, `prompt_version`, `model` snapshot, `label_status='unverified'`. Never into the ground-truth column.
- **Blind audit before promotion.** Stratify ≥ 100 images per VLM label (more for rare classes) plus a random slice, human label blind to the VLM, report κ and accuracy with 95% bootstrap CI. Promote to `verified` only if κ ≥ 0.6 and no class recall < 0.5. Sampling only the low-confidence images makes the audit overstate errors; add a random slice for the prevalence-weighted view.
- **Cap and price first.** Run 200-500 images, project cost for the full set, then decide. Temperature 0, pinned model snapshot, allow `nenhuma`.
- **Photos leave the machine.** Faces, plates, and addresses are personal data. REQUIRED SUB-SKILL: protecting-pii-lgpd. Prompt, versioning, and agreement mechanics: REQUIRED SUB-SKILL: labeling-with-llms; versioning: versioning-prompts.
- **Do not label train and test with the same VLM** and call the supervised model better than it. It inherits the VLM's mistakes.

## Pipeline Code

Embed once, reuse for classification and search.

```python
import logging
import faiss
import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score
from transformers import SiglipConfig, SiglipModel

logger = logging.getLogger(__name__)


def embed_images(model, pixel_values: torch.Tensor) -> np.ndarray:
    """Gera embeddings de imagem normalizados (L2), float32."""
    with torch.no_grad():
        f = model.get_image_features(pixel_values=pixel_values)
    f = getattr(f, "pooler_output", f)
    return torch.nn.functional.normalize(f, dim=-1).float().numpy()


def embed_texts(model, tokenizer, texts: list[str]) -> np.ndarray:
    """Embeddings de texto; SigLIP exige padding='max_length' (como no treino)."""
    inputs = tokenizer(texts, padding="max_length", max_length=64, truncation=True, return_tensors="pt")
    with torch.no_grad():
        f = model.get_text_features(input_ids=inputs["input_ids"])
    f = getattr(f, "pooler_output", f)
    return torch.nn.functional.normalize(f, dim=-1).float().numpy()


def predict_zero_shot(model, img: np.ndarray, txt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """SigLIP: sigmoid por classe (nao softmax); probs NAO somam 1 e permitem 'nenhuma'."""
    logits = img @ txt.T * model.logit_scale.exp().item() + model.logit_bias.item()
    probs = 1 / (1 + np.exp(-logits))
    return probs.argmax(1), probs.max(1)


def wilson_ci(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """IC de Wilson para uma proporcao (n pequeno, p perto de 1)."""
    p = hits / n
    c = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return float(c - h), float(c + h)
```

```python
def build_index(emb: np.ndarray) -> faiss.IndexFlatIP:
    """Indice exato por produto interno (= cosseno com vetores normalizados)."""
    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(np.ascontiguousarray(emb, dtype="float32"))
    return index


def evaluate_retrieval(index, query_emb: np.ndarray, gold: np.ndarray, k: int = 10) -> dict[str, float]:
    """recall@k e MRR em pares (consulta -> imagem gold) escritos por humanos."""
    _, ids = index.search(np.ascontiguousarray(query_emb, dtype="float32"), k)
    hit = ids == gold[:, None]
    rank = np.where(hit.any(1), hit.argmax(1) + 1, np.inf)
    return {f"recall@{k}": float(hit.any(1).mean()), "mrr": float((1 / rank).mean())}


def select_audit_sample(preds: np.ndarray, n_per_class: int, seed: int = 42) -> np.ndarray:
    """Amostra de auditoria estratificada pelo rotulo da VLM (>= n por classe)."""
    rng = np.random.default_rng(seed)
    idx = [rng.choice(np.flatnonzero(preds == c), min(n_per_class, (preds == c).sum()), replace=False)
           for c in np.unique(preds)]
    return np.concatenate(idx)


def evaluate_vlm_audit(human: np.ndarray, vlm: np.ndarray, n_boot: int = 1000, seed: int = 42) -> dict[str, float]:
    """Kappa humano x VLM com IC 95% bootstrap; gate de promocao a 'verified'."""
    rng = np.random.default_rng(seed)
    boots = [cohen_kappa_score(human[i], vlm[i]) for i in rng.integers(0, len(human), (n_boot, len(human)))]
    low, high = np.nanpercentile(boots, [2.5, 97.5])
    kappa = float(cohen_kappa_score(human, vlm))
    logger.info(f"Kappa humano x VLM: {kappa:.3f} (IC 95% {low:.3f}-{high:.3f}, n={len(human)})")
    return {"kappa": kappa, "ci_low": float(low), "ci_high": float(high), "accuracy": float((human == vlm).mean())}
```

Run on synthetic data only: `torch.randn` pixels and a randomly initialized `SiglipModel(SiglipConfig(...))` are enough to test shapes, normalization and metrics in pytest. FAISS: `IndexFlatIP` is exact and enough up to ~1M x 512 vectors; move to HNSW/IVF only after measuring latency, and re-measure recall@k against the flat index. Embeddings change per model and checkpoint, so store the checkpoint id with the index.

## Rationalizations

| Excuse | Reality |
|---|---|
| "10 of 10 right, so it is validated" | Hand-picked n=10 gives a Wilson lower bound ~72%. Draw ≥ 50 random per class |
| "Class names as prompts are fine" | Untested hyperparameter. Compare templates on a dev split, freeze, then test |
| "The CEO approved the labels as final" | Approval is not accuracy. Store `unverified`, audit 300, report κ with CI, then promote |
| "GPT-4o vision is better than CLIP on hard photos" | Hypothesis. Hard photos are where it errs most; audit them |
| "No time for human labeling" | 600 labels is 1-2 person-hours. Ship the audit CSV in the same script |
| "Tune the threshold on the same 600" | Optimistic. Threshold on dev, report on test |
| "Single script, no tests, given the deadline" | Ship `pytest` on the metric helpers with synthetic arrays; they are the part that decides what the board hears |
| "FAISS or a vector DB first" | Exact search is enough; measure recall@k and MRR on real queries first |

## Red Flags

- A "validated" claim from fewer than 50 random images per class, or from images the author chose
- `f"a photo of a {class_name}"` with no template comparison and no dev/test split
- Softmax over classes with no abstain option on a catalog containing other items
- VLM labels written into a ground-truth or training column, with no `label_status`
- No audit CSV, no κ/CI, no per-class breakdown, or no out-of-catalog slice
- Photos sent to an external API with no privacy review
- A search demo with no recall@k/MRR on real queries
