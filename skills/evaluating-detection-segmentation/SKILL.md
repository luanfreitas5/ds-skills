---
name: evaluating-detection-segmentation
description: Use when reporting results of an object detector or segmentation model (YOLO/ultralytics, Detectron2, Mask R-CNN, U-Net), when the headline is "accuracy" or pixel accuracy on images that are mostly background, when only mAP@0.5 is reported, when computing IoU, Dice or mAP with torchmetrics, pycocotools or model.val(), when a confidence threshold is picked by looking at the test set, or when one class is rare or mostly small objects.
---

# Evaluating Detection and Segmentation

## Overview

Images that are 95% background make "accuracy" a background detector's score, and a mean over classes hides the rare class that the client actually cares about. **Core principle:** report per-class AP (with counts and size slices) for detection, per-class IoU/Dice without background for segmentation, and an operating point whose threshold was chosen on data other than the test set, with a CI sized by the number of objects of the rare class.

## Evaluation Contract

Every detection/segmentation result you deliver has these parts, in order:

1. **Counts:** test images, objects per class, objects per class that are small (< 32×32 px, COCO), images with no objects.
2. **Per-class AP@0.5 and AP@0.5:0.95**, then mAP. The rare class leads the paragraph if it is the costly miss.
3. **Size slice:** AP@0.5 on small objects per class (COCO area ranges). "Mostly small" class → this is its real number.
4. **Operating point** (precision, recall, FP per image) at a threshold chosen on validation data — never on the test set. Threshold chosen from costs or a recall target agreed with the client (REQUIRED SUB-SKILL: handling-class-imbalance).
5. **Uncertainty:** recall of the rare class with a Clopper–Pearson CI from its object count; bootstrap over images for AP (REQUIRED SUB-SKILL: evaluating-models-rigorously).
6. **Segmentation:** IoU/Dice per class **excluding background**, per image on images that contain the class, plus image-level false-positive rate on empty images.
7. **No pixel accuracy as a result.** If a contract demands it, it goes in a footnote next to the all-background score, never in the headline or in bold.

If the test set is the only reviewed annotation: split it into two halves by image, choose the threshold on one, count TP/FP/GT on the other, swap, and pool the counts before computing metrics and CIs — and say so. Thresholds may differ per class when miss costs differ; each is chosen the same way. A new reviewed validation set is the fix for the next milestone.

## What the Numbers Hide

Synthetic inspection set: 400 test images, 800 corrosion boxes (large), 36 crack boxes (25 smaller than 32×32); simulated detector with size-dependent recall:

| Reported | Value | What it hides |
|---|---|---|
| Pixel accuracy | **98.6%** | Predicting "all background" scores 96.2% |
| mAP@0.5 | 0.51 | Corrosion AP@0.5 **0.90**, crack AP@0.5 **0.12** |
| COCO `map_small` | 0.03 | Crack AP@0.5 on small objects 0.13 |
| Crack recall at threshold chosen on test (0.40) | 0.33 | Threshold chosen on a validation set (0.50): **0.14** (5/36, 95% CI 0.05–0.29) |

- Over 30 repetitions, choosing the threshold on the test set overstated crack F1 by 0.04 on average (up to 0.09).
- With 36 cracks in test, 5 found gives a 95% CI of 0.05–0.29 — "the model finds X% of cracks" needs the interval.

Segmentation (torchmetrics 1.9.0 `DiceScore`, 8 masks, lesion ≈ 5% of pixels, 4 empty masks):

| Setting | Dice |
|---|---|
| Predict all background, `include_background=True` (the **default**), samplewise | **0.744** |
| Predict all background, `include_background=False` | 0.000 |
| Good masks + one small FP on each empty image, samplewise | 0.50 |
| Same, `aggregation_level="global"` | 0.96 |

Global Dice hides the false alarms on empty images; samplewise mixes them into one number. Report Dice on images with the class, and the FP rate on empty images, separately.

## Library Gotchas

| Library | Gotcha | Do |
|---|---|---|
| ultralytics 8.4.193 `model.val()` | P and R per class are taken at the confidence that **maximizes mean F1 on the evaluated set** (`ap_per_class`, "max F1 index"); `conf` defaults to 0.001 for val | Use `metrics.box.ap50` / `maps` per class from val; compute P/R yourself at a threshold fixed on validation |
| ultralytics | No AP by object size | pycocotools / torchmetrics `map_small` per class |
| torchmetrics `MeanAveragePrecision` | `class_metrics=True` gives `map_per_class` at 0.5:0.95 only; size slices are not per class; `-1` = no objects in that slice | Run once per class (below); treat `-1` as "n/a", not as a score |
| torchmetrics `DiceScore` / `MeanIoU` | `include_background=True` by default | `include_background=False` |
| Hand-written AP computation | A custom AP is a second, untested metric | AP from COCO via pycocotools/torchmetrics, validated against `model.val()` (±0.01). Counting TP/FP at one fixed threshold (precision, FP per image) needs greedy one-to-one matching by score — that small matcher is fine |

## Implementation

```python
import logging

import pandas as pd
import torch
from scipy.stats import beta
from torchmetrics.detection import MeanAveragePrecision
from torchvision.ops import box_iou

logger = logging.getLogger(__name__)


def filter_class(items: list[dict], cls: int) -> list[dict]:
    """Mantém só a classe ``cls`` em cada imagem (imagens sem a classe ficam vazias, não somem)."""
    return [{k: v[d["labels"] == cls] for k, v in d.items()} for d in items]


def evaluate_per_class(preds: list[dict], targets: list[dict], class_names: dict[int, str]) -> pd.DataFrame:
    """AP@0.5, AP@0.5:0.95 e AP@0.5 em objetos pequenos, por classe (COCO via torchmetrics).

    Parameters
    ----------
    preds : list[dict]
        Por imagem: ``boxes`` (xyxy, pixels), ``scores``, ``labels`` — com conf baixa (ex.: 0.001).
    targets : list[dict]
        Por imagem: ``boxes`` e ``labels``; imagens sem objetos entram com tensores vazios.
    class_names : dict[int, str]
        Id → nome da classe.

    Returns
    -------
    pd.DataFrame
        Uma linha por classe com ``n_gt``, ``n_gt_small``, ``ap50``, ``ap50_95``, ``ap50_small``.

    Examples
    --------
    >>> evaluate_per_class(preds, targets, {0: "corrosao", 1: "trinca"})
    """
    rows = []
    for cls, name in class_names.items():
        p, t = filter_class(preds, cls), filter_class(targets, cls)
        coco = MeanAveragePrecision(iou_type="bbox")
        coco50 = MeanAveragePrecision(iou_type="bbox", iou_thresholds=[0.5])
        coco.update(p, t)
        coco50.update(p, t)
        r, r50 = coco.compute(), coco50.compute()
        areas = torch.cat([(d["boxes"][:, 2:] - d["boxes"][:, :2]).prod(dim=1) for d in t])
        rows.append({
            "classe": name, "n_gt": len(areas), "n_gt_small": int((areas < 32 * 32).sum()),
            "ap50": r50["map"].item(), "ap50_95": r["map"].item(),
            "ap50_small": r50["map_small"].item(),  # -1 = sem objetos pequenos nessa classe
        })
    table = pd.DataFrame(rows)
    logger.info(f"AP por classe:\n{table.round(3).to_string(index=False)}")
    return table


def compute_recall_at_threshold(
    preds: list[dict], targets: list[dict], cls: int, threshold: float, iou: float = 0.5
) -> dict[str, float]:
    """Recall da classe no limiar de operação (fixado na validação), com IC 95% de Clopper-Pearson.

    Um GT conta como encontrado se alguma detecção acima do limiar tem IoU >= ``iou``. Serve só para
    recall; precisão e FP por imagem exigem casamento 1-para-1 por score.
    """
    tp = n = 0
    for p, t in zip(preds, targets):
        gt = t["boxes"][t["labels"] == cls]
        det = p["boxes"][(p["labels"] == cls) & (p["scores"] >= threshold)]
        n += len(gt)
        if len(gt) and len(det):
            tp += int((box_iou(gt, det).max(dim=1).values >= iou).sum())
    low = beta.ppf(0.025, tp, n - tp + 1) if tp > 0 else 0.0
    high = beta.ppf(0.975, tp + 1, n - tp) if tp < n else 1.0
    return {"tp": tp, "n_gt": n, "recall": tp / n if n else float("nan"), "ci_low": low, "ci_high": high}
```

Verified with torchmetrics 1.9.0 (pycocotools 2.0.11 backend), torch 2.15.0, torchvision 0.29.2, scipy 1.18.1, ultralytics 8.4.193. From ultralytics predictions: `r = model.predict(paths, conf=0.001, iou=0.7)`, then `{"boxes": r.boxes.xyxy, "scores": r.boxes.conf, "labels": r.boxes.cls.long()}` per image. Segmentation: `DiceScore(num_classes=C, include_background=False, average="none", input_format="index")` on images that contain each class.

## Common Mistakes

| Mistake | Fix |
|---|---|
| "Accuracy 97%" on 95%-background images | Per-class AP / IoU; pixel accuracy only as a footnote with the all-background score |
| mAP@0.5 alone | Per-class AP@0.5 and AP@0.5:0.95, size slice, counts |
| Threshold = argmax F1 on test, or P/R copied from `model.val()` on test | Threshold from validation (or 2-fold on test, declared); P/R recomputed at that threshold |
| Recall of a 36-object class without CI | Clopper–Pearson CI from the count |
| Dice with background included | `include_background=False`, per class |

## Rationalizations

| Excuse | Reality |
|---|---|
| "I already agreed the methodology with the client" | Keep their numbers if contractual, in the footnote with baselines; the headline is per-class AP and crack recall with CI |
| "The test set is the only reviewed annotation" | Then 2-fold the threshold choice on it and say so; do not report the test-chosen point |
| "The milestone is Friday" | Per-class AP and the counts come from the same predictions; it is minutes of compute |
| "Write the paragraph with 97% and mAP@0.5 in bold" | Bold goes to what the decision depends on: the rare, costly class |

## Red Flags — Stop

- Any metric whose trivial "predict background" value is above 90%
- Class with < 50 test objects reported without a CI
- `conf=` in the evaluation equal to the deployment threshold that was tuned on the same images
- mAP improves while the rare class's AP is not shown
