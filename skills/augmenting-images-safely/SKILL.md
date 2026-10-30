---
name: augmenting-images-safely
description: Use when adding image augmentation (Albumentations, torchvision v2) to a classifier, detector or segmenter, when flips or rotations are applied to digits, text, or left/right medical labels, when one shared Dataset/Compose feeds both train and validation, when MixUp or CutMix is added, or when RandomResizedCrop with a small scale may cut out the object of interest.
---

# Augmenting Images Safely

## Overview

An augmentation is a claim: "the label does not change under this transform." A flip turns a normal heart into dextrocardia, `6` into a mirrored `6`, text into noise; a 5% crop can remove the lesion and keep the "lesion" label. **Core principle:** augment only the training stream, only with transforms the task's label survives, and prove it with a check, not with a belief. Validation, test and serving use one deterministic transform.

Verified with albumentations 2.0.8, torchvision 0.29.2 (`transforms.v2`), numpy 2.5.4, synthetic data. Albumentations 2.x renamed arguments (`RandomResizedCrop(size=(h, w), ...)`, `Affine(rotate=..., translate_percent=...)`); re-check names when upgrading.

## Protocol

1. **Write the invariance list per task** before any `Compose`: which of flip-H, flip-V, rot90, rotation angle, crop scale, color changes keep the label. Laterality (heart, lung side, left/right marker): no flip-H, no rot90. Digits and text: no flips, no rot90, small rotation only (±10°). Detection/segmentation: spatial ops must transform boxes/masks with the image (`bbox_params`, `mask=`) and bboxes need `min_visibility`.
2. **Two builders, never one shared Compose.** `build_train_transform(task)` and `build_eval_transform` (Resize + Normalize + tensor, nothing random). A shared Dataset takes `transform=` as an argument. Normalization uses the **backbone's** mean/std (REQUIRED SUB-SKILL: fine-tuning-vision-models), the same in both builders; `[0.5]` or dataset stats silently shift activations.
3. **Gate the pipeline in tests/CI:** `validate_forbidden_ops(compose, task)` (recurses into `OneOf`/`Sequential`), `validate_eval_is_deterministic(eval_tf, img)`, and `compute_object_survival(train_tf, img, mask)` on a few real annotated samples with the object off-centre.
4. **Crop budget:** survival rate (object keeps ≥80% of its area) must be ≥ 0.95 on the real object-size distribution. Measured, 8×8 object in a 64×64 image: `RandomResizedCrop` scale (0.05, 1) → survival **0.57**; (0.5, 1) → 0.83; (0.8, 1) → **0.99**. Small lesions: scale ≥ 0.8 or a bbox-safe crop.
5. **MixUp/CutMix only for single-label, whole-image classification where the class is not a local or positional property.** Not for laterality, small-lesion presence, detection, segmentation, OCR. Apply in `collate_fn`/after the loader via `build_batch_mixer`; labels become soft `(B, K)`, so use a soft-target loss and do not read training accuracy as comparable to clean accuracy. Never mix validation batches.
6. **Evaluate on clean data and say so.** Val metric = deterministic transform, CI, split by group (REQUIRED SUB-SKILL: splitting-image-datasets, evaluating-models-rigorously). A val number computed under random flips/crops is not a held-out score; re-score the existing checkpoint with the eval transform (minutes, no retraining) before it goes on a slide. If a flipped/cropped pipeline was already used to train a laterality or lesion model, the model is suspect: report it as unvalidated until retrained without the transform.
7. **Seed and log:** `A.Compose([...], seed=SEED)` and a seeded `DataLoader` generator/`worker_init_fn` (REQUIRED SUB-SKILL: ensuring-reproducibility); log the transform spec with the run.

## Implementation

Reusable code: `augmenting-images-safely_template.py` (copy to `src/data/augmentation.py`): `build_eval_transform`, `validate_forbidden_ops`, `validate_eval_is_deterministic`, `compute_object_survival`, `build_batch_mixer`. Verified on synthetic images: a `OneOf([HorizontalFlip, Blur])` nested in a Compose was caught for `laterality`; a transform with `RandomBrightnessContrast` was flagged as non-deterministic; `build_batch_mixer(10)` returned `(8, 10)` soft labels summing to 1.

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2
from data.augmentation import (build_batch_mixer, build_eval_transform,
                               compute_object_survival, validate_forbidden_ops)

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)   # do backbone: weights.transforms()
train_tf = A.Compose([
    A.Resize(224, 224),
    A.Affine(rotate=(-5, 5), scale=(0.95, 1.05), p=0.5),    # sem flip: lateralidade
    A.RandomBrightnessContrast(0.1, 0.1, p=0.5),
    A.Normalize(mean=MEAN, std=STD), ToTensorV2(),
], seed=42)
eval_tf = build_eval_transform(224, MEAN, STD)
validate_forbidden_ops(train_tf, "laterality")
```

## Common Mistakes

| Mistake | Fix |
|---|---|
| `HorizontalFlip` on digits, text, heart position, L/R markers | Invariance list per task; `validate_forbidden_ops` in CI |
| `VerticalFlip`/`RandomRotate90` "for variety" | Same: 6/9, text, anatomy have an up and a down |
| Same Compose for train and val Dataset | Two builders; eval has no random op |
| Reporting val accuracy measured under augmentation | Re-score with `eval_tf`; CI over groups |
| `RandomResizedCrop(scale=(0.05, 1))` on small objects | Scale ≥ 0.8 or bbox-safe crop; measure survival |
| Image transformed but box/mask not | Pass `bboxes=`/`mask=` with `bbox_params(min_visibility=...)` |
| CutMix on lesion/laterality/detection data | Skip it; use photometric/mild affine only |
| `Normalize(0.5, 0.5)` with a pretrained backbone | Backbone's stats in both builders |
| Reading train accuracy under MixUp as a clean metric | Soft labels: log loss; judge on clean val |

## Rationalizations

| Excuse | Reality |
|---|---|
| "Demo in 3h, 0.97 is what we show, don't touch it" | Do not retrain; re-score the same checkpoint with `eval_tf` (minutes) and show that number with a CI |
| "Two weeks already invested in this pipeline" | The weights stay; the number is what gets re-measured. If the clean score drops, better now than in the room |
| "The PI said use the shared Compose, val is robust" | Random val is noisy and optimistic or pessimistic by luck; it is not a held-out score |
| "Flip is standard, the net will learn invariance" | For laterality it learns the wrong label; for digits it learns mirrored glyphs |
| "I will run the gates after the demo" | The gates take seconds; a pipeline that has not passed them is reported as unvalidated, not as done |
| "CutMix helps everywhere, I read a paper" | It helps whole-image class tasks; on local/positional labels the pasted patch removes or fakes the evidence |

## Red Flags: Stop

- `HorizontalFlip` or `RandomRotate90` in a config for digits, text or laterality
- The val `Dataset` built from the same object as train
- `scale=(0.05, ...)` crops with no survival measurement
- CutMix/MixUp on detection, segmentation, or small-lesion data
- A validation number that changes on re-run with the same checkpoint

**All of these mean: split the builders, run the three gates, re-score on clean data, and label the old number as measured under augmentation.**
