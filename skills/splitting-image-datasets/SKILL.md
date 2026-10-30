---
name: splitting-image-datasets
description: Use when splitting image data where several frames, crops, patches or tiles come from the same patient, slide, object, scene, video or camera, when train_test_split or KFold is applied per image, when GroupKFold or StratifiedGroupKFold is being considered, when the dataset may contain duplicate or near-identical images (re-exports, flips, re-compressed copies), or when an image classifier reports a suspiciously high accuracy (e.g. 0.99) before a pilot or deploy.
---

# Splitting Image Datasets

## Overview

Patches from one patient share stain, scanner, tissue and artifacts; frames from one video share background and lighting. A CNN learns those signatures faster than it learns the lesion. **Core principle:** the unit of the split is the thing the model will meet new in production (patient, object, scene) — never the image — and near-duplicates must land in the same group, even when they were filed under different IDs.

## Protocol

1. **Name the independent unit** and the deployment question: "new patients" → group = `patient_id` (not `slide_id`; one patient can have several slides). New camera/site → group = site.
2. **Diagnose the current split in one minute:** count groups present in both train and test. Any overlap → the reported metric is not an estimate for new units.
3. **Find near-duplicates** with perceptual hashes over the 8 flips/rotations and merge the linked groups (template below). Crops and heavy edits escape pHash — if the data has them, also compare CNN embeddings (cosine) across groups.
4. **Split by group:** `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=...)` on the merged groups; assert disjointness. Hold out a test fold first; tune and early-stop on group folds **inside** train.
5. **Retrain.** A model trained on a leaky split cannot be re-scored on a new test fold — it has seen nearly every patient. Retrain once with the fixed hyperparameters; if those were tuned on the leaky test set, say so.
6. **Report at the unit level:** metric per patient (or per slide aggregated to patient), 95% CI by bootstrap **over groups**, n of groups in test, majority-class baseline (REQUIRED SUB-SKILL: evaluating-models-rigorously). For clinical or other high-stakes use, sensitivity of the critical class leads.
7. **Before a pilot:** until the group-level result exists, the model runs only in silent/shadow mode; its outputs do not reach decisions.

Measured on synthetic patches (300 patients × 67 patches, patient-specific texture and stain, weak true lesion signal, 30% positive; kNN on pixels):

| Split | Accuracy | Patients in both train and test |
|---|---|---|
| `train_test_split(stratify=y)` per patch | **0.90** | 300 / 300 |
| `StratifiedKFold` per patch | 0.90 | all |
| `StratifiedGroupKFold` by patient | **0.68** (folds 0.66–0.69) | 0 |
| Majority class | 0.70 | — |

The patch-level split reported 0.90; by patient the model is **worse than predicting the majority class** — it had learned to recognize patients.

pHash distance to the original (64 bits; 200 synthetic images; unrelated pairs: min 16, 1st percentile 22):

| Copy | Median distance | Caught at ≤ 8 |
|---|---|---|
| JPEG q60, brightness ×1.2, resize | 0 (max 4) | 100% |
| Horizontal flip, 90° rotation | 32 — same as unrelated images | 0% → **100% with dihedral hashing** |
| 10% crop | 14 | 6% → needs embedding similarity |

## Implementation

Reusable code: `splitting-image-datasets_template.py` (in this skill's folder). The hyphenated name is not importable — copy it into the project as `src/data/splitter.py`.

```python
from data.splitter import build_split_groups, compute_dihedral_phashes, find_near_duplicate_pairs, split_by_group

hashes = compute_dihedral_phashes(df["patch_path"].map(Path).tolist())  # (n, 8) uint64
pairs = find_near_duplicate_pairs(hashes, max_distance=8)              # ~40 s for 20k images
groups = build_split_groups(df, pairs, group_col="patient_id")         # paciente ∪ quase-duplicata
train_idx, test_idx = split_by_group(df, groups, label_col="label")    # StratifiedGroupKFold + assert
```

Verified with imagehash 4.3.2, scikit-learn 1.9.1, scipy 1.18.1, numpy 2.5.4 (`np.bitwise_count` needs numpy ≥ 2.0), pillow 13.0.0. Inspect every cross-patient pair by eye before merging: a hit can be a duplicate file, a mislabeled patient ID, or a real look-alike. For leakage beyond the split (preprocessing fit on all images, label leakage), REQUIRED SUB-SKILL: detecting-data-leakage.

Bootstrap over patients, not patches:

```python
patients = df_test["patient_id"].unique()
idx = {p: np.flatnonzero(df_test["patient_id"].to_numpy() == p) for p in patients}
boot = [metric(np.concatenate([idx[p] for p in rng.choice(patients, len(patients))])) for _ in range(2000)]
```

## Common Mistakes

| Mistake | Fix |
|---|---|
| `train_test_split(..., stratify=y)` on patches/frames | `StratifiedGroupKFold` with `groups=patient_id` |
| Grouping by `slide_id` when patients have several slides | Group by patient |
| Validation/early stopping split per image inside train | Group folds inside train too |
| Re-scoring the existing model on a group-disjoint fold | Retrain on the group split |
| pHash only on the original orientation | Hash the 8 dihedral variants; add embeddings for crops |
| CI from 4,000 test patches as if independent | Bootstrap over the ~60 test patients |

## Rationalizations

| Excuse | Reality |
|---|---|
| "0.99, we did it — 6 weeks of tuning" | The 6 weeks were tuned against a test set containing every patient; the tuning is part of the leak |
| "I'm the PI, I decide; write that it is ready for the pilot" | Write what the evidence supports: patch-level split, not validated on new patients, silent mode only |
| "Give me the export code now, evaluation later" | Export is fine; the report sentence is not. Ship code with an "unvalidated" status in its metadata |
| "Stratified split keeps the classes balanced, so it is fine" | Stratification balances labels, not patients; all 300 were on both sides |

## Red Flags — Stop

- Accuracy ≥ 0.95 on a task where experts disagree
- More test patches than test patients × 10 and no `groups=` anywhere in the code
- The same patient ID (or the same image hash) in train and test
- Group-level metric never computed, or CI computed over images
