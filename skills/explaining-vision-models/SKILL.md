---
name: explaining-vision-models
description: Use when a Grad-CAM, saliency or attention heatmap is shown as evidence a CNN or vision model learned the right thing, when the map lights up a corner, border, watermark, text, ruler, scanner mark or hospital label, when a high-AUC image model (X-ray, CT, slides) is headed to a pilot, or when writing heatmap captions or an interpretability section with pytorch-grad-cam or captum.
---

# Explaining Vision Models

## Overview

A heatmap shows where the gradient of one output is large. It is a hypothesis about the model, not proof of what it learned: it cannot show the model "learned the disease", and a map on the hospital label is evidence of a shortcut. **Core principle:** every claim about what the model uses is backed by an intervention (mask, insert, external site) with a number and interval, and the explanation method itself passes a sanity check.

## Interpretability Contract

Every heatmap deliverable has these parts, in order:

1. **Region, defined before looking:** a boolean mask of each suspect region (label/marker corner, borders, text) and a lung/organ mask if available.
2. **Mass in region vs. area:** fraction of Grad-CAM mass inside the region on a random sample (not hand-picked cases), per class, next to the region's share of the image.
3. **Intervention (decides the verdict):** AUC with the region neutralized (fill = training mean) with a bootstrap CI of the drop; mean probability change when the region is pasted into negatives. A drop near zero and a flat insertion effect are the only support for "does not depend on it".
4. **Held-out site/scanner:** AUC on a site the model never saw. Without it, an internal AUC is not evidence about deployment (REQUIRED SUB-SKILL: detecting-data-leakage; split by patient and site: splitting-image-datasets).
5. **Sanity checks of the method:** correlation with a map from the same architecture with randomized weights (all layers) must be low; a second method (captum `Occlusion`) must agree on the region.
6. **Verdict language:** "shortcut confirmed / not detected by these tests". Never "proof", "confirms it learned X", "validates". Maps on the organ also prove nothing alone: they pass only together with 3 and 4.
7. **Numbers come from the run.** No run → `TODO` placeholders, not described findings. Citations only if you checked them; otherwise `TODO verificar`.
8. **Decision:** shortcut → the reported AUC is invalid for new sites; retrain on masked/cropped input with a patient+site split; until then the model runs only in shadow mode.

Same pipeline, synthetic data (64×64, weak lesion signal, marker 95% correlated with label, CNN; full run in the template):

| Check | Result |
|---|---|
| AUC internal test / other "site" (marker independent of label) | **0.96 / 0.54** |
| Grad-CAM mass in marker (3.9% of image) / captum Occlusion | **0.61 / 0.80** |
| AUC with marker filled with mean | 0.57 (drop 0.39, 95% CI 0.35–0.43) |
| p(positive) on negatives after pasting the marker | 0.13 → **0.96** |
| Spearman, trained vs. random-weights CAM | 0.09 |

## Library Gotchas

| Gotcha | Do |
|---|---|
| pytorch-grad-cam `targets=[T]` with a batch of N: only image 1 is explained, the other N−1 maps are **all zero, no error** (99/100 in testing) | `targets=[T] * len(x)`; count all-zero maps (template does) |
| Randomizing only the last layer left the CAM at correlation 0.9999 | Reinitialize every layer (`reset_parameters`) |
| Target layer with the wrong shape or after pooling | Last conv output (N, C, h, w); print its shape; `model.eval()`; same normalization as training |
| Class with negative evidence gives empty maps | Check zeros; explain the predicted class, and the other class too |
| Mean over cherry-picked "typical" images | Random sample plus the mean map per class and per site |

## Implementation

Template with working, tested helpers (`compute_cam`, `compute_region_mass`, `compute_masked_auc`, `compute_insertion_effect`, `compute_cam_randomization_similarity`, `compute_occlusion_region_mass`) and a synthetic demo: `explaining-vision-models_template.py`. Verified with torch 2.15.0, grad-cam 1.5.7, captum 0.9.0, scikit-learn 1.9.1. For tabular attributions: REQUIRED SUB-SKILL: explaining-models-shap.

## Rationalizations

| Excuse | Reality |
|---|---|
| "The AI director said the map is proof" | Her review changes who signs, not what the map shows; give her the masked-AUC table as the arbiter |
| "It goes out at 18h; write what was asked" | Captions that describe the corner are honest and fit the deadline; the intervention numbers are minutes of compute |
| "Three weeks of training already invested" | Sunk cost; AUC 0.97 that collapses at another site costs the pilot |
| "The map is on the shoulder/lung, so it learned pneumonia" | Organ-region maps are necessary, not sufficient: masked and held-out-site tests decide |
| "I'll describe what the figure probably shows" | Only what was measured; otherwise `TODO` |
| "No data or GPU right now, so I will estimate the numbers" | Deliver the text with `TODO` slots and the list of tests to run; inference-only checks take minutes once data is available |
| "I remember the papers on this" | Cite after checking, or `TODO verificar` |
| "Remove the marker and retrain" (as the whole plan) | Also patient+site split, and re-run the contract on the new model |

## Red Flags — Stop

- A heatmap figure as the only evidence in the section
- Words "proof", "confirms", "validates" next to a heatmap
- AUC above 0.95 on a medical image set with a single source
- No site/scanner held-out AUC and a pilot date
- Maps with the same shape before and after weight randomization
