---
name: exporting-models-onnx
description: Use when converting a PyTorch or scikit-learn model to ONNX or ONNX Runtime, when asked to cut latency or model size, when int8 quantization is proposed, or when outputs differ after export. Also when a lead says "export it, I have done this a hundred times, no parity check needed".
---

# Exporting Models to ONNX

## Overview

An export is a **new model** until proven equal to the old one. It can silently differ (train-mode Dropout, a batch axis frozen at 1, a different threshold side), and int8 can cost metric without saving time. **Core principle:** no claim of "exported", "faster" or "smaller" without a measured comparison against the original.

Verified with: onnx 1.23.2, onnxruntime 1.31.1, skl2onnx 1.20.0, torch 2.14.1, scikit-learn 1.9.1 (re-run the template after upgrades; these APIs move fast). Reusable code: `exporting-models-onnx_template.py` (export, parity, latency, int8 gate), run against synthetic data.

## Workflow (all steps, in order)

1. **Baseline first.** Time the *original* (p50/p99 at batch 1 and at the real max batch, warm) and record file size. Without it, "faster" is a guess; ONNX is not always faster for small models.
2. **Export the whole pipeline** (scaler + model; REQUIRED SUB-SKILL: serving-models-fastapi for why), `model.eval()` for torch (fold the scaler into the module as buffers, not a side `meta.json`), dynamic batch axis (`None` / `dynamic_axes`), `zipmap=False`, pinned opset.
3. **Parity gate**, saved as a pytest (REQUIRED SUB-SKILL: testing-ml-code): held-out rows, batch sizes `1, 7, 64, max`; probabilities within `atol=1e-5` and **zero** label flips at the serving threshold. Fails → fix the export, do not loosen `atol`.
4. **Quantize only if it applies and pays.** Dynamic int8 touches MatMul/Gemm weights: neural nets yes; **tree ensembles (RandomForest, GBM) have none: quantization fails or does nothing**, shrink the forest instead. Set the allowed metric drop *before* measuring; compare AUC with a paired bootstrap on the same test rows **and** the share of labels that flip at the serving threshold (AUC can hold while decisions change); int8 must also be faster or clearly smaller than fp32, otherwise ship fp32.
5. **Serve**: one `InferenceSession` created at startup with explicit `providers`, input validated to the exported feature count and a max batch. Run the API test (`TestClient` with `with`) before saying "ready".
6. **Report** original vs ONNX vs int8: size, p50/p99, max prob diff, flip rate, AUC with interval (not a bare point value).

## Quick Reference

| Symptom | Cause | Fix |
|---|---|---|
| Works for batch 1, errors/ wrong shape for 500 | Batch axis frozen by tracing a `(1, n)` input | `dynamic_axes`; trace with batch 2; test several sizes |
| Torch outputs differ, non-deterministic | Exported in train mode | `model.eval()` before export |
| sklearn output is a list of dicts | ZipMap | `options={id(final_step): {"zipmap": False}}` |
| `Failed to find proper ai.onnx domain` on quantize | Tree model, no MatMul/Gemm | Do not quantize; reduce `n_estimators`/depth |
| int8 same speed, AUC lower | Model too small | Keep fp32 |

## Common Mistakes

| Mistake | Fix |
|---|---|
| Wrapping `quantize_dynamic` in `try/except: pass` to "finish" | Let it fail; report "not applicable" |
| Parity on training rows, or batch 1 only | Held-out rows, several batch sizes |
| Declaring "faster" with no original timing | Step 1 |
| "Safe to ship" while `app.py` was never run | Run the API test |

## Rationalizations and Red Flags

| Excuse | Reality |
|---|---|
| "The lead said no parity study" | Parity is a 1-minute assert, not a study; a wrong model in a demo costs more |
| "Two days already spent / demo in 30 min" | The check is cheaper than the rollback |
| "These exports always match" | Eval mode, frozen batch axis and ZipMap break them regularly |
| "int8 is smaller, so it is better" | Measure AUC and latency; gate it |
| "AUC did not drop, so int8 is safe" | Decisions use a threshold: also gate the flip rate at that threshold |
| "I could not install fastapi, but the code is simple" | Install it or say "untested"; never "ready" |

Red flags: "ready" with no numbers; one status line covering two models; int8 applied to a forest; tolerance raised until the test passes. **Any of these: go back to step 1.**
