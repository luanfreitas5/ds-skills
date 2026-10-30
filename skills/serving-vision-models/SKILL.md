---
name: serving-vision-models
description: Use when exposing an image classifier or detector as an API (FastAPI, ONNX Runtime, TorchServe, Triton), when an endpoint resizes with cv2 but training used PIL (or vice versa), when uploads have no size limit, when the model is loaded per request, or when latency, batching or throughput must be measured. Also when told "internal demo, no parity or size-limit stuff".
---

# Serving Vision Models

## Overview

Vision endpoints fail quietly in four ways: preprocessing differs from training (BGR vs RGB, cv2 vs PIL interpolation, missing normalization), the upload is unbounded (huge file or decompression bomb), the model is loaded per request, and nobody measured latency. **Core principle:** the serving preprocessing is part of the model; prove it equal to the training transform with a number, then bound the input and measure.

Verified with: onnxruntime 1.31.1, torch/torchvision CPU, Pillow, opencv-python-headless, FastAPI + httpx (re-run the template after upgrades). Reusable code: `serving-vision-models_template.py` (preprocess parity, bounded decode, ONNX session at startup, parity/limit tests, latency), run against synthetic images.

REQUIRED SUB-SKILL: serving-models-fastapi (lifespan, schemas, `/health`, `/version`). REQUIRED SUB-SKILL: exporting-models-onnx (export, parity, int8 gate). REQUIRED SUB-SKILL: containerizing-ml-apps (image, non-root, model not baked into COPY).

## Workflow (all steps, in order)

1. **Pin the training transform.** Copy it verbatim (library, interpolation, size, color order, mean/std). The serving code uses the **same library** (PIL if trained with torchvision); cv2 "bilinear" differs from PIL by up to ~73/255 on a random image because PIL antialiases when downscaling. Different library only if you prove parity.
2. **Parity test, kept as pytest** (REQUIRED SUB-SKILL: testing-ml-code): preprocessed tensor `max|diff| < 1e-4` against the training transform, and API class == training-pipeline class on synthetic images of several sizes (portrait, landscape, tiny, grayscale, RGBA).
3. **Bound the input**: reject by `Content-Length` early, read at most `MAX+1` bytes, cap pixels (`Image.MAX_IMAGE_PIXELS`, explicit width*height check; `DecompressionBombError` -> 413, not 400), `convert("RGB")`, other decode errors -> 400. Test with a tiny PNG declaring 20000x20000. Do not trust `content_type`; the decode is the validation.
4. **Load once**: `lifespan` creates one `InferenceSession` (ONNX) or the `eval()` model; startup fails if the file is missing. `torch.load(..., weights_only=False)` executes pickle: even for "our own file" flag it in the report and name the follow-up (state_dict or ONNX); never on uploaded/untrusted files.
5. **Do not block the event loop**: inference is CPU-bound, so declare the endpoint `def` (threadpool), not `async def`, or offload explicitly.
6. **Measure**: warm p50/p99 at batch 1 and the real max batch, plus file size, before and after any change. Batching: client-side batch endpoint with `MAX_BATCH`, or server-side (Triton dynamic batching, TorchServe `batch_size`) only if measured throughput justifies it. Repeat the run: if two runs differ by more than ~30%, report the spread, not one number. No latency claims without numbers.
7. **Report** parity diff, limits tested (413/400), p50/p99, and what was *not* run.

## Choosing the stack

| Need | Use |
|---|---|
| One model, one service, CPU | FastAPI + ONNX Runtime |
| Many models, GPU, dynamic batching | Triton |
| PyTorch-native handlers, built-in batching | TorchServe |

Start with FastAPI + ONNX; move only when a measured need appears.

## Quick Reference

| Symptom | Cause | Fix |
|---|---|---|
| Correct in notebook, wrong in API | cv2 BGR / other interpolation / no normalize | Steps 1-2 |
| RAM spike, worker killed | Unbounded upload, decompression bomb | Step 3 |
| Every request slow | Model loaded per request | Step 4 |
| Health checks time out under load | `async def` running inference | Step 5 |
| Wrong shape for batch > 1 | Frozen batch axis in export | `dynamic_axes` (exporting-models-onnx) |

## Rationalizations and Red Flags

Observed in baseline (no skill): kept the lead's scope, but wrote the parity check as "not verified, I could not run anything", made the tests "optional smoke tests", skipped batching/latency numbers ("one-image requests do not need it"), kept `async def` with blocking inference and `weights_only=False`.

| Excuse | Reality |
|---|---|
| "Demo in 40 minutes, three days invested" | Parity is a 10-line assert; wrong classes on stage cost more |
| "Internal demo, no size limits" | One 200 MB file or bomb takes the demo down |
| "Lead has done this for years" | cv2 vs PIL differences are numeric, not opinion: show the diff |
| "Optional smoke test" | Required, and it must be run |
| "I could not run it, but the code is simple" | Install deps and run, or say "untested"; never "ready" |
| "No benchmark, so no claims" | Then measure; p50/p99 takes a minute |
| "It is our own pickle, keep weights_only=False" | Allowed only flagged, with the safer follow-up named |
| "Latency was noisy, so no number" | Repeat and report p50/p99 with the spread |
| "content_type says image/png" | Client-controlled; decode is the check |

Red flags: `cv2.imread/imdecode` with a PIL-trained model; `UploadFile.read()` without a cap; `torch.load` inside the handler; `async def` calling `model(...)`; "ready" with no p50/p99. **Any of these: return to step 1.**
