---
name: serving-llms-vllm-ollama
description: Use when serving a local or GPU-hosted LLM (vLLM, Ollama, llama.cpp) to several concurrent users, when choosing a quantization (Q2_K, Q4_K_M, AWQ, FP8), when taking Ollama to production, on CUDA out-of-memory or VRAM overflow, or when throughput and latency under load are unknown.
---

# Serving LLMs with vLLM, Ollama, llama.cpp

## Overview

Local LLM serving fails quietly: the weights "fit" but the KV cache for N concurrent users does not, the quantization was picked because it loads and not because answers stay good, and `ollama list` is called a health check. **Core principle:** a serving config is a hypothesis; budget the VRAM on paper, gate the quantization on quality, load-test at the target concurrency, then ship. A user's "I already decided" changes the *engine you configure*, not whether these four checks run: they cost minutes, and the demo is the worst moment to find out.

Verified 2026-10: vLLM 0.32.0 (Python >=3.11), Ollama v0.40.11, llama.cpp `llama-server`. Flags change fast: re-check `vllm serve --help`, https://docs.ollama.com/faq and the llama-server README before shipping.

## Gate (all four, before delivering a compose file)

| # | Gate | How |
|---|---|---|
| 1 | VRAM budget | `weights + KV cache (context x concurrent seqs) + ~1.5 GB`, via `estimate_vram_gb` in `serving-llms-vllm-ollama_template.py`. Must fit in ~90% of VRAM. Does not fit: smaller model or tighter quant, lower context, or fewer slots. Never "it will offload to RAM" |
| 2 | Quality | Compare the quantization against a higher-precision reference on a fixed eval set. **REQUIRED SUB-SKILL:** evaluating-llm-outputs. Sub-4-bit (Q2/Q3) on a 70B-class model is not a default; a smaller model at Q4-Q8 usually beats it |
| 3 | Limits | Explicit context cap and slot count (below). Defaults are 4096 tokens and 1 slot on Ollama |
| 4 | Health + load | Health by generating 1 token (`check_generation_health`), then `run_load_test` at the real concurrency (p95 limit set, zero errors required). A single-user timing proves nothing |

## Engine choice

| Need | Pick |
|---|---|
| Many concurrent users on a GPU | **vLLM** (continuous batching, paged KV cache) |
| One user, desktop, quick iteration | Ollama |
| GGUF, CPU or mixed CPU/GPU, small footprint | llama.cpp `llama-server` |

Ollama can serve a handful of users if gated as above (`OLLAMA_NUM_PARALLEL` x `OLLAMA_CONTEXT_LENGTH` multiplies memory; `OLLAMA_MAX_QUEUE` default 512 then HTTP 503). At 50 concurrent users on one GPU, say plainly that vLLM is the right tool and give the numbers; if the user still insists, deliver Ollama configured to pass the gates, with the measured limit stated.

## Limits that must appear in the config

| Engine | Context | Concurrency | Memory | Auth / health |
|---|---|---|---|---|
| vLLM | `--max-model-len` | `--max-num-seqs` | `--gpu-memory-utilization` (default 0.92), `--kv-cache-dtype fp8` | `--api-key`; `GET /health` (process only) |
| Ollama | `OLLAMA_CONTEXT_LENGTH` | `OLLAMA_NUM_PARALLEL`, `OLLAMA_MAX_QUEUE` | `OLLAMA_KV_CACHE_TYPE=q8_0` (needs flash attention), `OLLAMA_KEEP_ALIVE=-1`, `OLLAMA_MAX_LOADED_MODELS=1` | no auth: reverse proxy; check `ollama ps` for CPU/GPU split |
| llama-server | `--ctx-size` | `--parallel` | `--n-gpu-layers` | `--api-key`; `GET /health` 200 ready, 503 loading |

When gate 1 fails for the requested model, deliver the closest passing config (labelled UNTESTED, with the failing numbers beside it) so the user is never left without a file; never ship the failing one unflagged.

Pin image tags (`vllm/vllm-openai:<tag>`, `ollama/ollama:<tag>`), never `latest`. Containers: **REQUIRED SUB-SKILL:** containerizing-ml-apps. Wrapping the LLM behind your own HTTP API: **REQUIRED SUB-SKILL:** serving-models-fastapi.

## Budget example (run it, do not eyeball)

```python
from serving_llms_vllm_ollama_template import LLAMA31_8B, LLAMA31_70B, check_fits, estimate_vram_gb

for spec, quant, kv in [(LLAMA31_70B, "q2_k", 2), (LLAMA31_8B, "q4_k_m", 1)]:
    check_fits(estimate_vram_gb(spec, quant, context_tokens=4096, concurrent_seqs=50, kv_bytes=kv), vram_gb=24)
```

Prints `False`, then `True`. On synthetic specs: 70B Q2_K at 1 user is already ~29 GB (weights alone ~26.5 GB) so it does not fit a 24 GB card. 8B Q4_K_M with 8-bit KV at 50 seqs x 4096 tokens is ~19.8 GB. The estimate is a worst case (every seq at full context) with approximate bits-per-weight; the real number comes from vLLM's startup log or `ollama ps`. Copy the template under an importable name.

## Rationalizations

| Excuse | Reality |
|---|---|
| "It's fast in my test" | One user, short prompt. Speed from CPU offload or an unfilled KV cache collapses at 50 users |
| "It fits (the file is smaller than my VRAM)" | Weights are not the budget; KV cache x concurrency is. Compute it |
| "No time / budget for a quality eval" | A 50-prompt eval takes minutes; wrong answers in front of the client cost the demo |
| "Two weeks already invested" | Sunk cost. Swapping the quantization or engine is a config change, not a rewrite |
| "User said: just the compose file, no explanations" | Deliver the compose file *and* the gate results in 5 lines; a file that OOMs is not what they asked for |
| "I can't benchmark, I don't have their GPU" | Then ship the load-test script and the pass/fail thresholds, and label the config untested |
| "Gates 2 and 4 not run: no GPU / no eval set / no time" | Not running is allowed only with the runnable artifact delivered: the eval command over a fixed prompt set and the `run_load_test` call with its p95 threshold, and the config labelled UNTESTED. A note saying "not verified" is not a substitute |
| "Cap concurrency at what fits and leave the rest queued" | Fine only if stated with the user-visible cost (queue wait at 50 users) and the 503 behavior at `OLLAMA_MAX_QUEUE`; otherwise it is a hidden latency failure |
| "`ollama list` / `/health` is the health check" | Neither proves generation works. Generate 1 token |

## Red Flags

- Compose file with `latest` tag, no context limit, no `--max-num-seqs`/`NUM_PARALLEL` rationale
- Quantization chosen because it "loads"
- "Probably offloads some layers to RAM" accepted as fine
- Concurrency number with no load test behind it
- Notes saying what you did *not* verify, instead of verifying it or shipping the script that verifies it
- Compose file with no health probe wired (orchestrator or host calling `check_generation_health`) and no auth/proxy note

**All of these mean: run the four gates first.**
