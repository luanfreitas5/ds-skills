---
name: tracing-llm-apps
description: Use when an LLM app in production has no visibility into cost per call or latency, when debugging a bad model response, when comparing prompt versions on live traffic, or when someone asks to "just print the prompt and the response" or to store raw prompts/responses that may contain CPF, names, or emails.
---

# Tracing LLM Apps

## Overview

A `print()` of prompt and response answers neither "what does a call cost?" nor "which prompt version made this?", and it copies personal data into stdout and infra logs. **Core principle:** one span per LLM call, carrying **metadata** (prompt version, model, tokens, cost, latency, status). **Content** is opt-in, masked, and never the only way to debug. Instrument with OpenTelemetry; Langfuse or LangSmith are just exporters you can add later.

**Violating the letter is violating the spirit:** "a quick log now, a proper one later" is the same as no tracing.

## Span Contract

| Attribute | Source | Rule |
|---|---|---|
| `app.prompt.id`, `app.prompt.sha256` | REQUIRED SUB-SKILL: versioning-prompts | The lock hash of the file in use, never a hash you compute from one string |
| `gen_ai.request.model` / `gen_ai.response.model` | request / API response | Store both; price by the **served** model |
| `gen_ai.usage.input_tokens` / `output_tokens` | API `usage` | Never estimate from characters |
| `app.cost.usd`, `app.price_table.version` | `compute_cost(table, model_served, ...)` | Versioned price table in `configs/` with source and verification date in `version`; keep a `verified` flag next to the table and label any displayed cost "estimated" while it is false (stakeholders asking for an "exact" number get the label, not a guess). Unknown model: span gets `app.cost.unpriced`, never cost 0 |
| `app.latency_ms`, span status | timer, `except` | Failed and timed-out calls are spans too (`ERROR`) |
| `app.input` / `app.output` | opt-in `capture_content` | Default off in production |

Tracing must never break serving: a pricing or exporter failure logs a warning and the user still gets the answer. Working code (OpenTelemetry SDK, JSONL exporter, optional OTLP, price table, per-prompt summary): `tracing-llm-apps_template.py`. Verified against `opentelemetry-sdk` 1.45.1 (`set_tracer_provider` is once per process: call `configure_tracing` in the entrypoint, use a module-scoped fixture in tests; success spans show status `UNSET`, which is normal); GenAI semantic-convention attribute names are still marked experimental, so check them when upgrading.

## Privacy Gate

Follows REQUIRED SUB-SKILL: protecting-pii-lgpd and REQUIRED SUB-SKILL: setting-up-project-logging (configure once in the entrypoint; no `basicConfig` in modules).

- Metadata-only is the default and answers cost, latency, and version questions.
- Content capture requires masking **plus** a test that sends CPF, email, and a truncated CPF through it and asserts none survives. Regex does not catch names; the template masks any 3+ digit run for this reason (in testing, a plain CPF regex let a truncated `123.456.7...` through).
- Content of real users: sample it, restrict access, set retention, and keep it out of third-party SaaS unless the legal basis is documented in the datasheet.
- Traces files (`logs/`, `*.jsonl`) go in `.gitignore`.

## Sending to Langfuse or LangSmith

Both accept OTLP/HTTP, so no code changes: set environment variables (keys only in `.env`, never in `configs/`). Verified 2026-10.

| Backend | `OTEL_EXPORTER_OTLP_ENDPOINT` | `OTEL_EXPORTER_OTLP_HEADERS` |
|---|---|---|
| Langfuse | `https://cloud.langfuse.com/api/public/otel` (HTTP only, no gRPC) | `Authorization=Basic <base64(pk:sk)>,x-langfuse-ingestion-version=4` |
| LangSmith | `https://api.smith.langchain.com/otel` | `x-api-key=<key>,Langsmith-Project=<project>` |

Needs `opentelemetry-exporter-otlp-proto-http`. Langfuse maps `gen_ai.*` attributes; custom `app.*` keys arrive as metadata only if you use its `langfuse.observation.metadata.<key>` form, so decide which backend before relying on filters.

## Comparing Prompt Versions

Group spans by `app.prompt.sha256` (`summarize_by_prompt`). Latency and cost differences between versions are descriptive. Quality differences need the paired eval of REQUIRED SUB-SKILL: evaluating-llm-outputs. Never declare a version "better" from trace averages.

## Rationalizations

| Excuse | Reality |
|---|---|
| "The director said just print, no tools" | The need is visibility for the demo. A local JSONL exporter is as fast as a print and answers cost and version. The user's constraint governs the platform, not the privacy gate |
| "I'll do tests and config after the demo" | Pricing hardcoded in code and untested masking are what break first. The template is the shortcut |
| "Hash of the system prompt is enough versioning" | Model and params change outputs too. Use the lock hash from versioning-prompts |
| "I'll assume the price per 1M tokens" | An assumed price becomes a fact in a dashboard. Versioned table with source and date; show costs as "estimated" until someone confirms it |
| "OpenTelemetry is banned, so I'll use my own logger" | OTel here is a local library, not a platform or a meeting. Say so in one line and ship it; a home-made format cannot be exported or correlated later |
| "Pricing/exporter error is a bug, let it raise" | Observability that fails a customer request is worse than none. Warn, flag the span, return the answer |
| "Regex masks CPF and email, so logging text is safe" | Names and fragments pass. Metadata by default; content opt-in with a leak test |
| "We already built our own logging for 3 days" | Sunk cost. If it cannot export traces or carry prompt hash, it is replaced by the template in minutes |
| "Failures are logged by the app anyway" | A bad answer after a timeout retry needs the failed span in the same trace |

## Red Flags: Stop

- `print(prompt)` or `logger.info(response)` on user text
- Cost computed from a constant inside the handler, or not stored per call
- Prompt version derived from a string, not from the lock
- Only the requested model recorded
- Content tracing enabled with no leak test
- Trace files not git-ignored
- Cost shown to stakeholders from an unverified price, unlabelled

**All of these mean: use the span contract and the privacy gate before shipping.**
