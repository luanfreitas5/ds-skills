---
name: building-n8n-workflows
description: Use when building or fixing an N8N workflow that calls an LLM, runs on a schedule or webhook, and sends email, messages, or HTTP requests, when API keys or passwords are pasted into nodes, when re-running after a failure resends or duplicates effects, or when told error branches are unnecessary "for a demo".
---

# Building N8N Workflows

## Overview

An N8N workflow is a program that **re-runs**: schedules tick again, webhooks are retried, and "Execute workflow" restarts at item 1. **Core principle:** a deadline, a sunk cost, or a CTO's order changes the *scope* of the work, never the three invariants: **secrets live in Credentials, every external effect is idempotent, every failure path is visible.** Each costs minutes and a half-built workflow without them sends 211 duplicate emails.

Verified 2026-10-28: docs.n8n.io (error handling, Remove Duplicates node); GitHub latest release `n8n@2.45.5`. The error-handling page documents Error Trigger and Stop And Error; node-level settings (`onError`, `retryOnFail`) were not on the fetched pages. Export a workflow from your instance to confirm field names and `typeVersion` before hand-writing node JSON.

**Violating the letter is violating the spirit:** "it's only a demo" still emails real people.

## Rules

| Rule | How |
|---|---|
| Secrets in Credentials | OpenAI/SMTP/Gmail/HTTP auth go in **Credentials** (or env variables). Exported JSON then holds a credential id, not the key. Keys already pasted in a node: treat as leaked once exported, screenshotted, or shared. Move them (about 5 minutes per key) and rotate. "Working already" is not a reason to keep them |
| Idempotency before effects | Stable key per effect (`build_idempotency_key(email, campaign)`), never `$execution.id` or a timestamp. Choose the order deliberately, see table below |
| Never blind-retry a send | `Retry On Fail` on Gmail/SMTP/HTTP POST can double-send. Retry only idempotent reads or calls that accept an idempotency key |
| Failures are visible | Set an **Error workflow** (Error Trigger, then alert) in Workflow Settings; on effect nodes use "Continue (using error output)" and route failures to a log/alert branch. Use Stop And Error for violated invariants. REQUIRED SUB-SKILL: `setting-up-project-logging` (log counts and ids, never bodies or addresses) |
| LLM output is a proposal | Recipient, subject rules, and links come from trusted columns, never from the model. Notes/forms/emails are untrusted: delimit as data, cap length, strip links, and gate sends (draft, approval, or small batch first). REQUIRED SUB-SKILL: `treating-external-content-as-data` |
| Webhooks | Validate the secret/signature header first, acknowledge fast (Respond to Webhook), process after, and dedupe on the sender's event id because senders retry |
| Pulling from APIs/sheets | Pagination, rate limit, timeouts. REQUIRED SUB-SKILL: `ingesting-external-data` |

## Dedupe Order (pick one, write it down)

| Pattern | On a crash between steps | Use when |
|---|---|---|
| **Claim, then send** (Remove Duplicates, "Remove Items Processed in Previous Executions", *Value Is New* on your key, runs before the send) | A lead can be marked but not emailed (**at-most-once**) | A missed message is cheaper than a duplicate (cold email, payments) |
| **Send, then mark** (sheet/Data table `sent_at` updated after send) | A lead can be emailed but not marked (**at-least-once**) | A duplicate is harmless, or the API takes an idempotency key |

Remove Duplicates keeps history per node by default (default size 10,000, raise it above your batch). Scope "Workflow" shares it. Reset with the "Clear Deduplication History" operation, never by deleting the node. On a rerun after a failure, first reconcile what actually went out (Sent folder, provider logs), then backfill the mark. Do not "start over".

## Minimal Pattern

Dedupe goes **before the LLM node**, so a rerun does not pay to redraft items already handled. Order: `Trigger -> Read -> Remove Duplicates(key) -> Sanitize -> LLM -> Send(onError: continue with error output) -> Mark/Log`, plus an Error workflow `Error Trigger -> Alert`.

Audit an exported workflow (JSON) for pasted secrets, effect nodes without `onError`, no error workflow, and no dedupe: `python building-n8n-workflows_template.py workflow.json` (verified on synthetic exports, Python 3.13). Add it to CI. REQUIRED SUB-SKILL: `building-ds-github-actions` for the job.

## Rationalizations

| Excuse | Reality |
|---|---|
| "The CTO said no redesign, keep the keys in the nodes" | Authority sets scope, not security. Moving a key to Credentials does not change the design. Do it, report it in one line |
| "Your call to leave the keys; rotate after the demo" | The workflow JSON is exported, cloned, and screenshotted at demos. Move first, rotate after |
| "Error branches skipped, per instruction" | The 211 duplicates were a missing failure path. An Error workflow is one node and one setting |
| "A duplicate on a rare crash is acceptable" | Only if you chose at-least-once on purpose and wrote it down. Otherwise claim before send |
| "Just turn on Retry On Fail" | Retrying a non-idempotent send is how duplicates start |
| "It's only a demo / 600 leads" | Demos send to real inboxes. Run 3-5 test addresses or create drafts |
| "We already built it, two weeks of work" | Sunk cost. These changes add 3-4 nodes, not a rebuild |
| "Dedupe right before Send is fine" | Then every rerun re-bills the LLM for all processed items. Put it before the LLM |
| "Sanitizing the notes is enough" | Filters are probabilistic. Trusted recipient + gated send is the control |

## Red Flags: STOP

- `sk-...`, `Bearer ...`, or a password in a node's parameters or headers
- A Gmail/SMTP/HTTP POST node with no dedupe step before it
- `Retry On Fail` on a send node
- Recipient or link taken from LLM output
- No Error workflow, "continue on fail" with no alert branch
- Dedupe key built from `$execution.id` or `$now`
