---
name: building-llm-agents-langgraph
description: Use when building or hardening an LLM agent with tools, state, and loops (LangGraph, LangChain), when asked for "an autonomous agent that solves everything", when an agent has no limit on steps, cost, or time, when state lives in module globals, or when an agent is called ready because a happy-path demo worked.
---

# Building LLM Agents with LangGraph

## Overview

A happy-path demo shows that one trajectory worked once. It says nothing about loops, tool failures, cost, or what a hostile page can make the agent do. **Core principle:** an agent is a loop whose exit you must guarantee. Put the **budget in the graph state**, so every limit ends in a `finalize` node that returns partial work plus a `stop_reason`. Classify tools by risk. Prove behavior with **trajectory tests** that drive the graph with a scripted model.

**Violating the letter is violating the spirit:** "autonomous" means no confirmation for read-only work. It never means no exit condition.

## Agent Contract

| Element | Rule |
|---|---|
| State | `TypedDict` with `add_messages`, plus `steps`, `tool_calls`, `cost_usd`, `started_at`, `seen_calls`, `stop_reason`. One state per `thread_id` via a checkpointer. No module globals, and no mutable object captured in tool closures |
| Budget | Steps, tool calls, cost (from `usage_metadata`), wall time, and repeated identical calls. It is checked in the routers. Reaching a limit routes to `finalize`, which keeps partial work |
| `recursion_limit` | A safety net set **above** the budget (`2 * max_steps + 4`). Catching `GraphRecursionError` as your only limit throws away the state and covers steps only |
| Tool errors | Become a `ToolMessage` the model can react to. They never crash the run |
| Read tools that return external text (fetch, email, search snippets) | Truncated, delimited, and treated as untrusted (REQUIRED SUB-SKILL: treating-external-content-as-data) |
| Side-effect tools (write, send, SQL, pay) | Pinned arguments, and `interrupt()` for approval. That requires a checkpointer. A tool that is too dangerous to approve is removed, not just prompted against |
| Model | Injected already bound (`llm.bind_tools(tools)`), so tests can pass a scripted model |

Working graph: `building-llm-agents-langgraph_template.py`. It was verified with langgraph 1.2.18 and langchain-core 1.7.0. `GenericFakeChatModel` raises `NotImplementedError` on `bind_tools`, which is why the model is injected already bound.

## Trajectory Tests (the minimum before "ready")

They run in under a second, with no API key. Drive the compiled graph with `GenericFakeChatModel(messages=iter([...AIMessage(tool_calls=...)]))` and assert on the **sequence of `ToolMessage`s and `stop_reason`**. For endless scripts, generate a fresh `AIMessage` per step. `itertools.cycle` over one object repeats its id, so `add_messages` deduplicates it and the loop test lies.

1. Happy path: the expected tool order, then `answered`
2. An endless tool-calling model stops at `max_steps` with a partial answer
3. The same call repeated stops at `repeated_call`
4. A tool that raises becomes an observation, and the run still finishes
5. A side-effect tool pauses (`__interrupt__`). A rejection runs nothing, and approval runs it once
6. Two `thread_id`s do not share counters

Answer **quality** (is the report right, are the sources real?) needs a frozen eval set (REQUIRED SUB-SKILL: evaluating-llm-outputs). Version the system prompt (REQUIRED SUB-SKILL: versioning-prompts). Emit one span per model and tool call with cost (REQUIRED SUB-SKILL: tracing-llm-apps).

## Rationalizations

| Excuse | Reality |
|---|---|
| "Manager said no tests, the demo proves it works" | The demo proves one path. Six trajectory tests cost less than the smoke script you would write anyway, so commit them. A throwaway check leaves no evidence for tomorrow |
| "Director wants no stops or confirmations" | The budget does not ask anyone anything. It ends a runaway run with partial work. Approval is only for side effects |
| "`recursion_limit` is my budget" | It counts super-steps, not cost or time, and raising it loses the state. Use the state budget, with the limit as a backstop |
| "Cost cap needs per-user infra" | `usage_metadata` per call summed in state is about 5 lines |
| "Injection defense is too much for one night" | Truncate and delimit fetched text, and keep side-effect tools behind the gate. Removing a dangerous tool is good, and the other tools still need the gate |
| "Tool state in a closure object is per run" | It is invisible to checkpoints, resumes, and tests. Put it in graph state |
| "Eval set and tracing are over-engineering for tomorrow" | They can wait for the demo, but not for unsupervised users. Without spans and an eval, nobody can say whether a report's sources are real. Write that gate in the reply |
| "Flat price per 1k tokens is fine for the cost cap" | Input and output prices differ several-fold. Use the versioned price table from tracing-llm-apps |

## Red Flags: Stop

- Module-level dict mutated by nodes or tools
- `while`-style loop with no `max_steps` and no cost or time limit in state
- Only `GraphRecursionError` handling
- `requests.get(url).text` returned raw to the model
- A tool that writes, sends, or runs SQL, with no `interrupt` and no pinned args
- "Production-ready" with zero committed trajectory tests

**All of these mean: apply the agent contract and write the six trajectory tests before calling it ready.**
