---
name: versioning-prompts
description: Use when a prompt used in production changes often, when someone asks for a "one-line" or "typo" edit to a prompt, when a prompt lives as a string constant in code, or when nobody can tell which prompt, model, or temperature produced a stored LLM output.
---

# Versioning Prompts

## Overview

A prompt is a model artifact: changing one word changes outputs just like retraining. If you edit a string in place and deploy, you lose the link between outputs and the prompt that produced them, and every "last Tuesday was wrong" becomes guesswork from deploy timestamps. **Core principle:** every released prompt version is an immutable, hashed file. Every output row stores that version. No version reaches production without an automated paired regression on the evaluation set.

**Violating the letter is violating the spirit:** a typo fix is a new version.

## The Contract

| Element | Rule |
|---|---|
| Location | `prompts/<task>/v<N>.yaml` holding `template`, `model` (pinned snapshot, never an alias), `params` (temperature, max_tokens, ...) |
| Immutability | `prompts/prompts.lock.json` maps `<task>/v<N>` → SHA-256. Loading a file whose hash differs from the lock **raises**. Any change, even a typo, goes into `v<N+1>` |
| Active version | Selected in config (`configs/llm.yaml: summarize: summarize/v4`). Rollback means changing that one line |
| Output provenance | Every stored output gets `prompt_id`, `prompt_sha256`, `model_served` (from the API response), `params`, `git_sha`. Add these columns **in the same PR** as the first versioned prompt, never "next week" |
| Regression | CI runs production vs candidate on the frozen eval set on every PR touching `prompts/` or `configs/llm.yaml`, and fails the build on regression |
| New requirement | A new instruction ("never include customer names") gets eval items that check it (`must_not_contain`, tag `pii`) **before** the candidate is judged. An average metric that cannot see the requirement does not count as evidence |

Eval set design, judge calibration, and significance tests are covered by REQUIRED SUB-SKILL: evaluating-llm-outputs. Log offline eval runs as MLflow runs tagged with `prompt_id`, `prompt_sha256`, and eval set version, following REQUIRED SUB-SKILL: tracking-experiments-mlflow.

## Implementation

```python
import hashlib
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PromptVersion:
    """Versão imutável de um prompt: template + modelo + parâmetros + hash."""

    prompt_id: str
    template: str
    model: str
    params: dict
    sha256: str


def compute_prompt_hash(spec: dict) -> str:
    """Hash de tudo que altera a saída (template, modelo, parâmetros)."""
    canonical = json.dumps({k: spec[k] for k in ("template", "model", "params")}, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_prompt(prompts_dir: Path, prompt_id: str) -> PromptVersion:
    """Carrega `prompts/<tarefa>/<versao>.yaml` e confere o hash no prompts.lock.json."""
    spec = yaml.safe_load((prompts_dir / f"{prompt_id}.yaml").read_text(encoding="utf-8"))
    lock = json.loads((prompts_dir / "prompts.lock.json").read_text(encoding="utf-8"))
    digest = compute_prompt_hash(spec)
    if lock.get(prompt_id) != digest:
        logger.error("Prompt %s ausente do lock ou alterado após o release", prompt_id)
        raise ValueError(f"Prompt {prompt_id} não confere com o lock; crie uma nova versão.")
    return PromptVersion(prompt_id, spec["template"], spec["model"], spec["params"], digest)


def release_prompt(prompts_dir: Path, prompt_id: str) -> str:
    """Registra o hash de uma versão nova; recusa sobrescrever uma versão já lançada."""
    lock_path = prompts_dir / "prompts.lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8")) if lock_path.exists() else {}
    spec = yaml.safe_load((prompts_dir / f"{prompt_id}.yaml").read_text(encoding="utf-8"))
    digest = compute_prompt_hash(spec)
    if lock.get(prompt_id, digest) != digest:
        raise ValueError(f"Versão {prompt_id} já existe com outro conteúdo; use a próxima versão.")
    lock[prompt_id] = digest
    lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True), encoding="utf-8")
    logger.info("Prompt %s registrado (sha256=%s)", prompt_id, digest[:12])
    return digest


def score_item(output: str, item: dict) -> bool:
    """Acerto por item: contém o esperado e nenhum termo proibido."""
    text = output.lower()
    return item["expected"].lower() in text and not any(t.lower() in text for t in item.get("must_not_contain", []))


def run_prompt_regression(
    generate: Callable[[PromptVersion, str], str],
    baseline: PromptVersion,
    candidate: PromptVersion,
    eval_path: Path,
    max_drop: float = 0.02,
    seed: int = 42,
) -> dict:
    """Compara candidato × produção nos mesmos itens (bootstrap pareado + fatias por tag)."""
    items = [json.loads(line) for line in eval_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    base = np.array([score_item(generate(baseline, it["input"]), it) for it in items])
    cand = np.array([score_item(generate(candidate, it["input"]), it) for it in items])
    idx = np.random.default_rng(seed).integers(0, len(items), (5000, len(items)))
    diffs = cand[idx].mean(axis=1) - base[idx].mean(axis=1)
    tags = {t for it in items for t in it.get("tags", [])}
    masks = {t: np.array([t in it.get("tags", []) for it in items]) for t in tags}
    slices = {t: float(cand[m].mean() - base[m].mean()) for t, m in masks.items()}
    ci_low = float(np.percentile(diffs, 2.5))
    report = {
        "baseline": f"{baseline.prompt_id}@{baseline.sha256[:12]}",
        "candidate": f"{candidate.prompt_id}@{candidate.sha256[:12]}",
        "diff": float(cand.mean() - base.mean()),
        "ci_low": ci_low,
        "slices": slices,
        "passed": ci_low >= -max_drop and all(d >= -max_drop for d in slices.values()),
    }
    logger.info("Regressão de prompt: %s", report)
    return report
```

In CI, `generate` calls the real model and the job exits non-zero when `passed` is False. In unit tests, stub `generate`. Also add a test that loads every prompt listed in the lock, so an in-place edit fails the build before it is deployed. Set `max_drop` and the slices you care about **before** seeing results.

## Rationalizations

| Excuse | Reality |
|---|---|
| "It's a one-line / typo change" | Any token changes outputs. New file, new version, same regression run: about 5 minutes |
| "We'll add `prompt_version` to the table Monday" | Every output written before Monday has no provenance. Ship the columns with the change |
| "Note the deploy time in Slack" | Deploys, rollbacks, and canaries overlap. The row must carry its own version |
| "Run the eval script manually and compare averages" | Manual steps get skipped on Friday at 6pm. Use a paired CI gate with a threshold set in advance |
| "The ROUGE drop is expected" | Say that before the run, as a gate on a named slice. Afterwards it is a rationalization |
| "Git history is our prompt versioning" | Git cannot tell you which SHA served a given row. The hash stored on the row can |
| "P1: hot-edit the live version and update the lock" | Roll back the active version in config to the previous gated version. Ship the fix as `v<N+1>` through the gate. Never rewrite an existing entry in the lock |

## Common Mistakes

| Mistake | Fix |
|---|---|
| `SYSTEM_PROMPT = f"""..."""` in code | Versioned YAML file loaded by id |
| Editing `v3.yaml` in place | Lock check raises; create `v4.yaml` |
| Model alias (`gpt-4o-mini`) or temperature outside the prompt file | Pin the snapshot. Model and params are part of the hash |
| Storing only the requested model | Also store `model_served` from the response |
| Eval tuned on the same items every week | Frozen holdout; add new eval versions, never edit old ones |
| New rule, or a production failure, with no eval item that checks it | Add tagged items (`must_not_contain`, `language`, ...) before judging the candidate |
