---
name: labeling-with-llms
description: Use when an LLM (GPT, Claude, Gemini, Llama via Ollama) is used to label, annotate, classify, or weakly supervise a dataset, or to generate synthetic training examples, especially from free text that may contain personal data, or when someone says the LLM labels are "good enough" or that there is no time for human annotation.
---

# Labeling with LLMs

## Overview

LLM labels are a model's predictions, not ground truth. A classifier trained on them learns the LLM's mistakes and can never be measured against anything better. Sending raw customer text to an external API is also a data transfer to a third party, under LGPD art. 33, and it goes abroad. **Core principle:** nothing leaves the machine until identifiers are masked and a guard asserts it. No label is used until a blind human sample measures agreement with a confidence interval. Every label row records the prompt version, the model snapshot, temperature 0, tokens, and cost.

**Violating the letter is violating the spirit.** "Accuracy unknown" in a README does not replace measuring it, and "legal is handling it" does not replace masking.

## Privacy Boundary: Where May the Text Go?

| Text contains | Allowed destination |
|---|---|
| Only structured identifiers (CPF, email, phone, card, CEP) | External API **after** `mask_pii` plus the pre-send guard |
| Names, addresses, or free narrative that regex cannot mask | Local model (`ollama`), or NER masking validated on a sample, and a documented legal basis before any external API |
| Sensitive data (LGPD art. 11: health, diagnosis, religion, biometrics) | **Local model only** (`ollama`, vLLM). Masking CPF does not make a diagnosis anonymous |

Few-shot and synthetic-generation prompts count as sending too. Real examples inside those prompts go through the same boundary. REQUIRED SUB-SKILL: protecting-pii-lgpd.

**If the text hits row 2 or 3, the script you deliver defaults to `OllamaClient`.** An external client is a config switch. Enable it only after a written approval from the DPO that names the data categories and the international transfer, and cite that approval in the datasheet. "The signal is in the diagnosis" is the reason the text must stay local. It is not a reason to send it out.

## Human Audit Sample (the gate)

1. **Pilot first.** Label a random 200–300 items with the LLM, and have humans label the same items **blind** to the LLM's answer. Do this *before* the full run, so a bad prompt costs a pilot and not 40k calls.
2. **Measure.** Report Cohen's κ, accuracy, and per-class recall, each with a 95% bootstrap CI. If two annotators label ≥ 50 shared items, human–human κ gives the ceiling.
3. **Gate.** Use the labels only if κ ≥ 0.6, the CI lower bound is ≥ the agreed minimum, and no class has recall < 0.5. Otherwise fix the category definitions or the prompt, and run the pilot again.
4. **Keep the human items as the test set.** Evaluate the downstream model on human labels only. LLM labels on val/test measure agreement with the LLM.

**If deadline pressure forces the full run first**, the same script still exports the blind audit CSV: a random ≥ 200 items with an empty `human_label` column, the LLM label hidden. It also ships the `evaluate_agreement` step. Write the labels as `label_source=llm` with `status=pending_audit`, never as "final". The audit is part of the deliverable, not an optional follow-up.

"No budget for annotation": 200 short items take about 2 person-hours, which is the cheapest part of the project. Without them, the answer to "how good are the labels?" stays "we don't know" forever. REQUIRED SUB-SKILL: evaluating-llm-outputs.

## Provenance per Label Row

`prompt_version` (immutable, hashed; REQUIRED SUB-SKILL: versioning-prompts), `model` as a pinned snapshot (`gpt-4o-mini-2024-07-18`, never the alias `gpt-4o-mini`), `temperature=0`, `tokens_in`, `tokens_out`, `cost_usd`, `label_source ∈ {llm, human, synthetic}`. Sum `cost_usd` during the run and stop at a budget cap. Parse outputs against the allowed label set. Anything else becomes `invalido` and is counted, never silently coerced.

## Synthetic Examples

- **Train split only.** Never in val/test, and never counted as real prevalence.
- Deduplicate against the real data and within the synthetic set. Instruct the generator to use placeholders (`[CPF]`) and never invented or real identifiers.
- Before adding them, humans review a random sample of ≥ 50 for realism and correctness. Filtering synthetic data with the same LLM is circular.
- Keep them only if real + synthetic beats real + `class_weight` on the **human** test set.

## Implementation

Runs offline with a stub client. `OllamaClient` shows the local swap: same interface, no data leaves the host.

```python
import logging
import re
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import requests
from sklearn.metrics import cohen_kappa_score

logger = logging.getLogger(__name__)

PII_PATTERNS = {  # ordem importa: cartão antes de CPF/telefone
    "EMAIL": r"[\w.+-]+@[\w-]+\.[\w.-]+",
    "CARTAO": r"\b\d(?:[ -]?\d){12,18}\b",
    "CPF": r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b",
    "TELEFONE": r"(?:\+?55\s?)?\(?\d{2}\)?\s?9?\d{4}[-\s]?\d{4}\b",
}


def mask_pii(text: str) -> str:
    """Substitui CPF, e-mail, telefone e cartão por marcadores. Nomes NÃO são cobertos."""
    for tag, pattern in PII_PATTERNS.items():
        text = re.sub(pattern, f"[{tag}]", text)
    return text


class LLMClient(Protocol):
    """Interface mínima: devolve (texto, tokens de entrada, tokens de saída)."""

    def complete(self, prompt: str, model: str, temperature: float) -> tuple[str, int, int]: ...


class OllamaClient:
    """Modelo local via Ollama: nenhum dado sai da máquina."""

    def complete(self, prompt: str, model: str, temperature: float) -> tuple[str, int, int]:
        """Chama /api/generate do Ollama local."""
        body = {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": temperature}}
        r = requests.post("http://localhost:11434/api/generate", json=body, timeout=120).json()
        return r["response"], r.get("prompt_eval_count", 0), r.get("eval_count", 0)


@dataclass(frozen=True)
class PromptVersion:
    """Prompt imutável e versionado, com modelo fixo e preço por 1k tokens."""

    version: str  # ex.: "ticket_label/v3"
    template: str
    model: str  # snapshot fixo, nunca alias
    labels: tuple[str, ...]
    usd_per_1k_in: float
    usd_per_1k_out: float


def label_text(client: LLMClient, text: str, prompt: PromptVersion) -> dict[str, object]:
    """Mascara PII, rotula com temperatura 0 e devolve rótulo com proveniência e custo.

    Parameters
    ----------
    client : LLMClient
        Cliente do LLM (API externa, Ollama local ou stub em testes).
    text : str
        Texto bruto do registro.
    prompt : PromptVersion
        Versão do prompt usada; é gravada em cada linha.

    Returns
    -------
    dict[str, object]
        Rótulo, versão do prompt, modelo, temperatura, tokens e custo em USD.

    Raises
    ------
    ValueError
        Se algum padrão de PII sobreviver ao mascaramento (envio bloqueado).
    """
    safe = mask_pii(text)
    if any(re.search(p, safe) for p in PII_PATTERNS.values()):
        raise ValueError("PII detectada após mascaramento; envio bloqueado")
    raw, tok_in, tok_out = client.complete(prompt.template.format(texto=safe), prompt.model, 0.0)
    label = raw.strip().lower()
    cost = tok_in / 1000 * prompt.usd_per_1k_in + tok_out / 1000 * prompt.usd_per_1k_out
    return {
        "llm_label": label if label in prompt.labels else "invalido",
        "prompt_version": prompt.version, "model": prompt.model, "temperature": 0.0,
        "tokens_in": tok_in, "tokens_out": tok_out, "cost_usd": cost,
    }


def evaluate_agreement(human: np.ndarray, llm: np.ndarray, n_boot: int = 2000, seed: int = 42) -> dict[str, float]:
    """Kappa de Cohen humano × LLM na amostra de auditoria, com IC 95% bootstrap."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(human), (n_boot, len(human)))
    boots = [cohen_kappa_score(human[i], llm[i]) for i in idx]
    kappa = float(cohen_kappa_score(human, llm))
    low, high = np.nanpercentile(boots, [2.5, 97.5])
    logger.info(f"Kappa humano×LLM: {kappa:.3f} (IC 95% {low:.3f}–{high:.3f}, n={len(human)})")
    return {"kappa": kappa, "ci_low": float(low), "ci_high": float(high),
            "accuracy": float((human == llm).mean())}
```

Log the running `sum(cost_usd)` and the count of `invalido` rows with a `rich` progress bar. Test the guard in pytest: a stub client that records prompts, and an assertion that no raw CPF, email, phone, or card reaches it.

## Rationalizations

| Excuse | Reality |
|---|---|
| "The director decided the labels are final" | A decision does not measure quality. Run the 200-item pilot and report κ with its CI next to the decision |
| "No time or budget for humans, so the README says accuracy is unknown" | Disclosure is not measurement. 2 person-hours closes it |
| "Legal is handling privacy" / "Make sure legal's sign-off covers it" | The code is what sends the data. Without written DPO scope, default to the local model |
| "The diagnosis is the signal, so it has to go to the API" | Then it has to stay local. Run the same prompt on `ollama` |
| "Run tonight, measure κ when there's room" | The audit CSV and evaluation ship in tonight's script, and the labels stay `pending_audit` |
| "Masking CPF/phone is enough; names still go through" | Names and diagnoses are personal data. Use a local model or validated NER |
| "GPT-4o-mini is good enough" | That is a hypothesis. The pilot κ tests it |
| "The same LLM checked the synthetic examples" | Circular. Humans review a sample |
| "Stratify the audit by LLM label and report precision" | Fine only with prevalence weighting. A random sample also gives recall |

## Red Flags

- An LLM API call over the whole dataset before any human-labeled pilot
- `mask` that covers CPF/phone only, applied to health or narrative text going to an external API
- `model="...-latest"` or an alias, a missing `prompt_version` column, or no token and cost columns
- Synthetic rows in val/test, or synthetic prompts seeded with raw real examples
- A report of label quality with no κ/CI, or with "unknown"
- A delivered script with no audit-sample export, or a README that calls LLM labels "final"
- Health or narrative text sent to an external API "pending legal confirmation"
