---
name: auditing-python-dependencies
description: Use when adding, upgrading, or removing a Python dependency (uv add, pip install, pyproject.toml, requirements.txt), when asked to "just add" a library for a small helper, when pip-audit or Dependabot reports a CVE or GHSA advisory, when CI is red on a vulnerability scan, or during a periodic review of a project's dependencies.
---

# Auditing Python Dependencies

## Overview

Every dependency is code you ship, update, and audit forever — including its transitive tree. **Core principle:** a dependency enters only with a written justification, lands in the narrowest group that needs it, and is scanned on a schedule, not only when someone opens a PR. New CVEs appear in unchanged lock files.

## Adding a Dependency

Answer in order; stop at the first "yes":

1. **Stdlib or an existing dependency does it?** Cosine similarity = `numpy`; slugify = `unicodedata` + `re`; PSI ≈ 20 lines of numpy; `train_test_split` is already in scikit-learn. A library for a < 30-line helper is not justified.
2. **Only tests, lint, docs, or notebooks need it?** → `[dependency-groups]` (`dev`, `docs`, `notebooks`), never `[project].dependencies`. Docker/production: `uv sync --locked --no-dev` (plus `--no-group <name>` for other groups).
3. **Otherwise** check cost before `uv add`: `uv tree --package <pkg>` (transitive size), maintenance (last release, open advisories), license. Prefer the thin engine (`fastexcel` for `pl.read_excel`) over a framework (`pandas` + `openpyxl`, `evidently` for one metric).

Then record the reason in `[tool.dependency-rationale]` — enforced by the test below. Stale entries fail too, so removals stay clean. Anything you `import` directly is declared, even if it already arrives transitively (`numpy` via scikit-learn) — transitive presence can vanish on any upgrade.

"The lead asked for it" is not a justification. Deliver the stdlib version, state the trade-off in one line, and offer `uv add` if they still want it.

## Implementation

```toml
[project]
dependencies = ["polars>=1.0", "fastexcel>=0.11", "numpy>=1.26"]

[dependency-groups]
dev = ["pytest>=8.0", "pytest-cov>=5.0", "ruff>=0.5", "pip-audit>=2.7"]
notebooks = ["jupyterlab>=4.2", "matplotlib>=3.8", "seaborn>=0.13"]

[tool.dependency-rationale]
polars = "DataFrame principal; lazy + streaming"
fastexcel = "engine de pl.read_excel (.xlsx do financeiro)"
numpy = "uso direto em PSI e similaridade"
pytest = "testes"
pytest-cov = "cobertura >= 80%"
ruff = "lint + format"
pip-audit = "auditoria de CVEs"
jupyterlab = "notebooks exploratórios"
matplotlib = "figuras"
seaborn = "figuras estatísticas"
```

```python
# tests/test_dependencies.py
"""Garante que toda dependência declarada tenha justificativa registrada."""

import re
import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*")


def _normalize(requirement: str) -> str:
    """Extrai o nome normalizado (PEP 503) de um requisito PEP 508.

    Parameters
    ----------
    requirement : str
        Requisito, ex.: ``"polars[pyarrow]>=1.0"``.

    Returns
    -------
    str
        Nome normalizado, ex.: ``"polars"``.

    Raises
    ------
    ValueError
        Se o requisito não começar com um nome de pacote válido.
    """
    match = _NAME.match(requirement.strip())
    if match is None:
        raise ValueError(f"Requisito inválido no pyproject: {requirement!r}")
    return re.sub(r"[-_.]+", "-", match.group(0)).lower()


def test_every_dependency_has_rationale() -> None:
    """Toda dependência (runtime e grupos) precisa de justificativa; nenhuma órfã."""
    config = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    declared = {_normalize(r) for r in config["project"].get("dependencies", [])}
    for group in config.get("dependency-groups", {}).values():
        declared |= {_normalize(r) for r in group if isinstance(r, str)}
    table = config.get("tool", {}).get("dependency-rationale")
    assert table is not None, "Tabela [tool.dependency-rationale] ausente no pyproject"
    rationale = {_normalize(k) for k in table}

    assert not (missing := sorted(declared - rationale)), f"Sem justificativa: {missing}"
    assert not (stale := sorted(rationale - declared)), f"Justificativa órfã: {stale}"
```

Scheduled audit of the **lock file** (a venv scan sees whatever was `pip install`ed by hand) — `.github/workflows/security.yml`:

```yaml
name: security
on:
  schedule: [{ cron: "0 6 * * 1" }]   # semanal: CVEs novos surgem sem mudança de código
  pull_request:
    paths: ["pyproject.toml", "uv.lock"]
  workflow_dispatch:
jobs:
  audit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6
      - run: uv lock --check
      - run: uv export --frozen --all-groups --no-hashes --no-emit-project -o req-audit.txt
      # Ignorar só por ID, com motivo, issue e data de revisão:
      # --ignore-vuln GHSA-xxxx  # sem fix; não alcançável (#123); revisar 2026-12-01
      - run: uvx pip-audit -r req-audit.txt --no-deps --disable-pip
```

`.github/dependabot.yml`:

```yaml
version: 2
updates:
  - package-ecosystem: "uv"
    directory: "/"
    schedule: { interval: "weekly" }
    groups:
      minor-and-patch: { update-types: ["minor", "patch"] }
  - package-ecosystem: "github-actions"
    directory: "/"
    schedule: { interval: "monthly" }
```

Both are part of the project baseline (REQUIRED SUB-SKILL: scaffolding-ds-projects). "We don't have Dependabot, don't bother" → add it anyway; it is two files and zero runtime cost. Say so in one line.

## Handling a Reported Vulnerability

| Situation | Action |
|---|---|
| Fix version exists, transitive | `uv lock --upgrade-package <pkg>`; re-run tests; commit `uv.lock` |
| Fix version exists, direct | Raise the floor (`uv add "pkg>=fix"`), own commit, run metric regression tests (model libs change predictions) |
| Upgrade blocked by a parent's upper bound | Upgrade the parent; else `[tool.uv] constraint-dependencies = ["pkg>=fix"]` |
| No fix available | Confirm the advisory (`pip-audit --desc`, osv.dev); find who pulls it (`uv tree --invert --package <pkg>`); assess reachability (is the vulnerable path used or exposed?); ignore **by ID** with reason, issue, review date — or replace the parent |

"It's only transitive" / "it's mlflow junk" is not a reason: transitive packages are installed and run (`mlflow server` runs Flask/werkzeug). **Never drop the audit step** or ignore a vuln that has a fix — the fix is as fast as the ignore.

## Common Mistakes

| Mistake | Fix |
|---|---|
| Library for a 5-line helper (`scipy`, `python-slugify`, `pandas` "for tests") | stdlib / existing dep; state the trade-off |
| `pytest`, `ruff`, `jupyter` in `[project].dependencies` | Move to `[dependency-groups]` now, not "after the release" — `uv sync` installs `dev` by default, so local runs and CI are unchanged |
| `--ignore-vuln` / delete audit step to make CI green | Upgrade; ignore only unfixable, by ID, dated |
| Audit only on PRs | Weekly `schedule` + Dependabot |
| `pip-audit` on a local venv | Audit `uv export` of the lock file |
| Dependency removed from code, left in `pyproject.toml` | `uv remove`; rationale test flags the orphan |
