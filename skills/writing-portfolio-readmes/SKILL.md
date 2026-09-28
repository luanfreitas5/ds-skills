---
name: writing-portfolio-readmes
description: Use when writing or polishing the README of a data science / ML project for a portfolio, recruiters, a paper's companion repository, or a public GitHub release, when asked to "make the README impressive" or "list all the technologies", when adding badges, or when writing installation and "how to run" sections.
---

# Writing Portfolio READMEs

## Overview

A reader decides in 30 seconds from the first screen, and a reviewer then tries to run it. **Core principle:** the README is a claim plus the evidence for it — problem, data, result with uncertainty against a baseline, and commands that reproduce that result from a clean clone. Technology lists and logo badges are not evidence.

## Required Structure (in this order)

Fill [readme_template.md](readme_template.md). Every section is required. If one can't be filled yet (no Model Card, no slice analysis), keep the heading and state what is missing — never drop it silently or fill it with invented numbers.

1. **Title + result line** — the headline number with its CI and the baseline's number, in the first screen. The baseline is the strongest *simple* model (logistic regression, seasonal naive), not `DummyClassifier` — beating chance is not a result (REQUIRED SUB-SKILL: establishing-baselines).
2. **Badges** — only the four evidence badges plus Python versions (table below).
3. **Problem** — who decides what, the cost of FP vs FN, why this metric (REQUIRED SUB-SKILL: evaluating-models-rigorously).
4. **Data** — source, license, size, period, prevalence; link the Datasheet.
5. **Results** — protocol, baseline vs model with CI and the method used for the CI, significance, worst slice, one figure; link the Model Card (REQUIRED SUB-SKILL: writing-model-cards-datasheets).
6. **Reproduce** — `uv sync --locked` + make targets inside `<!-- verify:start/end -->`, expected output, the environment that produced it, runtime; point to the committed metrics file each number comes from.
7. **Limitations**, then **License / citation** (`CITATION.cff` for a paper).

Libraries appear only where they explain a decision ("LightGBM: 3× faster than XGBoost at equal PR-AUC"), not as a stack table or logo wall.

## Badges

| Badge | Source | Real only if |
|---|---|---|
| CI | `github.com/<owner>/<repo>/actions/workflows/ci.yml/badge.svg` | workflow exists and is green on `main` |
| Coverage | `codecov.io/gh/<owner>/<repo>/graph/badge.svg` | CI uploads coverage (REQUIRED SUB-SKILL: building-ds-github-actions) |
| Docs | `docs.yml` status badge, linked to the Pages site | site is published |
| License | `img.shields.io/github/license/<owner>/<repo>` | GitHub detects the license — a stub `LICENSE` renders "not identifiable" |
| Python | static badge listing the **CI matrix** versions | each version is tested in CI |

A badge that isn't real yet: fix the project (add the workflow, upload coverage) or leave it out and say so in the handoff. Never ship a badge that renders "no status" or "unknown". Workflow badges show the default branch — check its real name (`git branch --show-current`), and add them once the first run on it is green.

## Clean-Environment Test (required before handoff)

"Works on my machine" hides undeclared dependencies (imported but missing from `pyproject.toml`), files never committed, and a global Python. Run the README's own commands in a fresh clone with a scrubbed environment, at the **minimum** Python you claim:

```bash
python skills/writing-portfolio-readmes/verify_readme.py path/to/repo --python 3.10
# antes do primeiro commit: acrescente --include-uncommitted
```

[verify_readme.py](verify_readme.py) clones `HEAD` (or copies tracked + non-ignored files) to a temp dir, strips `VIRTUAL_ENV`/`CONDA_PREFIX`/`PYTHONPATH`/user site-packages, sets `UV_PYTHON`, and runs the `bash` blocks in the verify region, stopping at the first failure. Tested: it catches `ModuleNotFoundError: pandas` for a project where `make data && make train` passes on the author's machine.

Then compare the run's numbers with the README. Different Python versions resolve different wheels from the same `uv.lock` and can shift metrics (observed: CV AUC 0.679 on 3.14 vs 0.673 on 3.10), so state which environment produced the published numbers.

Can't run it (needs GPU, paid API, credentials)? Run everything up to that step, provide a small sample path (`make train SAMPLE=1`), and write in the handoff exactly what was not executed.

## Common Mistakes

| Mistake | Fix |
|---|---|
| "Tecnologias" table + pandas/sklearn/polars badges, no result | Result line first; stack only where it justifies a choice |
| `pip install -r requirements.txt` with no such file, or `uv sync` without a lock | `uv sync --locked` with a committed `uv.lock`; verify in clean clone |
| Makefile calls bare `python` | `uv run python ...` |
| "AUC 0.85" alone | Baseline + CI + method + p-value vs baseline |
| "Python 3.10+" badge, only tested on 3.12 | Badge = CI matrix; verify at the minimum version |
| Hiding that the model doesn't beat the baseline | Say it; it's a finding, and reviewers will find the JSON |
| Metrics file not committed / link 404 | Commit `reports/metrics/*.json`; README cites it |
| `.gitignore` has `data/` → also hides `src/data/`; clean clone fails | Anchor to root: `/data/` |
| README checked only by reading it | Run `verify_readme.py`; reading is not testing |
