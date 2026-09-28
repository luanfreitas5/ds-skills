---
name: writing-conventional-commits
description: Use when committing changes in a data science or ML repo, when the working tree mixes refactors, new data, model swaps, or fixes, when someone asks to "just commit everything" or use a message like "update" or "wip", when bumping a version, tagging a release, or generating a CHANGELOG.
---

# Writing Conventional Commits

## Overview

In an ML repo the history is how you explain a metric change: "AUC moved from 0.78 to 0.81 — was it the new data, the new model, or the leakage fix?" **Core principle:** one commit = one logical change that can be reverted or bisected on its own, and its message is the CHANGELOG entry. The version and CHANGELOG are **generated** from those messages by commitizen, never written by hand.

## Types

Format: `<type>(<scope>)!: <subject>` — scope = module (`features`, `model`, `train`, `api`, `data`), subject in imperative, ≤ 72 chars, in the language the repo's history already uses.

| Type | Use for | Bump | CHANGELOG |
|---|---|---|---|
| `feat` | New capability; **adopting** a model (evidence in body) | MINOR | Adicionado |
| `fix` | Wrong behavior corrected — incl. leakage, wrong label, bad join | PATCH | Corrigido |
| `data` | Training/reference data changed: new snapshot, `.dvc` pointer, new period, label definition | PATCH | Dados |
| `perf` | Faster/lighter, same outputs | PATCH | Desempenho |
| `refactor` | Structure only — **same computed values/predictions** (rename, type hints, pandas → polars); signature change → `refactor!` | — | Refatorado |
| `exp` | Experiment not (yet) adopted: trying LightGBM, new hyperparameters | — | — |
| `docs` `test` `ci` `build` `chore` | Everything else | — | — |
| `!` or `BREAKING CHANGE:` footer | Public API/schema/artifact contract broken | MAJOR (MINOR in 0.x) | BREAKING CHANGE |

## Rules

1. **Split by logical change, not by file.** Before committing, list the changes in the diff; each one becomes a commit. Data + model + fix + refactor = 4 commits, even if "it's all one piece of work".
2. **Never bundle data with model.** `data(train): trocar base para customers_2025` and `feat(model): adotar lightgbm` are separate — otherwise the metric shift cannot be attributed or reverted.
3. **Model swap = `exp` until evaluated.** Use `feat(model)` only with evidence in the body: metric ± 95% CI vs baseline and the MLflow run id. No evidence → `exp(model)`, which does not bump and stays out of the CHANGELOG. Commit `data` and `fix` **before** `feat(model)`, and state in the body which data/fixes the metric was measured with — if the run mixed several changes, say so instead of crediting the model.
4. **Removing a leaking column is `fix`, not `refactor`** — predictions change. `refactor` means every computed value is unchanged; a changed public signature with the same values is `refactor!` + footer.
5. **Breaking change:** `!` in the header **and** a `BREAKING CHANGE: <what to do>` footer — commitizen only puts the footer in the CHANGELOG.
6. **"update", "wip", "misc", "changes" are never the message** — the commit-msg hook rejects them. Splitting takes one minute; do it and tell the user.
7. **Release only with commitizen:** `cz bump` (with `update_changelog_on_bump`) bumps `pyproject.toml`, writes `CHANGELOG.md`, commits and tags. Not installed? `uvx --from commitizen cz bump`. Never hand-edit the version, the CHANGELOG, or `git tag`.

## Implementation

`pyproject.toml` (tested with commitizen 4.19: `data` → PATCH, `feat` → MINOR, `!`/footer → MINOR in 0.x, only `exp` → "not eligible to be bumped"; `cz check` accepts a body after a blank line, rejects "update", "wip", uppercase scope, subject > 72 chars). On Windows add `.gitattributes` with `* text=auto eol=lf` **before** the first `cz bump`, or the bump commit stores CRLF files:

```toml
[tool.commitizen]
name = "cz_customize"
version_provider = "pep621"        # lê e grava [project].version
tag_format = "v$version"
update_changelog_on_bump = true
major_version_zero = true          # em 0.x, breaking change sobe MINOR
annotated_tag = true               # tag anotada (git describe, GitHub Releases)

[tool.commitizen.customize]
message_template = "{{change_type}}{% if scope %}({{scope}}){% endif %}: {{message}}"
schema = "<type>(<scope>)!: <subject>"
schema_pattern = '(?s)^(feat|fix|perf|refactor|data|exp|docs|test|ci|build|chore|style|revert|bump)(\([a-z0-9_-]+\))?!?: \S[^\r\n]{0,70}(\r?\n\r?\n.*|\s*)$'
bump_pattern = '^((BREAKING[\- ]CHANGE|\w+)(\([^)]*\))?!?):'
bump_map = {"^.+!$" = "MAJOR", "^BREAKING" = "MAJOR", "^feat" = "MINOR", "^fix" = "PATCH", "^perf" = "PATCH", "^data" = "PATCH"}
bump_map_major_version_zero = {"^.+!$" = "MINOR", "^BREAKING" = "MINOR", "^feat" = "MINOR", "^fix" = "PATCH", "^perf" = "PATCH", "^data" = "PATCH"}
commit_parser = '^(?P<change_type>feat|fix|perf|refactor|data|BREAKING CHANGE)(?:\((?P<scope>[^()\r\n]*)\))?(?P<breaking>!)?:\s(?P<message>.*)?'
changelog_pattern = '^(feat|fix|perf|refactor|data|BREAKING CHANGE)(\(.+\))?!?:'
change_type_map = {"BREAKING CHANGE" = "BREAKING CHANGE", "feat" = "Adicionado", "fix" = "Corrigido", "data" = "Dados", "perf" = "Desempenho", "refactor" = "Refatorado"}
change_type_order = ["BREAKING CHANGE", "Adicionado", "Corrigido", "Dados", "Desempenho", "Refatorado"]
```

Missing in the repo? Add it (plus the hook below) in a `build:` commit **first**, so the hook checks every commit after it.

`.pre-commit-config.yaml` — then `uvx pre-commit install --hook-type commit-msg`:

```yaml
- repo: https://github.com/commitizen-tools/commitizen
  rev: v4.19.0
  hooks:
    - id: commitizen   # rejeita "update" no estágio commit-msg
```

Two changes in the **same file**, no TTY for `git add -p` — stage an intermediate version:

```bash
F=src/features/engineering.py
FINAL=$(mktemp)
cp "$F" "$FINAL"                     # guarda a versão final
git show HEAD:"$F" > "$F"            # volta ao último commit
# edite $F aplicando SÓ a mudança A (ex.: remover a coluna que vaza)
git add "$F" && git commit -m "fix(features): remover cancel_status que vazava o rótulo"
cp "$FINAL" "$F"                     # o restante é a mudança B
git add "$F" && git commit -m "refactor(features)!: renomear compute_tenure" \
  -m "BREAKING CHANGE: compute_tenure removida; use compute_tenure_days."
```

Mixed hunks inside a **config file** (e.g. `train_path` + `model:` in `config.yaml`) split the same way: the data hunk goes with `data(...)`, the model hunk with `exp(model)`/`feat(model)`.

Release: `cz bump --dry-run` (review) → `cz bump` → `git log --oneline && git tag`.

## Common Mistakes

| Mistake | Fix |
|---|---|
| One commit "update" for the whole day | One commit per logical change (Rule 1) |
| `feat(model): switch to LightGBM and retrain on 2025 data` | Two commits: `data(train)` + `feat(model)`/`exp(model)` |
| `feat(data): add 2025 customers` | `data(train): ...` — own type, own CHANGELOG section |
| Model swap as `feat` with no metrics | `exp(model)`, or `feat` with metric ± CI + run id in body |
| "commitizen isn't installed, I'll write CHANGELOG by hand" | `uvx --from commitizen cz bump` |
| Manual `version = "0.4.0"` + `git tag v0.4.0` | `cz bump` does version, CHANGELOG, commit and tag together |
| `refactor` that changes predictions | `fix` / `feat` |

## Red Flags — Stop

- "It's all one piece of work anyway"
- "The user said 'update' is fine"
- "I'll write the CHANGELOG by hand, it's faster"
- One commit touching `src/models/`, `configs/` **and** a `.dvc` file
