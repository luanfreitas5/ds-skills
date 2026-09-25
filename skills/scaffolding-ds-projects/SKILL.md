---
name: scaffolding-ds-projects
description: Use when creating a new data science or ML project, setting up pyproject.toml, uv, pre-commit, CI workflows, Makefile, or the src/ layout, or when asked to put type checking, security scans, or the full test suite into pre-commit hooks.
---

# Scaffolding Data Science Projects

## Overview

**Core principle:** split quality gates by speed. Pre-commit runs only checks that finish in seconds on staged files; CI runs everything heavy and is the real gate (branch protection: green CI before merge). Slow pre-commit trains people to use `--no-verify`, which removes *all* gates.

## Gate Placement

| Check | pre-commit | CI |
|---|---|---|
| `ruff` lint + format | yes | yes |
| `detect-secrets`, `nbstripout`, large-file / merge-conflict checks | yes | yes |
| `commitizen` message check (commit-msg stage) | yes | — |
| Smoke tests (`pytest -m smoke`, < 10 s) | optional | yes |
| `basedpyright` | **no** | yes |
| `bandit`, `vulture`, `refurb`, `xenon`/`radon`, `interrogate` | **no** | yes |
| Full `pytest --cov --cov-fail-under=80` | **no** | yes |
| `pip-audit` | **no** | yes (+ weekly schedule) |

User asks for heavy checks in pre-commit "so nothing bad gets committed"? Explain the trade-off, deliver the split config, and offer `make check` (runs the CI suite locally) for the extra safety they want. Commits are local and cheap to fix; merges are the gate.

## Templates (in this skill directory)

- `pyproject_template.toml` — uv, dependency groups, all tool configs
- `pre-commit_template.yaml` — fast hooks only
- `Makefile_template` — `install`, `lint`, `typecheck`, `test`, `check`, `train`, `reproduce`

Run `pre-commit autoupdate` after copying — pinned `rev`s age quickly.

## Layout Rules

- Code in `src/<functional packages>/`; `src` is on the import path (`pythonpath = ["src"]` for pytest, `extraPaths` for basedpyright), so imports read `from features.engineering import ...` — never `from src.features ...`.
- **Never name a package after a stdlib module** — `src/logging/`, `src/io/`, `src/random/`, `src/types/` shadow `logging`, `io`, ... once `src` is on `sys.path`, breaking third-party libraries. Use `log_config/`, `io_utils/`, etc.
- Every package has `__init__.py` with a pt-BR docstring listing its modules.
- `data/`, `models/`, `logs/`, `.env` in `.gitignore`; `uv.lock` committed; `.gitattributes` with `* text=auto eol=lf` and `*.ipynb` diff settings.
- Required root files: `pyproject.toml`, `uv.lock`, `Makefile`, `.pre-commit-config.yaml`, `.gitignore`, `.gitattributes`, `.env.example`, `CHANGELOG.md`, `LICENSE`, `README.md`.
- `.github/workflows/`: `ci.yml` (lint, types, security, complexity), `tests.yml` (pytest + coverage upload), `docs.yml`; `.github/dependabot.yml`.

## Common Mistakes

| Mistake | Fix |
|---|---|
| basedpyright/bandit/full pytest in pre-commit | Move to CI + `make check` |
| `[build-system] hatchling` with no package config for a multi-package `src/` | `[tool.uv] package = false` for apps; build config only if publishing |
| `from src.module import x` | `pythonpath = ["src"]` |
| Dependencies without lock file | `uv lock` and commit |
| Dev tools in runtime deps | `[dependency-groups] dev = [...]` |
| Every stack library added "just in case" | Only what the project uses; justify each |
