---
name: building-ds-github-actions
description: Use when creating or fixing GitHub Actions workflows (ci.yml, tests.yml, docs.yml) for a Python data science or ML project, when CI is slow or reinstalls everything each job, when tests need datasets or trained models in CI, or when choosing a Python/OS test matrix.
---

# Building GitHub Actions for DS Projects

## Overview

DS workflows go slow and fragile in the same three ways: every job resolves and installs the full environment, the matrix multiplies heavy jobs by 9, and tests download real data (gigabytes, credentials, PII, flaky network). **Core principle:** CI installs exactly the locked environment from cache, runs the matrix only where behavior can differ, and tests against synthetic fixtures — real data never enters CI.

## Rules

| Rule | How |
|---|---|
| Locked, cached install | `astral-sh/setup-uv` with `enable-cache: true`, `cache-dependency-glob: uv.lock`; `uv sync --frozen` (fails if lock is stale instead of silently re-resolving) |
| Install only what the job needs | `uv sync --frozen --group lint` / `--group test` / `--group docs`, not `--all-groups` everywhere |
| Matrix only where it matters | Lint, type check, security, docs: one Python, `ubuntu-latest`. Tests: Python versions in `requires-python` on Linux; add Windows/macOS for a single Python only if the code touches paths, multiprocessing, or native wheels |
| No real data | Tests use synthetic fixtures (`conftest.py` generators or small committed parquet in `tests/fixtures/`). Model tests use a model trained in a session fixture on synthetic data (REQUIRED SUB-SKILL: testing-ml-code) |
| Real-data validation elsewhere | Scheduled/manual pipeline with OIDC-scoped read access (`aws-actions/configure-aws-credentials` + `role-to-assume`), never long-lived keys in PR jobs |
| Coverage is a gate and a report | `pytest --cov --cov-fail-under=80` + `codecov/codecov-action` (token from secrets) once, not per matrix cell |
| Cheap and safe defaults | `concurrency` cancels superseded runs; `permissions: contents: read`; `timeout-minutes` on every job |

Which checks go in pre-commit vs CI, and the `pyproject.toml`/`Makefile` they call: REQUIRED SUB-SKILL: scaffolding-ds-projects.

## tests.yml

```yaml
name: tests
on:
  pull_request:
  push:
    branches: [main]
concurrency:
  group: tests-${{ github.ref }}
  cancel-in-progress: true
permissions:
  contents: read
jobs:
  pytest:
    runs-on: ubuntu-latest
    timeout-minutes: 20
    strategy:
      fail-fast: false
      matrix:
        python-version: ["3.10", "3.11", "3.12"]
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6
        with:
          enable-cache: true
          cache-dependency-glob: uv.lock
          python-version: ${{ matrix.python-version }}
      - run: uv sync --frozen --group test
      - run: uv run pytest -m "not slow" --cov --cov-report=xml --cov-fail-under=80
      - if: matrix.python-version == '3.12'
        uses: codecov/codecov-action@v5
        with:
          files: coverage.xml
          token: ${{ secrets.CODECOV_TOKEN }}
```

`ci.yml` follows the same skeleton with one job, no matrix: `uv sync --frozen --group lint` then `ruff check`, `ruff format --check`, `basedpyright`, `bandit -r src`, `vulture`, `xenon`, `interrogate`, `pip-audit`. `docs.yml`: `--group docs`, `mkdocs build --strict`, deploy with `actions/upload-pages-artifact` + `actions/deploy-pages` on `main` only.

Pin action majors and let `dependabot.yml` (`package-ecosystem: github-actions`) bump them.

## Common Mistakes

| Mistake | Fix |
|---|---|
| `uv sync` without `--frozen` | `--frozen`: CI uses exactly `uv.lock` |
| `--all-groups` in every job | Group per job (`lint`, `test`, `docs`) |
| 3 OS × 3 Python for lint and tests | Lint once; tests on Python versions; one extra OS only with a reason |
| `aws s3 cp` of the training set in tests | Synthetic fixtures; real-data checks in a scheduled job with OIDC |
| `skipif(not Path("data/...").exists())` | Fixture generates data; nothing skips silently |
| Coverage uploaded from 9 cells | Upload once |
| No `timeout-minutes` | A hung job burns 6 h of minutes |
