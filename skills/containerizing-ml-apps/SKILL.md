---
name: containerizing-ml-apps
description: Use when writing or reviewing a Dockerfile, .dockerignore, or docker-compose.yml for an ML API, Streamlit app, batch job, or MLflow server, when an image is several GB or runs as root, when someone wants to COPY the model into the image or pip install without the lock file, or when MLflow runs in a container with SQLite.
---

# Containerizing ML Apps

## Overview

An image is **code + locked dependencies**, nothing else. The model is a separately versioned artifact that the container loads at startup from `MODEL_URI` (read-only volume or MLflow Registry alias). **Core principle:** a new model must never require a new image, and a new image must never silently change a library version the model was pickled with.

## Rules

| Rule | How |
|---|---|
| Locked install | Multi-stage: builder copies only `pyproject.toml` + `uv.lock`, runs `uv sync --locked --no-dev --no-install-project`; runtime copies `/app/.venv`. Never `pip install` from `pyproject.toml`/`requirements.txt` without the lock. A hand-written `requirements.txt` next to `uv.lock` is stale by definition: delete it, or regenerate with `uv export --locked --no-dev` |
| `--locked`, not `--frozen` | `--locked` fails the build if the lock is stale; `--frozen` installs a stale lock silently |
| Versions match the project | Python tag satisfies `requires-python`; MLflow server tag == `mlflow` in `uv.lock`. Template tags are examples: re-pin to current patch releases |
| Pinned bases | Patch-level tags (`python:3.12.11-slim-bookworm`, `ghcr.io/astral-sh/uv:0.8.15`), same tag in every stage; add `@sha256:` digest for production. Never `latest`, `3.12`, `uv:0.8` |
| Non-root | `useradd --system --uid 10001`, `USER app` before `CMD`. Code and venv stay root-owned (no `--chown`): the runtime user reads, never writes |
| Model outside the image | `MODEL_URI` env var; compose mounts `./models:/models:ro` or points to `models:/<name>@champion`. No `COPY models/` |
| Allowlist `.dockerignore` | Ignore `*`, re-include only `pyproject.toml`, `uv.lock`, `src/`, `app/`, `configs/`, `docker/`. Keeps host `.venv`, `.env`, `data/`, `mlruns/`, `.git` out |
| MLflow = server + database | SQLite backend store (project default, never the `./mlruns` file store): `--backend-store-uri=sqlite:////mlflow/mlflow.db` (4 slashes = absolute) **inside** the named volume, with `--artifacts-destination` + `--serve-artifacts` on the same volume. Say Postgres is needed once it is shared by several users: then add a `db` service, `psycopg2-binary` in the MLflow image, DB healthcheck + `depends_on: condition: service_healthy` |
| Local-only ports | `"127.0.0.1:5000:5000"`; MLflow has no auth by default |
| Logs to stdout | Containers log to stdout (`RichHandler`/`StreamHandler`); the file handler is off or writes to a `tmpfs` (`tmpfs: ["/app/logs:uid=10001,gid=10001"]`), else `read_only` + non-root fails at startup |
| Startup order | API `depends_on: mlflow: service_healthy` only when `MODEL_URI` is a `models:/` URI; a mounted file must not make the API wait for MLflow |
| Secrets | `${VAR:?message}` from `.env`; never in the Dockerfile or `ENV` |
| Measure size | `docker images`, `docker history <img>`; state the number. Target: API image < 1 GB |

## Templates (in this skill directory)

- `Dockerfile_template` — API: uv stage → builder → slim non-root runtime, `HEALTHCHECK` on `/health` (the app must expose it)
- `mlflow.Dockerfile_template` — official MLflow image + non-root user + writable `/mlflow` (SQLite db and artifacts)
- `docker-compose_template.yml` — `mlflow` (SQLite) + `api`, healthchecks, `read_only` API, volumes
- `dockerignore_template` — allowlist

Validate before claiming it works: `docker compose config -q`, `hadolint Dockerfile`, then `docker compose up --build`, `docker images` (size), `docker compose exec api id -u` (not 0), and one real `/predict` request. Could not run Docker? Say exactly which of these were not executed.

## Loading the Model from `MODEL_URI`

```python
import os
from pathlib import Path

import joblib
from sklearn.pipeline import Pipeline


def load_model(uri: str | None = None) -> Pipeline:
    """Carrega o modelo a partir de um arquivo montado ou de um alias do MLflow Registry.

    Parameters
    ----------
    uri : str | None
        Caminho local ou URI ``models:/<nome>@<alias>``; padrão: variável ``MODEL_URI``.

    Returns
    -------
    Pipeline
        Pipeline ajustado no treino (pré-processamento + modelo).

    Raises
    ------
    RuntimeError
        Se nenhuma URI for informada.
    FileNotFoundError
        Se o arquivo local não existir.
    """
    uri = uri or os.environ.get("MODEL_URI")
    if not uri:
        raise RuntimeError("MODEL_URI não definida; informe um caminho ou models:/<nome>@<alias>")
    if uri.startswith("models:/"):
        import mlflow.sklearn  # mlflow-skinny basta na imagem da API

        return mlflow.sklearn.load_model(uri)
    path = Path(uri)
    if not path.exists():
        raise FileNotFoundError(f"Modelo não encontrado em {path}; monte o volume de modelos")
    return joblib.load(path)  # artefato próprio, confiável
```

Call it once in the FastAPI `lifespan` (REQUIRED SUB-SKILL: serving-models-fastapi).

## "Just Put the Model in the Image, It's Only a Demo"

Deliver the volume/registry version anyway. It is the same number of lines, runs identically tomorrow, and avoids: a 220 MB layer re-pushed per retrain, an image tag that no longer says which model it serves, and rollbacks that need a rebuild. Tell the user the demo command is unchanged (`docker compose up`). "Self-contained" is met by compose + `./models` mount, not by baking.

## Where Image Size Comes From

| Cause | Fix |
|---|---|
| `xgboost` on Linux pulls `nvidia-nccl-cu*` (hundreds of MB) | `xgboost-cpu` at the **same version** used in training (changes the lock: re-run tests, not the night before a demo) |
| `torch` default wheel is CUDA | CPU index via `[tool.uv.sources]` for CPU-only serving |
| Full `mlflow` in the API image | Main deps: `mlflow-skinny`; `[dependency-groups] tracking = ["mlflow"]` for local UI/training; server image has full `mlflow` |
| Jupyter, optuna, plotting libs in the API | `[dependency-groups] train = [...]`; `uv sync --no-dev` excludes non-default groups |
| `python:3.x` (non-slim), build tools in runtime | `-slim` runtime, compile only in builder |
| Model or data in the build context | Allowlist `.dockerignore` |

## Common Mistakes

| Mistake | Fix |
|---|---|
| `COPY . .` | Copy manifests first, then `src/`, `app/`, `configs/` |
| `COPY --chown=app:app src/` | Root-owned code; the app user only needs read |
| `uv sync --frozen` | `--locked` |
| Base image tag `3.11-slim`, `uv:0.8`, `mlflow:latest` | Full patch tag (+ digest) |
| Official MLflow image with Postgres URI (shared setup) | `ModuleNotFoundError: psycopg2` → add `pip install psycopg2-binary` to `mlflow.Dockerfile_template` |
| Named volume unwritable by non-root MLflow | Create and `chown` the mount dir in the image before `USER` |
| API calls to `http://mlflow:5000` rejected (MLflow ≥ 3.5) | `--allowed-hosts=mlflow:5000,localhost:5000` |
| `libgomp.so.1: cannot open shared object` | `apt-get install --no-install-recommends libgomp1` in runtime |
| `ports: "5000:5000"` | `127.0.0.1:5000:5000` |

## Red Flags — stop and fix

- `COPY models/` or `*.joblib`/`*.pkl` in a Dockerfile
- No `USER` line, or `pip install` without `uv.lock`
- MLflow server without `--backend-store-uri` (falls back to the `./mlruns` file store)
- A size claim without a `docker images` number

REQUIRED SUB-SKILL: scaffolding-ds-projects (uv, `uv.lock`, dependency groups, `Makefile` targets such as `make docker-build`).
