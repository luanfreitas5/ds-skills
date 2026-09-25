---
name: ensuring-reproducibility
description: Use when writing training or experiment scripts whose results will be published, reviewed, compared, or deployed, when reviewers or teammates cannot reproduce numbers, or when someone claims "random_state=42 everywhere" makes it reproducible.
---

# Ensuring Reproducibility

## Overview

`random_state=42` fixes one source of randomness out of many. **Core principle:** a result is reproducible only when you can recover the exact **code + data + environment + config + seeds** that produced it. Save that lineage next to every model as a run manifest.

## The Five Pins

| Pin | How | Missing it causes |
|---|---|---|
| Code | Git SHA; refuse to train (or tag loudly) on a dirty tree | "Which version of features.py?" |
| Data | SHA-256 of every input file (chunked for big files); DVC for storage | Silently re-exported parquet |
| Environment | Committed `uv.lock`; log Python + key library versions | sklearn default changes between versions |
| Config | Resolved config (YAML + CLI overrides) saved as JSON | Unknown hyperparameters |
| Seeds | `PYTHONHASHSEED`, `random`, `numpy`, framework; `shuffle=True` splitters with `random_state` | Different folds each run |

## Implementation

```python
import hashlib
import json
import logging
import os
import platform
import random
import subprocess
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    """Fixa todas as fontes de aleatoriedade conhecidas."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def compute_file_hash(path: Path, chunk_size: int = 1 << 20) -> str:
    """Calcula o SHA-256 de um arquivo em blocos (suporta arquivos grandes)."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def get_git_sha() -> str:
    """Retorna o SHA do commit atual; marca '-dirty' se houver alterações não commitadas."""
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()  # noqa: S603, S607
    dirty = subprocess.run(["git", "diff", "--quiet"], check=False).returncode != 0  # noqa: S603, S607
    if dirty:
        logger.warning("Árvore Git com alterações não commitadas — resultado não reprodutível")
    return f"{sha}-dirty" if dirty else sha


def write_run_manifest(
    output_dir: Path, data_paths: list[Path], config: dict, metrics: dict, seed: int
) -> Path:
    """Grava manifest.json com toda a linhagem necessária para reproduzir o modelo."""
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": get_git_sha(),
        "data_sha256": {str(p): compute_file_hash(p) for p in data_paths},
        "python": platform.python_version(),
        "packages": {pkg: version(pkg) for pkg in ["numpy", "polars", "scikit-learn"]},
        "seed": seed,
        "config": config,
        "metrics": metrics,
    }
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
```

With MLflow: log the same fields as tags (`git_sha`, `data_sha256`) and params, and register the model — the manifest file stays as the portable fallback.

## Paper Checklist

- Reproducibility section states: Git tag/SHA, data hash (or DOI), lock file, seeds, hardware, CV protocol
- `make reproduce` (or `dvc repro`) regenerates every table and figure from raw data
- Report variance across seeds (e.g. 5 seeds) for stochastic models, not one lucky seed

## Common Mistakes

| Mistake | Fix |
|---|---|
| `StratifiedKFold(n_splits=5)` without `shuffle=True, random_state` | Explicit shuffle + seed, or document that no shuffle is intended |
| `>=` version ranges without lock file | Commit `uv.lock` |
| Hash logged only to console | Persist in manifest + MLflow tags |
| GPU training assumed deterministic | `torch.use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG`; document residual nondeterminism |
| Model saved without its preprocessing | Persist the whole `Pipeline` |
| `1.96 * std / sqrt(5)` as CV CI in the manifest | Corrected CI (REQUIRED SUB-SKILL: evaluating-models-rigorously) |
