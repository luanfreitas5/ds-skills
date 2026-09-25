---
name: versioning-data-with-dvc
description: Use when datasets or trained models need versioning, when someone is about to git add a .parquet, .csv, .joblib, .pkl, or .onnx file, when GitHub rejects files over 100 MB, when Git LFS is proposed for data, or when nobody can tell which data produced which model.
---

# Versioning Data with DVC

## Overview

Git LFS stores big blobs but knows nothing about lineage: it cannot tell that `xgb.joblib` came from `train.parquet` + `learning_rate=0.05` + commit `a1b2c3`. **Core principle:** Git versions code, params, and small pointer files; DVC versions the data and model content and records the pipeline DAG in `dvc.lock`, so `git checkout <tag> && dvc pull && dvc repro` rebuilds the exact result.

## Quick Reference

| Goal | Command |
|---|---|
| Start | `dvc init` → commit `.dvc/`, `.dvcignore` |
| Remote (S3/GCS/Azure/SSH/local) | `dvc remote add -d storage s3://bucket/project` → commit `.dvc/config` |
| Track a **source** file (raw, external) | `dvc add data/raw/vendas.csv` → commit `vendas.csv.dvc` |
| Track a **generated** file | Declare it as a stage `out` in `dvc.yaml` — never `dvc add` it |
| Run pipeline (only stale stages) | `dvc repro` → commit `dvc.lock` |
| Share content | `dvc push` **before** `git push` |
| Get content | `dvc pull` (or `dvc pull <target>` for one file) |
| What changed | `dvc status`, `dvc params diff`, `dvc metrics diff main` |
| Experiments | `dvc exp run -S configs/model_params.yaml:learning_rate=0.1 -n lr01`, `dvc exp show --only-changed`, `dvc exp apply lr01` |
| Batch of experiments | `dvc exp run --queue -S configs/model_params.yaml:learning_rate=0.01` (repeat) → `dvc queue start` |

## Pipeline (`dvc.yaml`)

```yaml
stages:
  prepare:
    cmd: python src/data/prepare.py
    deps: [src/data/prepare.py, data/raw/vendas.csv]
    outs: [data/processed/train.parquet, data/processed/test.parquet]
  train:
    cmd: python src/models/train.py
    deps: [src/models/train.py, data/processed/train.parquet]
    params:
      - configs/model_params.yaml:
          - seed
          - learning_rate
          - n_estimators
    outs: [models/xgb.joblib]
  evaluate:
    cmd: python src/models/evaluate.py
    deps: [src/models/evaluate.py, models/xgb.joblib, data/processed/test.parquet]
    metrics:
      - reports/metrics.json:
          cache: false
```

- Every file a stage reads is a `dep`; every hyperparameter it reads is in `params` (a param change must re-run the stage).
- `evaluate` depends on the **test** split. If the described pipeline evaluates on the training file, change it: `prepare` also writes `test.parquet` and `evaluate` depends on it. Do not encode a train-set metric in the DAG "to stay faithful".
- `metrics` with `cache: false` stay in Git so `dvc metrics diff` works without pulling.

## Linking to the Manifest

`dvc.lock` stores the `md5` of every dep and out. Use it as the `data_hash` in the run manifest and MLflow tags (REQUIRED SUB-SKILL: ensuring-reproducibility), so one hash identifies the data in DVC, the manifest, and the tracker:

```python
from pathlib import Path

import yaml


def read_dvc_out_md5(stage: str, out_path: str, lock_path: Path = Path("dvc.lock")) -> str:
    """Lê o md5 de uma saída de estágio registrada no dvc.lock."""
    lock = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    for out in lock["stages"][stage]["outs"]:
        if out["path"] == out_path:
            return out["md5"]
    raise KeyError(f"Saída {out_path!r} não encontrada no estágio {stage!r} do dvc.lock")
```

## Common Mistakes

| Mistake | Fix |
|---|---|
| `git add train.parquet` / Git LFS for data | Stage `out` + `dvc push`; Git holds only `dvc.lock` |
| `dvc add` on a file a stage produces | Error "overlaps with an output of stage"; keep it only in `dvc.yaml` |
| `.gitignore` with `data/*` | Hides new `.dvc` pointer files (`data/*` + `!data/**/*.dvc` still hides them: git cannot re-include inside an excluded dir). Let DVC write per-folder `.gitignore` entries, or use `data/**` + `!data/**/` + `!data/**/*.dvc` |
| `git push` without `dvc push` | Collaborators get pointers to content that is not in the remote |
| Param read by script but not listed in `params` | `dvc repro` skips the stage after a change; list every param |
| Running 3 variants by editing YAML and overwriting the model | `dvc exp run -S <file>:<key>=<v>`; compare with `dvc exp show`; `dvc exp apply` the winner |
| `-S learning_rate=0.1` with params outside `params.yaml` | Error "Key 'learning_rate' is not in struct"; prefix the file: `-S configs/model_params.yaml:learning_rate=0.1` |
| Committing code without `dvc.lock` | Commit `dvc.lock` in the same commit as the code change that produced it |
