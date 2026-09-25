---
name: serving-models-fastapi
description: Use when exposing a trained model as an HTTP API with FastAPI (or Flask), when preprocessing was saved as separate scaler/encoder/model files, when the API must accept JSON from a front end, or when predictions in production differ from predictions in the notebook.
---

# Serving Models with FastAPI

## Overview

Serving fails in three quiet ways: the model is loaded per request (slow), the input is an untyped dict (garbage in, 500 out), and preprocessing is re-implemented by hand in the API (train–serve skew: a different median, a different column order, a new category). **Core principle:** serve **one** artifact — the fitted sklearn `Pipeline` that did the preprocessing in training — loaded once at startup, behind a request schema derived from the training data contract.

## Rules

| Rule | How |
|---|---|
| One artifact | Train and save `Pipeline([("prep", ColumnTransformer(...)), ("model", ...)])`. Separate `scaler.joblib` / `encoder.joblib` / `xgb.joblib` → go back to training and persist the pipeline; do not rebuild the steps in the API |
| Load once | `lifespan` loads the model into `app.state`; startup fails if the artifact is missing |
| Schema from the contract | Request model mirrors the training schema (REQUIRED SUB-SKILL: validating-data-contracts): same names, dtypes, bounds, `Literal` categories, `extra="forbid"` |
| Missing values | Optional field → `None` → pipeline's fitted imputer handles it. No constants typed into the API |
| Version endpoint | `/version` returns model name, registry version/alias, training git SHA, data hash (from the model's metadata) |
| Health endpoint | `/health` = process up **and** model loaded |
| Tests | `TestClient` inside `with` (runs lifespan): valid payload, invalid category → 422, extra field → 422, parity test: API output == `pipeline.predict_proba` on the same rows |
| Monitoring | Log inputs (no PII) and scores for drift checks (REQUIRED SUB-SKILL: monitoring-model-drift) |

## If Training Saved Separate Artifacts

The deliverable has three parts, in this order:

1. **Training change** — code that fits and saves `{"pipeline": Pipeline(...), "metadata": {...}}` (imputer + scaler + encoder + model in one `Pipeline`), replacing the three `joblib.dump` calls.
2. **API** that loads only that bundle (below).
3. **Tests**: `TestClient` payload/422 tests and the parity test.

Do not ship an API that stacks `scaler.transform` + `encoder.transform` + `np.hstack`: column order and the missing imputer are exactly the skew this skill prevents. "The user already saved them separately" means step 1 is needed, not that step 1 is skipped.

## Implementation

```python
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field

MODEL_PATH = Path("models/churn_pipeline.joblib")


class Customer(BaseModel):
    """Entrada do /predict, espelhando o contrato de dados do treino."""

    model_config = ConfigDict(extra="forbid")

    renda: float | None = Field(default=None, ge=0)  # None -> imputador ajustado no treino
    idade: int = Field(ge=18, le=110)
    meses_cliente: int = Field(ge=0)
    gasto_mes: float = Field(ge=0)
    plano: Literal["basico", "pro", "premium"]


class Prediction(BaseModel):
    """Saída do /predict."""

    churn_probability: float
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Carrega o pipeline completo uma única vez na inicialização."""
    bundle = joblib.load(MODEL_PATH)  # {"pipeline": Pipeline, "metadata": {...}}; artefato próprio, confiável
    app.state.pipeline = bundle["pipeline"]
    app.state.metadata = bundle["metadata"]
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer, request: Request) -> Prediction:
    """Aplica o mesmo pipeline do treino (pré-processamento + modelo)."""
    frame = pd.DataFrame([customer.model_dump()])
    proba = float(request.app.state.pipeline.predict_proba(frame)[0, 1])
    return Prediction(churn_probability=proba, model_version=request.app.state.metadata["version"])


@app.get("/health")
def health(request: Request) -> dict[str, bool]:
    """Processo ativo e modelo carregado."""
    return {"ok": hasattr(request.app.state, "pipeline")}


@app.get("/version")
def version(request: Request) -> dict[str, str]:
    """Versão do modelo, commit e hash dos dados de treino."""
    return request.app.state.metadata
```

`metadata` = `{"version": ..., "git_sha": ..., "data_hash": ...}` saved with the pipeline at training time (or read from the MLflow registry alias). Sync `def` endpoints run in FastAPI's thread pool, so CPU-bound `predict_proba` does not block the event loop.

```python
from fastapi.testclient import TestClient


def test_predict_matches_pipeline(sample_rows: pd.DataFrame) -> None:
    """Paridade: a API devolve o mesmo score que o pipeline salvo."""
    with TestClient(app) as client:  # `with` executa o lifespan
        body = sample_rows.iloc[0].to_dict()
        got = client.post("/predict", json=body).json()["churn_probability"]
        expected = app.state.pipeline.predict_proba(sample_rows.iloc[[0]])[0, 1]
        assert abs(got - expected) < 1e-9
        assert client.post("/predict", json={**body, "plano": "gold"}).status_code == 422
```

## Common Mistakes

| Mistake | Fix |
|---|---|
| `joblib.load` inside the endpoint | `lifespan` + `app.state` |
| Scaler, encoder, model loaded separately; `np.hstack` in the API | One fitted `Pipeline` from training |
| `renda_median_fallback: 3500` in serving config | Imputer inside the pipeline |
| `dict` / `Any` as request body | Pydantic model from the data contract, `extra="forbid"` |
| `TestClient(app)` without `with` | Lifespan never runs; model not loaded |
| No way to know which model answered | `/version` + `model_version` in each response |
| "Tests are out of scope" | `TestClient` tests are part of the API deliverable |

## Red Flags — you are rebuilding training in the API

- A helper named `_transform` / `preprocess` in `app/`
- `np.hstack` or a hardcoded feature order in serving code
- A required field that was optional in training "because the imputer was not saved"
