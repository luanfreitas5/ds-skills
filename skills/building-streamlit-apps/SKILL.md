---
name: building-streamlit-apps
description: Use when creating or changing a Streamlit dashboard or app, when every widget interaction is slow, when an app reloads data or models on each click, when the app trains models, or when API keys or personal data appear in app code.
---

# Building Streamlit Apps

## Overview

Streamlit reruns the whole script on every widget change. Anything expensive that is not cached — reading a 1.5 GB parquet, loading a model, training — runs again on each click. **Core principle:** the app reads prepared, PII-free data and loads a trained model, both cached with the right decorator; interactions are batched in forms; training and secrets live outside the script.

## Rules

| Rule | How |
|---|---|
| Data → `st.cache_data` | Returns a copy per call; for DataFrames, query results. Set `ttl` or `max_entries` when args vary (each filter combo is an entry) |
| Models, connections, clients → `st.cache_resource` | One shared object, not copied; never mutate it per user |
| Invalidation | Pass what changes as an argument (file `mtime`, model version); `func.clear()` after a data refresh; leading-underscore args (`_conn`) are **not** hashed |
| Batch inputs | `with st.form("filtros"):` + `st.form_submit_button` — one rerun per submit, not per widget |
| Keep state | `st.session_state` for results that must survive reruns (last prediction, selected customer) |
| No training in the app | Training runs as a pipeline job (REQUIRED SUB-SKILL: tracking-experiments-mlflow); the app loads the registered model. A "retrain" button, if required, only **triggers** the job |
| Secrets | `st.secrets["openai_api_key"]` from `.streamlit/secrets.toml` (git-ignored) or the host's secret manager; never a literal |
| No PII in the app | Read a pre-built aggregated/pseudonymized dataset without direct identifiers; mask free text before sending it to an LLM (REQUIRED SUB-SKILL: protecting-pii-lgpd) |
| Structure | `app/Home.py` + `app/pages/1_Churn.py`, `2_Assistente.py`; logic in `src/`, pages only call it |
| Tests | `streamlit.testing.v1.AppTest`: run the page, set widgets, submit, assert on output and no exceptions |

## Implementation

```python
# app/pages/1_Churn.py
from pathlib import Path

import polars as pl
import streamlit as st

DATA = Path("data/processed/churn_dashboard.parquet")  # sem nome/CPF/e-mail


@st.cache_data(ttl="1h", max_entries=64)
def load_churn_by_region(regions: tuple[str, ...], plans: tuple[str, ...], mtime: float) -> pl.DataFrame:
    """Lê só as colunas necessárias e agrega; mtime invalida o cache quando o arquivo muda."""
    return (
        pl.scan_parquet(DATA)
        .filter(pl.col("regiao").is_in(regions) & pl.col("plano").is_in(plans))
        .group_by("regiao")
        .agg(pl.col("churn").mean().alias("taxa_churn"), pl.len().alias("n_clientes"))
        .sort("regiao")
        .collect()
    )


st.title("Churn por região")
with st.form("filtros"):
    regions = st.multiselect("Região", ["norte", "nordeste", "sul"], default=["norte", "nordeste", "sul"])
    plans = st.multiselect("Plano", ["basico", "pro"], default=["basico", "pro"])
    submitted = st.form_submit_button("Aplicar")

if submitted or "summary" not in st.session_state:
    st.session_state["summary"] = load_churn_by_region(tuple(sorted(regions)), tuple(sorted(plans)), DATA.stat().st_mtime)

st.bar_chart(st.session_state["summary"], x="regiao", y="taxa_churn")
```

```python
# tests/test_app.py
from streamlit.testing.v1 import AppTest


def test_churn_page_filters_without_errors() -> None:
    """A página roda, aceita filtros pelo formulário e não gera exceções."""
    at = AppTest.from_file("../app/pages/1_Churn.py").run()  # relativo a este arquivo de teste
    at.multiselect[0].set_value(["sul"])
    at.button[0].click().run()
    assert not at.exception
    assert at.session_state["summary"]["regiao"].to_list() == ["sul"]
```

## Common Mistakes

| Mistake | Fix |
|---|---|
| `pd.read_parquet` at top level | `st.cache_data` function with projection + filter |
| Model in `st.cache_data` | `st.cache_resource` (no copy, no pickling per call) |
| Unbounded `cache_data` keyed by filters | `ttl` / `max_entries` |
| Each multiselect triggers a rerun | `st.form` |
| `fit()` behind a button in the app | Pipeline job + registry; app loads `@champion` |
| `OPENAI_API_KEY = "sk-..."` or custom `.env` loader in the app | `st.secrets` |
| CPF/e-mail columns loaded "but not shown" | Build a PII-free dataset for the app |
| Only manual clicking as a test | `AppTest` in `pytest` |
| `AppTest.from_file("app/...")` from `tests/` → `FileNotFoundError` | Path is relative to the test file: `"../app/..."` |
