---
name: validating-data-contracts
description: Use when writing any function that reads, cleans, transforms, or writes a dataset between pipeline stages (raw → interim → processed), when serving a model that receives external input, or when told the data "is clean", "was checked in Excel", or that validation is "over-engineering".
---

# Validating Data Contracts

## Overview

Code tests check logic; data contracts check the data. **Core principle:** every stage boundary validates its input and output against an explicit, version-controlled `pandera` schema, and every row dropped or changed is counted and logged.

"I eyeballed it in Excel" is not validation — Excel reformats dates, hides whitespace, and shows the first screen only. A schema is ~15 lines; debugging a silently corrupted training set costs days.

## The Contract Pattern

For every stage function:

1. **Validate input** against the upstream schema (fail fast, `lazy=True` to report all errors at once).
2. **Transform**, logging counts of rows removed per rule — never silent `dropna()`.
3. **Validate output** against this stage's schema before writing.
4. **Write** to the next stage directory; never overwrite `data/raw/`.

Schemas live in `src/schemas/` — one per stage. Ranges and category sets come from the data dictionary or the user; when unknown, derive them from a profile of the raw data (`df.describe()`, `unique()`), state them as assumptions in your reply, and mark them `# TODO: confirmar domínio`. The minimal-code request still gets the schema; drop logging config, argparse, or MLflow before dropping the contract.

## What a Schema Must Check

| Check | Example |
|---|---|
| Types | `amount: float`, `timestamp: datetime` |
| Ranges | `amount >= 0` |
| Nullability | IDs and target never null |
| Uniqueness | `transaction_id` unique |
| Categories | `region` in accepted set |
| Label domain | `is_fraud` in `{0, 1}` |
| Cross-column | `end_date >= start_date` |
| Strictness | `strict=True` — unexpected columns fail |

## Implementation

```python
import logging
from pathlib import Path

import pandera.polars as pa
import polars as pl

logger = logging.getLogger(__name__)


class TransactionsProcessedSchema(pa.DataFrameModel):
    """Contrato de dados para transações prontas para modelagem."""

    transaction_id: str = pa.Field(unique=True, nullable=False)
    customer_id: str = pa.Field(nullable=False)
    amount: float = pa.Field(ge=0, le=1_000_000)
    region: str = pa.Field(isin=["norte", "nordeste", "centro-oeste", "sudeste", "sul"])
    timestamp: pl.Datetime = pa.Field(nullable=False)
    is_fraud: int = pa.Field(isin=[0, 1])

    class Config:
        strict = True


def clean_transactions(raw_path: Path, processed_path: Path) -> pl.DataFrame:
    """Limpa transações brutas e grava o resultado validado em parquet."""
    df = pl.read_csv(raw_path, try_parse_dates=True)
    n_raw = df.height

    df = df.unique(subset="transaction_id", keep="first")
    logger.info(f"Duplicatas removidas: {n_raw - df.height}")

    n_before = df.height
    df = df.drop_nulls(subset=["transaction_id", "customer_id", "amount", "is_fraud"])
    logger.info(f"Linhas removidas por nulos em colunas-chave: {n_before - df.height}")

    df = df.with_columns(pl.col("region").str.strip_chars().str.to_lowercase())

    validated = TransactionsProcessedSchema.validate(df, lazy=True)
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    validated.write_parquet(processed_path)
    logger.info(f"{validated.height}/{n_raw} linhas válidas gravadas em {processed_path}")
    return validated
```

Serving: validate each request against the *training* schema to catch train–serve skew; reject with a clear error instead of predicting on garbage.

## Common Mistakes

| Mistake | Fix |
|---|---|
| Skipping schema to "keep it short" | Schema is the short part; cut other ceremony first |
| `dropna()` without logging | Log removed-row count per rule; alert if above a threshold |
| Schema only on output | Validate input too — catch upstream changes at the boundary |
| `strict=False` by default | Strict; new columns are a contract change, not noise |
| Validating only a `head()` sample | Validate the full frame |
