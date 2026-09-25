---
name: protecting-pii-lgpd
description: Use when code reads, logs, samples, exports, plots, or commits data containing personal information (CPF, RG, name, email, phone, address, CEP, birth date, health, biometric), when asked to "log sample rows" or "export examples" of people, or when working under Brazilian LGPD or similar privacy law.
---

# Protecting PII under LGPD

## Overview

**Core principle:** personal data enters the pipeline once, is minimized and pseudonymized at the boundary, and never appears in logs, reports, figures, notebooks outputs, or commits. Analysts get pseudonymous IDs; re-identification happens only in the system of record (CRM) under its own access control.

## Boundary Rules

1. **Classify columns at ingestion** in a single constant: direct identifiers (CPF, name, email, phone), quasi-identifiers (birth date, CEP, city, sex), sensitive data (LGPD art. 5 II: health, race, religion, biometrics).
2. **Minimize:** drop direct identifiers not needed; generalize quasi-identifiers (birth date → age band, CEP → 3-digit prefix).
3. **Pseudonymize** needed IDs with keyed HMAC-SHA256; key comes from `.env` via `pydantic-settings`, never a CLI argument (visible in shell history and process list) or code.
4. **Logs:** counts, shapes, dtypes, null rates, and pseudonymous IDs only. Never `df.head()` of raw data.
5. **Exports/reports:** aggregates or pseudonymized rows; check k-anonymity on quasi-identifiers (every combination appears ≥ k=5 times) before sharing row-level samples.
6. **Git:** `data/` ignored, `detect-secrets` in pre-commit, `nbstripout` for notebooks.
7. **Document** legal basis (art. 7 / art. 11), retention, and deletion path in the datasheet.

## Implementation

```python
import hashlib
import hmac

import polars as pl
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

DIRECT_IDENTIFIERS = ["cpf", "full_name", "email", "phone"]
QUASI_IDENTIFIERS = ["age_band", "city"]


class PrivacySettings(BaseSettings):
    """Segredos de privacidade carregados do .env (nunca commitado)."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")
    pseudonymization_key: SecretStr


def pseudonymize_value(value: str, key: bytes) -> str:
    """Pseudonimiza um identificador com HMAC-SHA256 (irreversível sem a chave)."""
    return hmac.new(key, value.encode("utf-8"), hashlib.sha256).hexdigest()[:20]


def anonymize_customers(df: pl.DataFrame, key: bytes) -> pl.DataFrame:
    """Aplica minimização, generalização e pseudonimização na fronteira de ingestão."""
    return (
        df.with_columns(
            pl.col("cpf").map_elements(lambda v: pseudonymize_value(v, key), return_dtype=pl.String)
            .alias("customer_key"),
            ((pl.lit(2026) - pl.col("birth_date").dt.year()) // 10 * 10).alias("age_band"),
        )
        .drop([*DIRECT_IDENTIFIERS, "birth_date"])
    )


def validate_k_anonymity(df: pl.DataFrame, k: int = 5) -> None:
    """Levanta erro se alguma combinação de quase-identificadores aparece menos de k vezes."""
    smallest = df.group_by(QUASI_IDENTIFIERS).len()["len"].min()
    if smallest is not None and smallest < k:
        raise ValueError(f"Exportação viola k-anonimato: grupo com {smallest} < {k} registros")
```

Replace the hardcoded year with the reference date of the snapshot.

## Common Mistakes

| Mistake | Fix |
|---|---|
| `logger.info(df.head())` "to debug" | Log `df.schema`, `df.null_count()`, row counts |
| Plain SHA-256 of CPF | CPF space is small (~10⁹); brute-forceable. Use keyed HMAC |
| Salt passed as CLI flag | `.env` + `SecretStr` |
| Export "50 example customers" with name/email for marketing | Export `customer_key`; marketing resolves in CRM |
| Exact birth date / full CEP kept as feature | Generalize to band / prefix |
| PII in notebook outputs committed | `nbstripout` in pre-commit |
