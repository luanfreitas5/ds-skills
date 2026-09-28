---
name: ingesting-external-data
description: Use when writing code that downloads data from a REST API, database, website, or public data portal into data/raw, when a download script runs on a schedule (cron, Airflow), or when told the downloader is "quick", "just a script", or that it can simply overwrite the previous file.
---

# Ingesting External Data

## Overview

`data/raw/` is the only copy of what the source said *at that moment*. Sources change, delete rows, and go offline. **Core principle:** every download is a new, immutable, dated snapshot of the bytes as received, with a manifest (hashes, license, request) — and the client is polite and resilient: timeout, retry with backoff, rate limit, bounded pagination.

A weekly job that writes `data/raw/ocorrencias.parquet` destroys last week's data every week. Atomic replace (`.tmp` + rename) protects against *partial* writes, not against *overwriting* — the old snapshot is still gone, and last month's model can no longer be reproduced.

## Rules

| Rule | How | Missing it causes |
|---|---|---|
| Raw is immutable | `raw/<source>/<collected_at UTC>/`; `FileExistsError` if it exists | Training data changes silently between runs |
| Collection date in the path | `%Y-%m-%dT%H%M%SZ` from `datetime.now(timezone.utc)` | "Which week's data trained this model?" |
| Store bytes as received | Write `response.content` (JSON/CSV/HTML) — convert to parquet in `interim/` | Parse/type-inference bugs baked into "raw" |
| Complete or nothing | Write to `<dir>.partial/`, rename only after the last page + manifest | Half-downloaded snapshot looks valid |
| Timeout on every request | `httpx.Timeout(30, connect=10)` | Cron job hangs forever |
| Retry transient only | 429, 5xx, transport errors; exponential backoff + jitter; honor `Retry-After`; fail fast on other 4xx | Hammering a down API, or retrying a 401 forever |
| Rate limit proactively | Min interval between requests (read the API's documented quota) | 429 storms, IP ban |
| Bounded pagination | `max_pages` guard; stop on `next_page is None` | Infinite loop on a buggy cursor |
| Manifest | URL, params, `collected_at`, SHA-256 per file, counts, client version | No proof of what was downloaded |
| License / terms | `license` + `terms_url` in manifest; if unknown, record `"UNKNOWN"` + log a warning — never omit the field; check robots.txt/ToS before scraping | Unpublishable dataset |
| Zero records = failure | Raise; the `.partial/` dir stays for debugging and is ignored downstream | Empty file feeds training |

Databases: same snapshot rule — dump the query result with the SQL text and `collected_at` in the manifest; use a read-only user and paginate with keyset (`WHERE id > :last ORDER BY id LIMIT n`), not `OFFSET`.

## Implementation

Tested against `httpx.MockTransport` (429 with `Retry-After` in seconds and as HTTP date, 503, read timeout, 404, 3-page pagination, re-run on an existing snapshot).

```python
import hashlib
import json
import logging
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def fetch_with_retry(
    client: httpx.Client, url: str, params: dict, max_attempts: int = 5, base_delay: float = 1.0
) -> httpx.Response:
    """Faz GET com backoff exponencial + jitter; respeita Retry-After em 429/503.

    Raises
    ------
    httpx.HTTPStatusError
        Em erro 4xx não transitório ou após esgotar as tentativas.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            response = client.get(url, params=params)
            if response.status_code not in RETRYABLE_STATUS:
                response.raise_for_status()  # 4xx definitivo: falha imediata
                return response
            reason = f"HTTP {response.status_code}"
            retry_after = response.headers.get("Retry-After")
        except httpx.TransportError as exc:  # timeout, conexão recusada/resetada
            reason, retry_after = type(exc).__name__, None
        if attempt == max_attempts:
            logger.error(f"Desistindo após {attempt} tentativas ({reason}): {url} {params}")
            raise httpx.HTTPError(f"Falha persistente ({reason}) em {url} {params}")
        delay = base_delay * 2 ** (attempt - 1)
        if retry_after and retry_after.isdigit():  # Retry-After também pode vir como data HTTP
            delay = float(retry_after)
        delay += random.uniform(0, base_delay)  # jitter (não é uso criptográfico)
        logger.warning(f"{reason}; tentativa {attempt}/{max_attempts}, aguardando {delay:.1f}s")
        time.sleep(delay)
    raise AssertionError("inalcançável")


def compute_sha256(path: Path) -> str:
    """Calcula o SHA-256 de um arquivo em blocos."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def download_snapshot(
    url: str, raw_root: Path, source: str, license_id: str, terms_url: str,
    per_page: int = 500, min_interval: float = 1.0, max_pages: int = 10_000,
) -> Path:
    """Baixa todas as páginas para um snapshot imutável em raw/<fonte>/<data de coleta>/.

    Grava as respostas exatamente como recebidas + manifest.json (hashes, licença).
    O diretório só ganha o nome final se o download inteiro terminar.

    Raises
    ------
    FileExistsError
        Se o snapshot já existir — raw nunca é sobrescrito.
    """
    collected_at = datetime.now(timezone.utc)
    final_dir = raw_root / source / collected_at.strftime("%Y-%m-%dT%H%M%SZ")
    tmp_dir = final_dir.with_name(final_dir.name + ".partial")
    if final_dir.exists():
        raise FileExistsError(f"Snapshot já existe, raw é imutável: {final_dir}")
    tmp_dir.mkdir(parents=True, exist_ok=False)

    files, n_records, page, last_call = [], 0, 1, 0.0
    timeout = httpx.Timeout(30.0, connect=10.0)
    with httpx.Client(timeout=timeout, headers={"User-Agent": "ds-pipeline (contato@exemplo.org)"}) as client:
        while page is not None:
            if len(files) >= max_pages:
                raise RuntimeError(f"Mais de {max_pages} páginas — paginação em loop?")
            time.sleep(max(0.0, min_interval - (time.monotonic() - last_call)))  # limite de taxa
            last_call = time.monotonic()
            response = fetch_with_retry(client, url, {"page": page, "per_page": per_page})
            payload = response.json()
            page_path = tmp_dir / f"page_{page:05d}.json"
            page_path.write_bytes(response.content)  # bytes originais, sem reparse
            files.append(page_path)
            n_records += len(payload["data"])
            logger.info(f"Página {page}: {len(payload['data'])} registros (total {n_records})")
            page = payload.get("next_page")

    if n_records == 0:
        raise RuntimeError(f"API retornou 0 registros — snapshot incompleto mantido em {tmp_dir}")
    manifest = {
        "source": source, "url": url, "params": {"per_page": per_page},
        "collected_at": collected_at.isoformat(), "license": license_id, "terms_url": terms_url,
        "n_pages": len(files), "n_records": n_records, "httpx": httpx.__version__,
        "files": {p.name: compute_sha256(p) for p in files},
    }
    (tmp_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_dir.rename(final_dir)  # atômico: snapshot parcial nunca parece completo
    logger.info(f"Snapshot gravado em {final_dir} ({n_records} registros)")
    return final_dir
```

Downstream: read the *latest complete* snapshot (`sorted(raw_root.glob(f"{source}/*Z"))[-1]` — skips `.partial`), verify hashes against its manifest, then parse + validate into `interim/`. Record the snapshot path and manifest hash in the run manifest (REQUIRED SUB-SKILL: ensuring-reproducibility). Validate the parsed data at the raw → interim boundary (REQUIRED SUB-SKILL: validating-data-contracts).

Credentials (API keys, DB passwords) come from `.env` via `pydantic-settings` — never in the URL stored in the manifest or in logs.

## Rationalizations

| Excuse | Reality |
|---|---|
| "Deadline is today, skip hash/manifest" | Manifest is ~10 lines, included above. Skipping it is the expensive option. |
| "Atomic replace keeps last week's data safe" | It keeps *a* file intact; it still deletes the previous snapshot. |
| "`urllib3.Retry` on 429 is enough rate limiting" | That is reactive. Space requests proactively; retry is the backup. |
| "It's public data, license doesn't matter" | Public ≠ open license. Record it; papers and portfolios need it. |
| "Saving parquet in raw is more convenient" | Convert in `interim/`; raw keeps the source's bytes so parsing bugs are fixable. |
| "Storage is expensive, keep only the latest" | Compress or prune by policy — never by overwriting. Use DVC for large snapshots. |

## Common Mistakes

| Mistake | Fix |
|---|---|
| `requests.get(url)` without `timeout` | Always pass a timeout; `requests` has none by default |
| Retrying every exception | Retry only transient; 400/401/403/404 fail immediately |
| Fixed `time.sleep(5)` retries | Exponential backoff + jitter, capped attempts |
| Local time in snapshot names | UTC with `Z` suffix |
| API key in logged URL/params | Send in headers; never log it |
| Scraping without checking ToS/robots.txt | Check first; record in manifest `terms_url` |
