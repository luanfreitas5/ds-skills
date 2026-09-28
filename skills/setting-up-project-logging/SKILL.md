---
name: setting-up-project-logging
description: Use when configuring Python logging for a project, when code tracks execution with print(), when log lines appear twice or not at all, when a rich progress bar is garbled by log output, when logging.basicConfig or addHandler appears in a non-entrypoint module, or when someone asks to log DataFrame rows "to debug".
---

# Setting Up Project Logging

## Overview

**Core principle:** logging is configured **once**, in the entrypoint, from `configs/logging.yaml` via `logging.config.dictConfig`. Handlers live only on the root logger. Every other module just does `logger = logging.getLogger(__name__)`. Logs carry metadata (counts, shapes, metrics), never data rows.

## Rules

| Rule | How |
|---|---|
| One configuration point | `configure_logging()` called first thing in `main()` and in each `if __name__ == "__main__":` block — never at import time |
| Modules never configure | No `basicConfig`, `addHandler`, `setLevel` or `StreamHandler` outside `src/config/logging.py` |
| Config is standard dictConfig YAML | Not a home-made schema parsed by hand; `disable_existing_loggers: false` is **mandatory** |
| Handlers only on root | `loggers:` in YAML sets levels of noisy libraries only; a handler on a child logger + root = duplicated lines |
| Daily file | `TimedRotatingFileHandler(when="midnight")`, `encoding="utf-8"`; not `FileHandler(f"log_{date.today()}")`, which never rotates in a run that crosses midnight |
| Progress bars share the console | One `rich.Console` passed to both `RichHandler` and `Progress(console=...)`; log inside the loop with `logger`, never `print()` |
| Level from CLI | `--log-level` overrides `config["root"]["level"]` before `dictConfig` |
| Lazy formatting | `logger.info("AUC: %.4f", auc)`, not f-strings |
| No rows in logs | Log `df.shape`, `df.schema`, `df.null_count()`, rates. Never `df`, `df.head()`, `df.sample()`, `describe()` of raw data. **REQUIRED SUB-SKILL: protecting-pii-lgpd** |

## Implementation

`configs/logging.yaml`: copy [logging_template.yaml](logging_template.yaml). Its `"()"` factories point to the module below, so `config` must be importable. Install the project with `uv pip install -e .` and run modules as `python -m data.loader`, not `python -m src.data.loader`. Mixing the two import styles creates two loggers with different names (`data.loader` and `src.data.loader`), and `caplog`/level filters then miss one of them.

```python
"""src/config/logging.py — configuração única de logging: console rich + arquivo diário."""

import logging
import logging.config
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any

import yaml
from rich.console import Console
from rich.logging import RichHandler

ROOT = Path(__file__).resolve().parents[2]
CONSOLE = Console(stderr=True)  # compartilhado por RichHandler e rich.progress


def create_console_handler() -> RichHandler:
    """Cria o handler de console ligado ao console compartilhado."""
    return RichHandler(console=CONSOLE, rich_tracebacks=True, show_path=True, markup=False)


def _rename_rotated(default_name: str) -> str:
    """Converte ``logs/log.log.2026-09-27`` em ``logs/log_2026-09-27.log``."""
    path = Path(default_name)
    stem, date_str = path.name.split(".log.")
    return str(path.with_name(f"{stem}_{date_str}.log"))


def create_daily_file_handler(filename: str, backup_count: int = 30) -> TimedRotatingFileHandler:
    """Cria o handler de arquivo com rotação diária à meia-noite.

    Parameters
    ----------
    filename : str
        Caminho do arquivo ativo, relativo à raiz do projeto.
    backup_count : int, optional
        Quantidade de arquivos diários mantidos, by default 30.

    Returns
    -------
    TimedRotatingFileHandler
        Handler que rotaciona para ``log_YYYY-MM-DD.log``.
    """
    path = ROOT / filename
    path.parent.mkdir(parents=True, exist_ok=True)  # dictConfig falha se o diretório não existe
    handler = TimedRotatingFileHandler(path, when="midnight", backupCount=backup_count, encoding="utf-8")
    handler.namer = _rename_rotated
    return handler


def configure_logging(level: str | None = None, config_path: Path = ROOT / "configs" / "logging.yaml") -> None:
    """Aplica a configuração de logging do YAML. Chamar apenas no ponto de entrada.

    Parameters
    ----------
    level : str | None, optional
        Sobrescreve o nível do root (ex.: ``--log-level DEBUG``), by default None.
    config_path : Path, optional
        YAML no formato dictConfig, by default ``configs/logging.yaml``.

    Raises
    ------
    FileNotFoundError
        Se o YAML não existir.
    ValueError
        Se o YAML ou o nível forem inválidos.

    Examples
    --------
    >>> configure_logging(level="DEBUG")
    """
    config: dict[str, Any] = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if level is not None:
        config["root"]["level"] = level.upper()
    logging.config.dictConfig(config)  # substitui os handlers do root: chamar de novo não duplica
    logging.captureWarnings(True)  # warnings.warn() passa pelos mesmos handlers
```

In the modules, the loop and the progress bar share the same console:

```python
logger = logging.getLogger(__name__)

progress = Progress(SpinnerColumn(), TextColumn("[bold blue]{task.description}"), BarColumn(),
                    MofNCompleteColumn(), TaskProgressColumn(), TimeElapsedColumn(),
                    TimeRemainingColumn(), console=CONSOLE)
with progress:
    task = progress.add_task("Treinando folds", total=n_folds)
    for fold in range(1, n_folds + 1):
        ...
        logger.info("Fold %d/%d — AUC: %.4f", fold, n_folds, auc)  # sai acima da barra
        progress.advance(task)
```

Verified behavior on Python 3.11–3.14:
- A logger created at import time, before `dictConfig`, is **silently disabled** when `disable_existing_loggers` is left at its default (`true`).
- Calling `configure_logging()` three times leaves exactly 2 root handlers.
- Rotated files are named `log_YYYY-MM-DD.log`, and `backupCount` still prunes them.
- `warnings.warn` output reaches the file log.
- A test that calls `configure_logging()` (or `main()`) removes pytest's handler, so `caplog.text` comes back empty. Tests should call the functions directly and never configure logging.

Why the bar gets garbled: a `StreamHandler` keeps the original `sys.stderr` it got at creation, so it writes underneath rich's live display. Only a handler that writes through the **same** `Console` as the `Progress` gets printed above the bar.

## Common Mistakes

| Mistake | Fix |
|---|---|
| `basicConfig` in `loader.py` "so it works standalone" | `configure_logging()` inside `if __name__ == "__main__":` |
| `logger.addHandler(RichHandler())` in a module | Remove it; root already has the handler (otherwise lines print twice) |
| Loader logs vanished after switching to dictConfig | `disable_existing_loggers: false` |
| Hand-rolled YAML plus a `_configured` global and `root.handlers.clear()` | Standard dictConfig YAML; `dictConfig` already replaces root handlers |
| Analyst asks for "the first rows in the log"; you drop `cpf`/`email` and log `head()` | A denylist leaks the next PII column (`telefone`). Log metadata only; inspect rows interactively, outside logs |
| `track()` or a separate `Console()` next to a `RichHandler` | `Progress(console=CONSOLE)` with the shared console |
| Package named `src/logging/` with `src/` on the path | It shadows the stdlib `logging`; put the module in `config/logging.py` |
| `print()` left over for progress | `logger.info` for events, `Progress` for per-iteration progress |
