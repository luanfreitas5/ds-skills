---
name: configuring-typed-settings
description: Use when creating or changing YAML configs, argparse flags, .env files, or config loaders, when adding a new hyperparameter or API key, when values are hardcoded in a training script, or when a long run could fail late on a bad config value.
---

# Configuring Typed Settings

## Overview

`yaml.safe_load` into a dict, or a pydantic model with the default `extra="ignore"`, accepts `min_child_weigth: 5` silently: the typo is dropped, the default is used, and nobody notices after six hours of training. **Core principle:** one typed settings object, built from every source in a documented precedence, that rejects unknown keys and invalid values **at startup** — before any data is loaded.

## Rules

| Rule | How |
|---|---|
| Unknown keys fail | `extra="forbid"` on the settings class **and on every nested `BaseModel`** (`model_config = ConfigDict(extra="forbid")`) |
| Values have bounds | `Field(gt=0, le=1)`, `Literal[...]`, `FilePath` / `DirectoryPath` for inputs that must exist |
| One precedence, written down | **CLI > environment variables > `.env` > YAML > class defaults**, via `settings_customise_sources` |
| Secrets never in YAML or code | `SecretStr`, read from env / `.env`; `.env` in `.gitignore`, `.env.example` committed |
| Fail at startup | Build settings as the first line of `main()`; no config reads later in the run |
| CLI overrides only what was passed | argparse defaults `None`; drop `None` before passing to the settings class |
| Log the resolved config | `settings.model_dump(mode="json")` to log/MLflow — `SecretStr` prints `**********` |

## Implementation

```python
import argparse
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, FilePath, SecretStr
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)


class ModelParams(BaseModel):
    """Hiperparâmetros do modelo; chaves desconhecidas são rejeitadas."""

    model_config = ConfigDict(extra="forbid")

    n_estimators: int = Field(gt=0)
    max_depth: int = Field(gt=0, le=32)
    learning_rate: float = Field(gt=0, le=1)
    min_child_weight: float = Field(ge=0)


class TrainSettings(BaseSettings):
    """Configuração do treino. Precedência: CLI > variáveis de ambiente > .env > YAML."""

    model_config = SettingsConfigDict(
        yaml_file=Path("configs/train.yaml"),
        env_file=".env",
        env_nested_delimiter="__",  # MODEL__LEARNING_RATE=0.1
        extra="forbid",
    )

    data_path: FilePath  # falha na inicialização se o arquivo não existir
    seed: int = 42
    model: ModelParams
    wandb_api_key: SecretStr  # lida de WANDB_API_KEY; nunca no YAML

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Define a ordem de precedência: a primeira fonte vence."""
        return (init_settings, env_settings, dotenv_settings, YamlConfigSettingsSource(settings_cls))


def parse_cli_overrides(argv: list[str] | None = None) -> dict[str, Any]:
    """Converte apenas os argumentos informados na CLI em overrides aninhados."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=Path)
    parser.add_argument("--n-estimators", type=int)
    parser.add_argument("--learning-rate", type=float)
    args = {k: v for k, v in vars(parser.parse_args(argv)).items() if v is not None}
    model = {k: args.pop(k) for k in ("n_estimators", "learning_rate") if k in args}
    return {**args, "model": model} if model else args


def load_settings(argv: list[str] | None = None) -> TrainSettings:
    """Carrega e valida toda a configuração antes de qualquer trabalho pesado."""
    return TrainSettings(**parse_cli_overrides(argv))
```

Verified behavior: `--n-estimators 500` overrides only that key (sources are deep-merged); `min_child_weigth` in YAML → `ValidationError` (`extra_forbidden`) at startup; missing `WANDB_API_KEY` → `missing`; nonexistent `--data-path` → `path_not_file`; `repr(settings)` shows `SecretStr('**********')`.

With `extra="forbid"`, **every** variable in `.env` must be a settings field (`OTHER_VAR=1` → `extra_forbidden`). Keep `.env` project-specific, or set `env_prefix`.

Log the resolved config and seed with the run (REQUIRED SUB-SKILL: ensuring-reproducibility). Keys and tokens follow the secret rules (REQUIRED SUB-SKILL: protecting-pii-lgpd).

## Common Mistakes

| Mistake | Fix |
|---|---|
| `yaml.safe_load` → dict → `cfg["lr"]` deep in the code | Typed settings object, validated once |
| `extra="forbid"` only on the top class | Also on nested `BaseModel`s — they default to `ignore` |
| argparse defaults duplicating YAML values | Defaults `None`; YAML/class defaults are the single source |
| Secret as `str` with a custom `__repr__` | `SecretStr`; call `.get_secret_value()` only where the key is used |
| Checking the data path when loading data, hours later | `FilePath` in settings |
| Precedence undocumented | Docstring + README state CLI > env > `.env` > YAML |
