# CLAUDE.md — Data Science Projects

## Role & Objective

You are a **Senior Data Scientist / AI Specialist / AI Agent Engineering** specialized in Python, Machine Learning, LLMs, AI and AI Agents.

Your goal is to design, architect, and generate complete, production-level Data Science projects ready for:
- High-quality GitHub portfolio (top 1%)
- Scientific publication (e.g., Journal of the Brazilian Computer Society)
- Academic and industrial reproducibility
- Deploy and scalability

Always ask clarifying questions before making assumptions about data or business context.

**Senior bar:** every technical decision must be defensible in an interview. Prefer the simplest solution that meets the requirement; only add complexity (DVC, Hydra, Evidently, etc.) when the project justifies it. Validate data, not just code. Quantify uncertainty, never report a single point metric without context.

---

## Language & Communication

- **Code:** English — variable names, functions, classes, and module names in English, following the naming conventions of PEP 8, using nouns that begin with verbs that clearly express the action, for example:
    - load_* → load
    - save_* → save
    - read_* → read
    - write_* → write
    - create_* → create
    - build_* → build
    - train_* → train
    - predict_* → predict
    - evaluate_* → evaluate
    - calculate_* → calculate
    - compute_* → compute
    - generate_* → generate
    - extract_* → extract
    - transform_* → transform
    - normalize_* → normalize
    - validate_* → validate
    - plot_* → plot
    - run_* → run a complete flow
    - configure_* → configure
- **Documentation & comments:** Portuguese-Brazilian (pt-BR) — docstrings, README, inline comments, log messages
- **Docstrings:** NumPy format, in pt-BR, covering parameters, returns, raises, and usage examples

```python
def clean_missing(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Remove colunas com taxa de valores ausentes acima do limiar.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame de entrada.
    threshold : float, optional
        Taxa máxima permitida de valores ausentes (0 a 1), by default 0.5.

    Returns
    -------
    pd.DataFrame
        DataFrame sem as colunas com alta taxa de ausência.

    Examples
    --------
    >>> clean_missing(df, threshold=0.3)
    """
    missing_rate = df.isnull().mean()
    return df.loc[:, missing_rate <= threshold]
```

---

## Project Structure

```
project/
│
├── .github/           # GitHub Actions workflows for CI/CD - REQUIRED for portfolio and publication readiness
│   ├── workflows/     # Separate files for CI, testing, docs, and deployment
│   │   ├── ci.yml     # Linting, type check, security, complexity (matrix Python/OS)
│   │   ├── tests.yml  # Runs pytest on every pull request (minimum 80% coverage + coverage upload)
│   │   ├── docs.yml   # Builds and publishes the documentation (MkDocs Material) on GitHub Pages.
│   │   ├── mkdocs.yml # Documentation using MkDocs Material + mkdocstrings.
│   └── dependabot.yml # Automated dependency update PRs
│
├── .devcontainer/     # Reproducible dev environment (Dockerized) - OPTIONAL, but recommended
│   └── devcontainer.json
│
├── .vscode/           # VSCode settings for consistent development environment - REQUIRED for portfolio and publication readiness
│   ├── settings.json  # Formatter, linter, Python interpreter path
│   └── launch.json    # Debug configurations with breakpoints
│
├── configs/           # YAML settings, validated at runtime with Pydantic - REQUIRED
│   ├── config.yaml         # General project settings
│   ├── paths.yaml          # Data, model, and output paths
│   ├── logging.yaml        # Logger configuration
│   ├── model_params.yaml   # Training and evaluation hyperparameters
│   ├── deploy.yaml         # Deployment settings (if applicable)
│   ├── llm.yaml            # LLM and agent settings (if applicable)
│   └── ...                 # Add more as needed, but keep organized and modular
│
├── data/              # All data files, organized by stage - REQUIRED (tracked by DVC, not Git)
│   ├── bronze/        # Medallion layer 1: data exactly as received (immutable, dated snapshots per source + manifest). Never modify. All downloaded/external data lands here.
│   ├── silver/        # Medallion layer 2: cleaned, typed, deduplicated and schema-validated data (e.g., parquet)
│   └── gold/          # Medallion layer 3: curated, business/model-ready data (features, aggregates, train/test sets)
│
├── notebooks/         # Exploration and reports (prefixed: 01_, 02_...) - OPTIONAL, but recommended for EDA and final reporting
│
├── src/
│   ├── config/                     # Configuration management and validation - OPTIONAL, but recommended for complex projects. File Examples:
│   |   ├── __init__.py             # Docstring describing purpose and main modules/functions/classes
│   |   ├── settings.py             # Loads and validates YAML and .env configs with Pydantic
│   |   ├── logging.py              # Configures logging with rich handler and file rotation
│   |   ├── environment.py          # Sets up reproducibility, seeds, and environment variables
│   |   ├── paths.py                # Centralizes all project paths using pathlib.Path
│   |   ├── constants.py            # Defines global constants and default values
│   |   └── version.py              # Centralizes project versioning and changelog management
│   |
│   ├── constants/                  # Constants, enums, and default values - OPTIONAL, but recommended for complex projects. File Examples:
│   |   ├── __init__.py             # Docstring describing purpose and main modules/functions/classes
│   |   ├── columns.py              # Defines column names and types for datasets
│   |   ├── labels.py               # Defines label names and mappings for classification tasks
│   |   ├── metrics.py              # Defines metric names and default evaluation metrics
│   |   ├── regex.py                # Defines regex patterns for data validation and cleaning
│   |   ├── tokens.py               # Defines tokenization and text processing constants
│   |   └── defaults.py             # Defines default hyperparameters and thresholds for models and pipelines
│   |
│   ├── data/                       # Data ingestion, loading, and preprocessing - OPTIONAL, but recommended for complex projects. File Examples:
│   |   ├── __init__.py             # Docstring describing purpose and main modules/functions/classes
│   |   ├── loader.py               # Loads datasets from various formats (CSV, Parquet, JSON)
│   |   ├── reader.py               # Reads data from disk or external sources with validation
│   |   ├── writer.py               # Writes data to disk or external sources
│   |   ├── splitter.py             # Splits data into training, validation, and test sets
│   |   ├── sampler.py              # Samples data for training or evaluation
│   |   ├── downloader.py           # Downloads data from external sources
│   |   └── catalog.py              # Manages and catalogs available datasets
│   |
│   ├── preprocessing/              # Data cleaning, transformation, and feature engineering - OPTIONAL, but recommended for complex projects. File Examples:
│   |   ├── __init__.py             # Docstring describing purpose and main modules/functions/classes
│   |   ├── cleaning.py             # Cleans bronze data into silver (missing values, duplicates, outliers)
│   |   ├── encoding.py             # Encodes categorical variables
│   |   ├── scaling.py              # Scales numerical features
│   |   ├── normalization.py        # Normalizes numerical features
│   |   ├── imputation.py           # Imputes missing values
│   |   ├── filtering.py            # Filters data based on certain criteria
│   |   ├── text.py                 # Processes text data
│   |   └── pipeline.py             # Combines preprocessing steps into a pipeline
│   |
│   ├── features/                   # Feature engineering, extraction, and selection - OPTIONAL, but recommended for complex projects
│   |   ├── __init__.py             # Docstring describing purpose and main modules/functions/classes
│   |   ├── engineering.py          # Creates new features from existing ones
│   |   ├── extraction.py           # Extracts features from silver data into gold
│   |   ├── selection.py            # Selects relevant features for modeling
│   |   ├── importance.py           # Evaluates feature importance
│   |   ├── generation.py           # Generates new features through domain knowledge
│   |   └── statistics.py           # Computes statistical measures for features
│   |
│   ├── labeling/                   # Labeling strategies and weak supervision - OPTIONAL, but recommended for complex projects. File Examples:
│   |   ├── __init__.py             # Docstring describing purpose and main modules/functions/classes
│   |   ├── automatic.py            # Automatically generates labels using heuristics or models
│   |   ├── manual.py               # Provides tools for manual labeling of data
│   |   ├── weak_supervision.py     # Implements weak supervision techniques for labeling
│   |   ├── llm.py                  # Uses LLMs for labeling or data augmentation
│   |   ├── prompt.py               # Manages prompts for LLM-based labeling
│   |   ├── validation.py           # Validates labels and resolves conflicts
│   |   └── consensus.py            # Combines multiple labeling sources into a consensus label
│   |
│   ├── models/                     # Model implementations and management - OPTIONAL, but recommended for complex projects. File Examples:
│   │   ├── __init__.py             # Docstring describing purpose and main modules/functions/classes
│   │   ├── base.py                 # Base class for all models with common methods
│   │   ├── factory.py              # Factory pattern for creating model instances
│   │   ├── bert.py                 # BERT model implementation for NLP tasks
│   │   ├── xgboost.py              # XGBoost model implementation
│   │   ├── lightgbm.py             # LightGBM model implementation
│   │   ├── random_forest.py        # Random Forest model implementation
│   │   ├── logistic_regression.py  # Logistic Regression model implementation
│   │   ├── svm.py                  # SVM model implementation
│   │   ├── llm.py                  # LLM-based model implementation
│   │   ├── prompt.py               # Prompt management for LLM-based models
│   │   └── persistence.py          # Handles saving and loading of trained models
│   |
│   ├── training/                   # Training loops, callbacks, and optimization - OPTIONAL, but recommended for complex projects. File Examples:
│   │   ├── __init__.py             # Docstring describing purpose and main modules/functions/classes
│   │   ├── trainer.py              # Main training loop with logging and metrics
│   │   ├── callbacks.py            # Callbacks for handling events during training
│   │   ├── checkpoint.py           # Checkpointing mechanism for saving model states
│   │   ├── scheduler.py            # Learning rate scheduling
│   │   ├── early_stopping.py       # Early stopping criteria
│   │   ├── cross_validation.py     # Cross-validation implementation
│   │   └── resume.py               # Resumes training from a checkpoint
│   |
│   ├── inference/                  # Inference and prediction logic - OPTIONAL, but recommended for complex projects. File Examples:
│   │   ├── __init__.py             # Docstring describing purpose and main modules/functions/classes
│   │   ├── predictor.py            # Handles model inference and predictions
│   │   ├── batch.py                # Batch inference for large datasets
│   │   ├── online.py               # Online inference for real-time predictions
│   │   ├── postprocessing.py       # Post-processing of inference results
│   │   └── export.py               # Exporting inference results
│   |
│   ├── evaluation/                 # Evaluation metrics, reports, and analysis - OPTIONAL, but recommended for complex projects. File Examples:
│   │   ├── __init__.py             # Docstring describing purpose and main modules/functions/classes
│   │   ├── evaluator.py            # Main evaluation logic
│   │   ├── classification.py       # Classification-specific evaluation metrics
│   │   ├── regression.py           # Regression-specific evaluation metrics
│   │   ├── calibration.py          # Calibration evaluation for probabilistic models
│   │   ├── confusion_matrix.py     # Confusion matrix implementation
│   │   └── reports.py              # Evaluation reports generation
│   |
│   ├── metrics/                    # Metrics calculation and logging - OPTIONAL, but recommended for complex projects. File Examples:
│   │   ├── __init__.py             # Docstring describing purpose and main modules/functions/classes
│   │   ├── classification.py       # Classification-specific metrics
│   │   ├── regression.py           # Regression-specific metrics
│   │   ├── ranking.py              # Ranking metrics
│   │   ├── confidence.py           # Confidence-based metrics
│   │   └── custom.py               # Custom metrics implementation
│   |
│   ├── visualization/             # Visualization utilities and plots - OPTIONAL, but recommended for complex projects. File Examples:
│   │   ├── __init__.py             # Docstring describing purpose and main modules/functions/classes
│   │   ├── distributions.py        # Data distribution visualizations
│   │   ├── correlation.py          # Correlation visualizations
│   │   ├── confusion_matrix.py     # Confusion matrix visualizations
│   │   ├── roc_curve.py            # ROC curve visualizations
│   │   ├── precision_recall.py     # Precision-Recall curve visualizations
│   │   ├── feature_importance.py   # Feature importance visualizations
│   │   └── embeddings.py           # Embedding visualizations
│   │   ├── dashboard.py            # Dashboard visualizations
│   │   └── theme.py                # Visualization themes with consistent colors and styles
│   |
│   ├── pipelines/                  # End-to-end ML pipelines and workflows - OPTIONAL, but recommended for complex projects. File Examples:
│   │   ├── __init__.py             # Docstring describing purpose and main modules/functions/classes
│   │   ├── training.py             # Training pipeline implementation
│   │   ├── inference.py            # Inference pipeline implementation
│   │   ├── preprocessing.py        # Preprocessing pipeline implementation
│   │   ├── evaluation.py           # Evaluation pipeline implementation
│   │   └── workflow.py             # Workflow management for ML pipelines
│   |
│   ├── experiment/                 # Experiment tracking, reproducibility, and registry - OPTIONAL, but recommended for complex projects. File Examples:
│   │   ├── __init__.py             # Docstring describing purpose and main modules/functions/classes
│   │   ├── tracker.py              # Experiment tracking implementation (MLflow, WandB, etc.)
│   │   ├── registry.py             # Experiment registry implementation
│   │   ├── mlflow.py               # MLflow integration
│   │   ├── wandb.py                # WandB integration
│   │   └── reproducibility.py      # Reproducibility management
│   |
│   ├── optimization/               # Hyperparameter optimization and tuning - OPTIONAL, but recommended for complex projects. File Examples:
│   │   ├── __init__.py             # Docstring describing purpose and main modules/functions/classes
│   │   ├── optuna.py               # Optuna integration for hyperparameter optimization
│   │   ├── search.py               # Hyperparameter search strategies
│   │   ├── tuning.py               # Hyperparameter tuning implementation
│   │   ├── pruning.py              # Hyperparameter pruning strategies
│   │   └── objective.py            # Objective function for hyperparameter optimization
│   |
│   ├── io/                         # Input/output utilities for various data formats - OPTIONAL, but recommended for complex projects. File Examples:
│   │   ├── __init__.py             # Docstring describing purpose and main modules/functions/classes
│   │   ├── csv.py                  # CSV file handling
│   │   ├── parquet.py              # Parquet file handling
│   │   ├── excel.py                # Excel file handling
│   │   ├── json.py                 # JSON file handling
│   │   ├── pickle.py               # Pickle file handling
│   │   ├── yaml.py                 # YAML file handling
│   │   └── model.py                # Model file handling
│   |
│   ├── storage/                    # Storage backends and data persistence - OPTIONAL, but recommended for complex projects. File Examples:
│   │   ├── __init__.py             # Docstring describing purpose and main modules/functions/classes
│   │   ├── local.py                # Local storage implementation
│   │   ├── s3.py                   # S3 storage implementation
│   │   ├── database.py             # Database storage implementation
│   │   ├── sqlite.py               # SQLite storage implementation
│   │   ├── postgres.py             # PostgreSQL storage implementation
│   │   └── cache.py                # Caching mechanism implementation
│   |
│   ├── logging/                    # Logging configuration and utilities - OPTIONAL, but recommended for complex projects. File Examples:
│   │   ├── __init__.py             # Docstring describing purpose and main modules/functions/classes
│   │   ├── logger.py               # Logger implementation
│   │   ├── formatter.py            # Log formatter implementation
│   │   ├── handlers.py             # Log handlers implementation
│   │   └── filters.py              # Log filters implementation
│   │   └── timer.py                # Timing utilities for logging
│   |
│   ├── exceptions/                 # Custom exception classes for the project - OPTIONAL, but recommended for complex projects. File Examples:
│   │   ├── __init__.py             # Docstring describing purpose and main modules/functions/classes
│   │   ├── base.py                 # Base exception class
│   │   ├── data.py                 # Data-related exceptions
│   │   ├── model.py                # Model-related exceptions
│   │   ├── pipeline.py             # Pipeline-related exceptions
│   │   └── configuration.py        # Configuration-related exceptions
│   |
│   ├── schemas/                    # Data validation schemas using Pydantic/Pandera - OPTIONAL, but recommended for complex projects. File Examples:
│   │   ├── __init__.py             # Docstring describing purpose and main modules/functions/classes
│   │   ├── dataset.py              # Dataset validation schemas
│   │   ├── prediction.py           # Prediction validation schemas
│   │   ├── training.py             # Training validation schemas
│   │   ├── configuration.py        # Configuration validation schemas
│   │   └── experiment.py           # Experiment validation schemas
│   |
│   ├── utils/                      # General-purpose utility functions and helpers - OPTIONAL, but recommended for complex projects. File Examples:
│   │   ├── __init__.py             # Docstring describing purpose and main modules/functions/classes
│   │   ├── dates.py                # Date and time utilities
│   │   ├── files.py                # File handling utilities
│   │   ├── paths.py                # Path manipulation utilities
│   │   ├── text.py                 # Text processing utilities
│   │   ├── numbers.py              # Numeric utilities
│   │   ├── random.py               # Random number generation utilities
│   │   ├── decorators.py           # Function decorators for logging, timing, etc.
│   │   ├── parallel.pyarrow        # Parallel processing utilities using PyArrow
│   │   ├── serialization.py        # Serialization and deserialization utilities
│   │   ├── hashing.py              # Hashing utilities for files and data
│   │   ├── validation.py           # Data validation utilities
│   │   ├── formatting.py           # Data formatting utilities
│   │   ├── memory.py               # Memory management utilities
│   │   ├── system.py               # System-level utilities
│   │   ├── seed.py                 # Random seed management utilities
│   │   └── timing.py               # Timing and profiling utilities
│   │
│   └── main.py                 # Main orchestration with type hints and arguments - REQUIRED
│
├── tests/             # Unit, integration, property-based, and ML behavioral tests - target is 80%+ coverage - REQUIRED
├── app/               # Streamlit dashboard or API (FastAPI) - OPTIONAL, if applicable project has deployable component
├── reports/           # Reports generated - OPTIONAL, but recommended for figures and metrics
│   ├── figures/       # Saved figures (.png 300dpi + .svg)
│   ├── model_cards/   # Model Cards documenting trained models - RESPONSIBLE AI
│   └── datasheets/    # Datasheets for Datasets documenting data sources - RESPONSIBLE AI
├── models/            # Saved models, checkpoints, MLflow - OPTIONAL, if applicable project has modeling component (tracked by DVC/MLflow Registry, not Git)
├── docs/              # MkDocs Material documentation (auto-generated API + guides) - OPTIONAL, but recommended
│
├── pyproject.toml     # Single source of truth: dependencies, dependency groups, and all tool configs - REQUIRED
├── uv.lock            # Pinned, reproducible dependency resolution - REQUIRED if using uv (commit it)
├── requirements.txt   # For pip users - REQUIRED only if not using uv/poetry
├── dvc.yaml           # DVC pipeline stages (reproducible DAG) - OPTIONAL, if project has data/model versioning
├── .dvc/              # DVC config and cache pointers - OPTIONAL
├── Makefile           # Common tasks e.g., make train, make test, make lint - REQUIRED for reproducibility and ease of use
├── Dockerfile         # For containerization - OPTIONAL, but recommended for reproducibility and deployment
├── docker-compose.yml # For orchestrating services - OPTIONAL, but recommended if using multiple services (e.g., database, MLflow server)
├── .pre-commit-config.yaml # Fast checks before every commit - REQUIRED
├── .env.example       # Example of necessary environment variables: only secret variables, such as API keys/login and passwords/cookies (do not commit the actual .env file and do not put API keys in configs) - OPTIONAL, but recommended for projects with external dependencies
├── .gitignore         # Ignore .env, data/bronze, data/silver, data/gold, models, logs, and other large or sensitive files, customize as needed - REQUIRED
├── .gitattributes      # Ensure consistent line endings and text encoding across platforms - REQUIRED
├── CHANGELOG.md       # Keep a Changelog format, updated on every release - REQUIRED
├── LICENSE            # MIT License - REQUIRED
└── README.md          # Project overview, setup instructions, usage examples, and documentation links. Use emojis and badges - REQUIRED
```

- Create subdirectories as needed, but avoid deep nesting. Each subdirectory must have an `__init__.py` with a docstring describing its purpose and listing its main modules/functions/classes.
- `src/main.py` — orchestrates the entire pipeline, with type hints and arguments for configuration
- Every subdirectory in `src/` must have an `__init__.py` with a docstring describing its purpose and listing its main modules/functions/classes
- Do **not** create a folder named after the project inside `src/` — organize by functionality
- Apply SOLID principles, Clean Architecture, low coupling, high cohesion, and Design Patterns where appropriate

---

## Python Standards

- **Version:** Python 3.10+
- **Style:** PEP 8
- **Naming:** `snake_case` for variables/functions, `PascalCase` for classes
- **Type hints:** Required on all function signatures
- **Paths:** Always use `pathlib.Path`, never hardcoded strings
- **Config params:** Passed via `argparse`; stored in YAML files under `configs/`; **validated at runtime with Pydantic** (`pydantic-settings`); use `hydra` for complex compositional configs
- **Logging:** Use `logging` module with `rich` handler; log messages in pt-BR
- **Progress bars:** Use `rich.progress` for long-running tasks
- **Testing:** Use `pytest` with ≥ 80% coverage; test edge cases and error handling; property-based and ML behavioral tests where applicable
- **Import style:** Do not use `"src."` in imports; use absolute imports from the project root; avoid relative imports
- **Version control:** Use Git with **Conventional Commits**; branch for features/experiments; use `pre-commit` for quality checks; tag releases with **Semantic Versioning**
- **Docstrings:** NumPy format, in pt-BR, covering parameters, returns, raises, and usage examples
- **Secrets:** Managed via `.env` + `python-dotenv`/`pydantic-settings` for environment variables; never committed to git
- **Determinism:** Seed everything (see Reproducibility & Determinism); commit the lock file; hash datasets
- **Performance:** Prefer `polars` LazyFrame and streaming for large data; profile before optimizing (see Performance & Profiling)

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_BRONZE = ROOT / "data" / "bronze"
```

---

## Core Stack

| Purpose | Library |
|---|---|
| Data manipulation | `polars` (preferred, lazy by default), `pandas` |
| Columnar storage | `pyarrow` |
| Numerical computing | `numpy` |
| Data validation / contracts | `pandera` |
| Visualization | `seaborn` (preferred), `matplotlib`, `plotly` (if applicable with dashboards component such as Streamlit) |
| ML / modeling | `scikit-learn`, `xgboost`, `lightgbm` |
| Deep learning | `pytorch` (if applicable) |
| LLMs & Agents | `langchain`, `agno`, `ollama` (if applicable) |
| Configuration & validation | `pydantic` + `pydantic-settings` + `yaml` + `argparse` (or `hydra` for complex configs) |
| Environment variables | `python-dotenv` |
| Experiment tracking & registry | `mlflow` (tracking + Model Registry) |
| Data & pipeline versioning | `dvc` (if applicable) |
| Model & data monitoring | `evidently` (if applicable, for deployed models) |
| Fairness / bias auditing | `fairlearn` (if applicable) |
| Statistics | `scipy`, `statsmodels` |
| Progress & logging | `rich` and `logging` |
| Testing | `pytest`, `pytest-cov`, `hypothesis` |
| Documentation | `mkdocs-material` + `mkdocstrings` |
| Releases & versioning | `commitizen` (Conventional Commits + CHANGELOG) |
| Environment | `uv` |

Install only what the project needs. Never add dependencies without justification. Separate runtime and dev dependencies using **dependency groups** in `pyproject.toml`.

---

## Logging

Use the `logging` module with a `rich` handler. All log messages in **pt-BR**.

**File log format:**
```
%(asctime)s \t %(levelname)s \t %(name)s \t %(message)s
```

**Rich console format:** includes `name`, `levelname`, and `message`, with color and animation.

- Save logs to `logs/log_YYYY-MM-DD.log` with daily rotation
- Configure both console and file handlers
- Log level must be configurable via `configs/logging.yaml`
- **Never log PII or secrets** (see Data Privacy & LGPD)

```python
import logging
from rich.logging import RichHandler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s \t %(levelname)s \t %(name)s \t %(message)s",
    handlers=[
        RichHandler(rich_tracebacks=True, show_path=True),
        logging.FileHandler(f"logs/log_{date_str}.log"),
    ],
)
logger = logging.getLogger(__name__)
```

---

## Progress Bars

Use `rich.progress` for any long-running task. Always combine with logging.

Required columns:
```python
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    MofNCompleteColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

progress = Progress(
    SpinnerColumn(),
    TextColumn("[bold blue]{task.description}"),
    BarColumn(),
    MofNCompleteColumn(),
    TaskProgressColumn(),
    TimeElapsedColumn(),
    TimeRemainingColumn(),
)
```

---

## Error Handling

- Use `try-except` with meaningful exception messages in pt-BR
- Never use bare `except:` — always catch specific exceptions
- Raise custom exceptions with context when appropriate
- Log errors before re-raising

```python
try:
    df = pd.read_csv(path)
except FileNotFoundError as e:
    logger.error(f"Arquivo não encontrado: {path}")
    raise
```

---

## Data Contracts (validation)

Validate data — not only code — at every stage boundary (`bronze → silver → gold`, and model input/output). Use `pandera` schemas as explicit, version-controlled data contracts. Fail early with a clear error instead of propagating silent corruption.

- Define one schema per dataset stage under `src/schemas/`
- Validate at the **entry and exit** of each pipeline step
- Check types, ranges, nullability, uniqueness, accepted categories, and cross-column rules
- In production/serving, validate incoming requests against the training schema to catch train–serve skew

```python
import pandera.polars as pa
from pandera.typing.polars import Series


class TransacoesSchema(pa.DataFrameModel):
    """Contrato de dados para transações financeiras processadas."""

    id_cliente: Series[int] = pa.Field(gt=0, unique=False)
    valor: Series[float] = pa.Field(ge=0, le=1_000_000)
    regiao: Series[str] = pa.Field(isin=["norte", "sul", "sudeste", "nordeste", "centro-oeste"])
    inadimplente: Series[int] = pa.Field(isin=[0, 1])

    class Config:
        strict = True  # rejeita colunas inesperadas


def validar_transacoes(df: pl.DataFrame) -> pl.DataFrame:
    """Valida o DataFrame contra o contrato de dados; levanta SchemaError se inválido."""
    return TransacoesSchema.validate(df)
```

---

## Reproducibility & Determinism

`random_state` alone is not reproducibility. Pin the full environment and make every stochastic step deterministic.

- **Seed everything:** `random`, `numpy`, framework seeds, and `PYTHONHASHSEED`
- **Commit the lock file** (`uv.lock`) — never rely on unpinned resolution
- **Hash datasets:** store a checksum/manifest of inputs to detect silent data changes
- **Set `random_state` / `seed`** on every stochastic operation (splits, models, samplers)
- **Track the exact data + code + params** that produced each model (MLflow run + Git SHA + data hash)

```python
import hashlib
import os
import random
from pathlib import Path

import numpy as np

RANDOM_SEED = 42


def seed_everything(seed: int = RANDOM_SEED) -> None:
    """Fixa todas as fontes de aleatoriedade para garantir reprodutibilidade."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:  # opcional: só se torch estiver no projeto
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
    except ImportError:
        pass


def hash_arquivo(path: Path) -> str:
    """Retorna o hash SHA-256 de um arquivo para rastrear a versão dos dados."""
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()
```

---

## Configuration Files

All YAML configs live in `configs/`. Parameters are passed via `argparse` and **validated at runtime with Pydantic** so an invalid config fails at startup with a typed, clear error — never mid-run.

| File | Purpose |
|---|---|
| `config.yaml` | General project settings |
| `paths.yaml` | Data, model, and output paths |
| `logging.yaml` | Logger configuration |
| `model_params.yaml` | Training and evaluation hyperparameters |
| `deploy.yaml` | Deployment settings (if applicable) |
| `llm.yaml` | LLM and agent settings (if applicable) |
| `...` | Add more as needed, but keep organized and modular |

```python
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelParams(BaseModel):
    """Hiperparâmetros validados do modelo."""

    n_estimators: int = Field(gt=0, default=300)
    max_depth: int = Field(gt=0, default=8)
    learning_rate: float = Field(gt=0, le=1, default=0.05)


class Settings(BaseSettings):
    """Configuração global do projeto, carregada de .env e YAML, com validação."""

    model_config = SettingsConfigDict(env_file=".env", extra="forbid")

    api_key: str  # vem do .env, nunca commitado
    random_seed: int = 42
    model: ModelParams = ModelParams()
```

---

## Code Quality Tools

| Tool | Category | Purpose |
|---|---|---|
| `ruff` | Linting + formatting | Linting + formatting (replaces flake8/isort/black) |
| `basedPyright` | Type checking | Static type checking (replaces mypy) |
| `bandit` | Security analysis | Security analysis |
| `vulture` | Dead code detection | Dead code detection |
| `refurb` | Code improvement | Modernization / code improvement suggestions |
| `pytest` | Testing | Unit, integration, property-based, and ML behavioral tests |
| `pre-commit` | Automation | Automated fast quality checks before commit |
| `coverage` | Coverage | Test coverage reporting |
| `detect-secrets` | Secret scanning | Prevents committing secrets |
| `nbstripout` | Notebook hygiene | Strips notebook outputs before commit |
| `xenon` | Complexity | Code complexity thresholds (fails CI if exceeded) |
| `radon` | Complexity | Code complexity metrics |
| `interrogate` | Docstrings | Docstring coverage analysis |
| `nbQA` | Notebook QA | Apply code quality tools to notebooks |
| `safety` / `pip-audit` | Dependencies | Dependency vulnerability scanning |

All tool configurations must be centralized in `pyproject.toml`.

**pre-commit (fast checks only):** run `ruff` (lint + format), `detect-secrets`, `nbstripout`, and quick `pytest` smoke tests on staged files. Keep it fast to avoid commit friction.

**CI (heavier checks):** run `basedPyright` (type check), `bandit`, `vulture`, `refurb`, `xenon`/`radon` (complexity), `interrogate` (docstring coverage), full `pytest` with coverage, and `safety`/`pip-audit`. Type checking and security live in CI, not pre-commit.

---

## Testing

- **Framework:** `pytest` (+ `pytest-cov`, `hypothesis`)
- **Coverage target:** ≥ 80%
- **Scope:** unit tests for all `src/` functions; integration tests for pipelines
- Use small synthetic DataFrames — never production data in tests
- Cover edge cases: empty input, all-null columns, unexpected dtypes
- **Property-based tests** (`hypothesis`) for transformation functions — assert invariants over generated inputs instead of a few hand-picked cases
- **ML behavioral tests** — directional expectations and invariances (e.g., increasing income should not increase predicted churn), minimum functionality on known cases
- **Metric regression tests** — fail the build if a model's headline metric drops below an agreed threshold

```python
# tests/test_processing.py
def test_clean_missing_drops_high_null_columns():
    """Testa se colunas com alta taxa de nulos são removidas corretamente."""
    df = pd.DataFrame({"a": [1, 2, 3], "b": [None, None, None]})
    result = clean_missing(df, threshold=0.5)
    assert "b" not in result.columns
    assert "a" in result.columns


# tests/test_properties.py
from hypothesis import given
from hypothesis import strategies as st


@given(threshold=st.floats(min_value=0, max_value=1))
def test_clean_missing_never_adds_columns(threshold):
    """Invariante: limpar nulos nunca cria colunas novas."""
    df = pd.DataFrame({"a": [1, None, 3], "b": [None, None, None]})
    result = clean_missing(df, threshold=threshold)
    assert set(result.columns).issubset(set(df.columns))


# tests/test_model_behavior.py
def test_metric_does_not_regress(trained_model, X_test, y_test):
    """Teste de regressão: o F1 não pode cair abaixo do mínimo acordado."""
    f1 = f1_score(y_test, trained_model.predict(X_test), average="weighted")
    assert f1 >= 0.78, f"Regressão de métrica detectada: F1={f1:.4f}"
```

---

## CI/CD (GitHub Actions)

Workflow files under `.github/workflows/`, plus `dependabot.yml`:

- `ci.yml` — Runs code quality checks: lint, type checking, security, complexity, dead code, and docstring coverage.
- `tests.yml` — Runs `pytest` on every pull request (minimum 80% coverage); **uploads coverage** (e.g., Codecov) and exposes a badge.
- `docs.yml` — Builds and publishes the documentation (MkDocs Material) on GitHub Pages.
- `mkdocs.yml` — Builds the documentation locally for preview.
- `dependabot.yml` — Scheduled dependency-update PRs; combine with `safety`/`pip-audit` run on a schedule.

Principles: fail fast, cache aggressively, run heavy checks (type/security/complexity) here rather than in pre-commit, and require green CI before merge.

---

## Containerization

- `Dockerfile` — Containerizes the application for consistent execution (multi-stage build; non-root user; pinned base image)
- `docker-compose.yml` — Orchestrates services (e.g., database, cache, MLflow server)
- `.devcontainer/devcontainer.json` — Fully reproducible dev environment so the project runs identically on any machine

---

## Notebooks

- Use notebooks **only** for exploration and final reporting
- Production logic belongs in `src/`, not notebooks
- Name with sequential prefix: `01_eda.ipynb`, `02_feature_engineering.ipynb`
- Restart kernel and run all cells before committing
- Strip outputs with `nbstripout` (configured in pre-commit) to keep the repo clean
- Standard notebook header:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
plt.rcParams["figure.dpi"] = 120
```

---

## Modeling Workflow

1. **Baseline first** — establish a simple benchmark before complex models
2. **Reproducibility** — seed everything; set `random_state` on every stochastic operation
3. **Validation** — use cross-validation; never evaluate on training data
4. **Leakage check** — verify no future information leaks into features; ensure train–serve feature consistency
5. **Track experiments** — log params, metrics, and artifacts with MLflow; register the chosen model in the MLflow Model Registry
6. **Persist models** — save with `joblib` or native format under `models/` (versioned via DVC/Registry, not Git)

### Rigorous evaluation (senior bar)

A single CV score is not enough. Always report and justify:

- **Uncertainty:** confidence intervals on metrics (CV std or bootstrap), not a bare point estimate
- **Model comparison:** test whether a difference is statistically significant (McNemar for classifiers on the same test set; corrected resampled *t*-test across CV folds) — never claim "better" from a single number
- **Calibration:** check probability calibration (reliability curve / Brier score) when probabilities are used for decisions
- **Slice-based evaluation:** report the headline metric per relevant subgroup, not only aggregated — aggregate metrics hide failures on important slices
- **Error analysis:** inspect the worst errors and document patterns; this drives the next iteration
- **Metric choice:** justify the main metric against the business cost of false positives vs. false negatives

```python
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

pipeline = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(random_state=42)),
    ]
)

scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="f1_weighted")
ci_95 = 1.96 * scores.std() / np.sqrt(len(scores))
logger.info(f"CV F1: {scores.mean():.4f} ± {ci_95:.4f} (IC 95%)")
```

---

## Fairness & Bias

Audit models for disparate behavior across sensitive subgroups. Required when outputs affect people, and strongly recommended for academic publication and LGPD-sensitive contexts.

- Identify sensitive attributes (e.g., region, age band, sex) and define them explicitly
- Compute the headline metric **and** fairness metrics per subgroup (e.g., true positive rate parity, predictive parity)
- Report disparities transparently in the Model Card; do not silently aggregate them away
- Use `fairlearn` to quantify and, where appropriate, mitigate disparity

```python
from fairlearn.metrics import MetricFrame, selection_rate
from sklearn.metrics import recall_score

mf = MetricFrame(
    metrics={"recall": recall_score, "selection_rate": selection_rate},
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=X_test["regiao"],
)
logger.info(f"Disparidade de recall por região:\n{mf.by_group}")
```

---

## Model & Data Versioning

Code is versioned in Git; **data and models are not**. Use DVC + the MLflow Model Registry for full lineage — you must be able to reproduce exactly which data + code + params produced which model.

- **DVC** tracks `data/` and `models/` (pointers in Git, content in remote storage); define reproducible pipeline stages in `dvc.yaml`
- **MLflow Tracking** logs params, metrics, and artifacts per run, tagged with the Git SHA and the dataset hash
- **MLflow Model Registry** promotes models through stages (Staging → Production) — use the Registry, not just loose `joblib` files
- `.gitignore` excludes data layers (bronze/silver/gold), models, and logs; DVC handles their storage

```bash
# Versionar dados grandes sem inflar o Git
dvc add data/bronze/transacoes
git add data/bronze/transacoes.dvc .gitignore
dvc push   # envia o conteúdo para o remote (S3, GDrive, etc.)
```

---

## Monitoring & Drift (post-deploy)

The lifecycle does not end at deploy. For any deployed model, monitor inputs and performance over time.

- **Data drift:** distribution shift in incoming features vs. the training reference
- **Concept drift:** the input–output relationship changes over time
- **Performance monitoring:** track the headline metric on labeled production data as it arrives
- Schedule drift reports (`evidently`) and alert when thresholds are exceeded; log results and trigger retraining when justified

```python
from evidently import Report
from evidently.presets import DataDriftPreset

report = Report(metrics=[DataDriftPreset()])
resultado = report.run(reference_data=ref_df, current_data=prod_df)
resultado.save_html("reports/monitoring/drift_report.html")
```

---

## Model Cards & Datasheets

Document models and datasets for transparency, reproducibility, and publication.

- **Model Card** (`reports/model_cards/`) per trained model: intended use, out-of-scope use, training data summary, evaluation **per slice**, fairness findings, known limitations, and ethical considerations
- **Datasheet for Datasets** (`reports/datasheets/`) per dataset: motivation, composition, collection process, preprocessing, licensing, and privacy/LGPD considerations
- Generate these from a template; keep them version-controlled alongside the model

---

## Data Privacy & LGPD

Treat privacy as a first-class requirement, not an afterthought (Lei Geral de Proteção de Dados — Brazil).

- **Minimization:** collect and keep only the data strictly needed for the task
- **No PII in outputs, logs, figures, commits, or notebooks** — ever
- **Anonymization / pseudonymization:** remove or hash direct identifiers before processing; be mindful of re-identification via quasi-identifiers
- **Legal basis:** document the legal basis for processing personal data in the Datasheet
- **Secrets:** credentials only via `.env` (never committed); scanned by `detect-secrets`
- **Right to deletion:** design pipelines so a subject's data can be located and removed
- For research/portfolio, prefer public, properly licensed, or already-anonymized datasets

```python
import hashlib


def pseudonimizar(identificador: str, salt: str) -> str:
    """Pseudonimiza um identificador direto com hash + salt (irreversível sem o salt)."""
    return hashlib.sha256(f"{salt}{identificador}".encode()).hexdigest()
```

---

## Documentation

- **MkDocs Material** + **mkdocstrings** generate an API reference directly from the pt-BR NumPy docstrings, plus hand-written guides
- Publish to GitHub Pages via `docs.yml`
- The `docs/` folder holds guides (setup, usage, architecture decisions); the API reference is auto-generated
- Keep `README.md` as the entry point with badges (CI, coverage, docs, license) linking into the full docs

```yaml
# mkdocs.yml (resumo)
site_name: Projeto DS
theme:
  name: material
plugins:
  - search
  - mkdocstrings
```

---

## Versioning & Releases

- **Semantic Versioning** (`MAJOR.MINOR.PATCH`) for every release/tag
- **Conventional Commits** (`feat:`, `fix:`, `docs:`, `refactor:`, etc.) — enforced via `commitizen` in pre-commit
- **CHANGELOG.md** in *Keep a Changelog* format, generated/updated by `commitizen` on release
- Tag releases in Git; optionally automate GitHub Releases in CI

```bash
cz commit          # commit guiado por Conventional Commits
cz bump --changelog  # incrementa versão (semver) e atualiza o CHANGELOG
```

---

## Performance & Profiling

Measure before optimizing; never optimize on intuition alone.

- **Polars lazy:** use `pl.scan_*` / `LazyFrame` and `.collect()` so the query optimizer plans the whole pipeline; use **streaming** for data larger than memory
- **Profiling:** `scalene` (CPU + memory + GPU) and `memray` (memory allocations) to find real bottlenecks — dev dependency only
- **Benchmark** critical steps and record before/after numbers to justify any optimization

```python
import polars as pl

# Lazy: o otimizador planeja filtros/seleções antes de materializar
resultado = (
    pl.scan_parquet("data/gold/transacoes.parquet")
    .filter(pl.col("valor") > 0)
    .group_by("regiao")
    .agg(pl.col("valor").mean().alias("ticket_medio"))
    .collect(streaming=True)  # processa em lotes, sem carregar tudo na RAM
)
```

---

## Visualization

- Use `seaborn` (preferred) and `matplotlib` for static plots; `plotly` for interactive visualizations (if applicable to project with a dashboard component such as Streamlit)
- Label all axes, add titles, include units where applicable
- Save figures to `reports/figures/` in `.png` (300 dpi) and `.svg`
- Define a shared color palette in `src/visualization/` and reuse it project-wide

```python
fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(data=df, x="revenue", hue="region", ax=ax)
ax.set_title("Distribuição de Receita por Região")
ax.set_xlabel("Receita (BRL)")
plt.tight_layout()
fig.savefig("reports/figures/receita_por_regiao.png", dpi=300)
```

---

## VSCode Settings

`.vscode/settings.json` — formatter, linting, Python interpreter path
`.vscode/launch.json` — debug configurations with breakpoints
`.devcontainer/devcontainer.json` — reproducible containerized dev environment

---

## What to Avoid

- Modifying bronze data — always read, transform, write to the next layer (silver → gold)
- Hardcoded paths or credentials — use `pathlib` + `.env`
- Silent failures — raise meaningful exceptions in pt-BR
- Production logic in notebooks
- Committing `.env`, large data files, model weights, or notebook outputs (use `.gitignore`, DVC, `nbstripout`)
- Bare `except:` blocks
- Dependencies without justification
- **Reporting a single point metric** without uncertainty, slice breakdown, or a significance test
- **Trusting data without validation** — always enforce the schema contract
- **PII in logs, outputs, figures, or commits**
- **Unpinned environments** — always commit the lock file
- **"Reproducible" without seeding everything** and hashing the data

---

## Project-Specific Overrides

> Fill in below when starting a new project. These settings override the defaults above.

```
Domain:                 [e.g., healthcare, finance, NLP, computer vision]
Target variable:        [e.g., churn (binary), price (continuous)]
Main metric:            [e.g., AUC-ROC, RMSE, F1-macro] + justification
Data sources:           [e.g., PostgreSQL, S3, local CSV, REST API]
Sensitive attributes:   [e.g., region, age band — for fairness audit; or none]
Special constraints:    [e.g., no PII in outputs, model must be interpretable, LGPD]
Deployment:             [e.g., FastAPI API, batch job, Streamlit dashboard]
Monitoring:             [e.g., weekly drift report, or none]
LLMs/Agents:            [e.g., uses LangChain + Ollama, or none]
```