---
name: plotting-publication-figures
description: Use when producing charts for a paper, thesis, report, slide, or README with matplotlib or seaborn, when figures must fit a journal column, when comparing models or folds in a figure, or when figures look different from one another across the project.
---

# Plotting Publication Figures

## Overview

A figure is read at print size, often in grayscale, by readers with color-vision deficiency, and it makes a claim that needs its uncertainty. **Core principle:** one project theme sized for the target column, colors that survive color blindness and grayscale, every axis with its unit, every comparison with an interval, saved as 300 dpi PNG + vector.

## Rules

| Rule | How |
|---|---|
| One theme | `src/visualization/theme.py`: `apply_theme()`, `PALETTE`, `save_figure()`; notebooks and scripts import it, never restyle locally |
| Size for the column | Create at final size: single column ≈ 8.4 cm (3.3 in), double ≈ 17.5 cm (6.9 in). Font 8–9 pt at that size; never shrink a 10-in figure in LaTeX |
| Color-blind safe | Okabe–Ito palette; model identity also encoded by marker/linestyle so it survives grayscale. Check: simulate deuteranopia (e.g. Coblis) and view `ImageOps.grayscale` |
| Units and labels | Axis label = quantity + unit: `Tempo de treino (s)`, `Receita (R$)`. Metric names spelled out once (`F1 (macro)`) |
| Uncertainty | Mean ± 95% CI (or all fold points + interval), never bare bars of means. CI method from REQUIRED SUB-SKILL: evaluating-models-rigorously (fold scores are correlated; corrected CI) |
| Output | `reports/figures/<name>.png` at 300 dpi **and** `.svg` (plus `.pdf` for LaTeX); `bbox_inches="tight"`; text kept as text (`svg.fonttype="none"`, `pdf.fonttype=42`) |
| Titles | Paper figures: no in-plot title (caption carries it). Report/README figures: title allowed |

## Implementation

```python
# src/visualization/theme.py
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# Okabe–Ito: distinguível em deuteranopia/protanopia
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
LINESTYLES = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)), "-", "--"]
SINGLE_COLUMN_IN = 3.3
DOUBLE_COLUMN_IN = 6.9
FIGURES_DIR = Path(__file__).resolve().parents[2] / "reports" / "figures"


def apply_theme() -> None:
    """Aplica o tema único do projeto (fontes de 8–9 pt no tamanho final)."""
    sns.set_theme(context="paper", style="ticks", palette=PALETTE)
    mpl.rcParams.update({
        "font.size": 8, "axes.labelsize": 9, "axes.titlesize": 9, "legend.fontsize": 7,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "lines.linewidth": 1.2,
        "savefig.dpi": 300, "svg.fonttype": "none", "pdf.fonttype": 42,
    })


def save_figure(fig: plt.Figure, name: str) -> list[Path]:
    """Salva a figura em PNG (300 dpi), SVG e PDF em reports/figures."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    paths = [FIGURES_DIR / f"{name}.{ext}" for ext in ("png", "svg", "pdf")]
    for path in paths:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    return paths
```

```python
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from visualization.theme import MARKERS, SINGLE_COLUMN_IN, apply_theme, save_figure


def plot_model_comparison(results: pd.DataFrame) -> plt.Figure:
    """Pontos por fold + média com IC 95% do F1 por modelo (coluna única)."""
    apply_theme()
    fig, ax = plt.subplots(figsize=(SINGLE_COLUMN_IN, 2.4))
    sns.stripplot(data=results, x="modelo", y="f1", color="0.6", size=3, jitter=0.15, ax=ax)
    sns.pointplot(data=results, x="modelo", y="f1", hue="modelo", errorbar=("ci", 95),
                  markers=MARKERS[: results["modelo"].nunique()], linestyles="none",
                  capsize=0.15, legend=False, ax=ax)
    ax.set(xlabel="Modelo", ylabel="F1 (macro), 10 folds")
    sns.despine(fig)
    save_figure(fig, "f1_por_modelo")
    return fig
```

`errorbar=("ci", 95)` bootstraps over folds, which treats them as independent; for the paper's claimed interval, compute the corrected CI and draw it with `ax.errorbar`.

## Common Mistakes

| Mistake | Fix |
|---|---|
| Bar chart of mean F1 per model | Fold points + mean with CI |
| Default `tab10`, color as only identity cue | Okabe–Ito + markers/linestyles |
| `figsize=(10, 6)` shrunk to a column | Create at column width; 8–9 pt fonts |
| `Tempo` without unit | `Tempo de treino (s)`; log scale labeled when used |
| Style set in each notebook | Import `apply_theme()` from `src/visualization/theme.py` |
| PNG 100 dpi only | PNG 300 dpi + SVG/PDF in `reports/figures/` |
| ROC legend with AUC and no interval | `AUC 0.91 [0.89, 0.93]` from bootstrap |
