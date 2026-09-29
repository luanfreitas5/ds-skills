---
name: fixing-python-warnings
description: Use when a Python run, notebook, or pytest prints FutureWarning, DeprecationWarning, PendingDeprecationWarning, Pandas4Warning, RuntimeWarning ("divide by zero", "invalid value encountered", "Mean of empty slice"), ConvergenceWarning, UserWarning, or SyntaxWarning; when asked to "clean up", "silence", or "get rid of" warnings; when upgrading pandas, numpy, scikit-learn, or Python; or when tempted to use warnings.filterwarnings("ignore"), np.errstate, or catch_warnings.
---

# Fixing Python Warnings

## Overview

A warning is one of two messages. **Deprecation warnings** (`FutureWarning`, `DeprecationWarning`, `Pandas4Warning`) say "this call breaks in the next version". **Runtime and data warnings** (`RuntimeWarning`, `ConvergenceWarning`, many `UserWarning`s) say "your numbers are wrong now". Silencing either hides a future crash or a present bug.

**Core principle:** fix the call site. Suppress only a warning that comes from code you don't own, with a narrow filter, in one central place, with a reason and a removal condition.

## Step 1: See Every Warning

Default filters hide `DeprecationWarning` (and `Pandas4Warning`, which subclasses it) unless it comes from `__main__`. If the code is imported from `src/main.py`, pandas deprecations are invisible (measured: 3 `Pandas4Warning`s appear only with `-W default`).

```bash
python -W default -X dev src/main.py          # show all, once per location
python -W error src/main.py                   # turn the first one into a traceback
python -m pytest -W error                     # same for tests (first pass only)
```

`-W` on the pytest command line overrides `filterwarnings` in `pyproject.toml`. After Step 5, run plain `python -m pytest`, or the third-party ignores stop working.

The printed location is often wrong (`pandas/core/arraylike.py:402: RuntimeWarning: divide by zero`). `-W error` gives a traceback that points to **your** line.

## Step 2: Triage by Category

| Category | Meaning | Fix |
|---|---|---|
| `FutureWarning` / `DeprecationWarning` / `Pandas4Warning` in your code | API changes or removal soon | Use the replacement the message names. Check it exists at your dependency **floor** (`datetime.UTC` needs 3.11; sklearn `l1_ratio` needs 1.8) |
| `RuntimeWarning: divide by zero` / `invalid value` / `Mean of empty slice` | `inf`/`NaN` in your data **now** | Find the rows. Handle them with an explicit mask. Log the count. See below |
| `ConvergenceWarning` | Model is not fitted | Scale features, then raise `max_iter`. Never ignore: metrics from an unconverged model are not reproducible |
| `SettingWithCopyWarning` (pandas < 3) / `ChainedAssignmentError` | Write does not land | `.loc[mask, col] = ...` on the frame you own |
| `SyntaxWarning: invalid escape sequence` | Becomes an error | Raw string `r"\d"` |
| Any warning raised **inside** a dependency you cannot edit | Not yours to fix | Upgrade if a fixed version exists; otherwise narrow filter (Step 4) |

"The message goes away" is not the test. The test is: same numbers, no hidden `inf`/`NaN`, and still valid on the next major version.

## Step 3: Data Warnings Without Changing the Numbers

"Don't change the numbers" is not a reason for `np.errstate(divide="ignore")`. An explicit mask usually gives **identical** downstream values and makes the bad rows visible:

```python
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cria ``valor_log`` e ``renda_por_valor``; valores <= 0 viram NaN explícito."""
    out = df.copy()
    positivo = out["valor"] > 0
    n_invalidos = int((~positivo).sum())
    if n_invalidos:
        logger.warning("%d linha(s) com valor <= 0: features viram NaN", n_invalidos)
    out["valor_log"] = np.log(out["valor"].where(positivo))
    out["renda_por_valor"] = out["renda"] / out["valor"].where(positivo)
    return out
```

Verified: F1 identical to 16 digits against the `-inf`/`inf` + `replace(inf, nan).fillna(0)` version, with no warning. If the numbers **would** change, report the bug and the size of the change. Don't hide it to keep a slide.

## Step 4: Third-Party Warnings (only these may be filtered)

- Filter by **message regex + category**. Don't filter by `module=`: the module is the caller chosen by `stacklevel`, often `pandas.core.algorithms`, so `module="geoutil"` does not match (tested).
- Put the filter in **one** place: `pyproject.toml` for tests, and one `configure_warnings()` called at the entry point for scripts. Don't use inline `with warnings.catch_warnings()`: it changes global state and is not thread-safe.
- Each filter has a comment: package, pinned version, and when to remove it.

```python
def configure_warnings() -> None:
    """Filtros de avisos de terceiros; chamado uma vez no ponto de entrada."""
    # geoutil 2.3.1 (outro time): remover ao migrar para geoutil 3.0.
    warnings.filterwarnings(
        "ignore", message=r"geoutil\.region_weight uses a legacy", category=FutureWarning
    )
```

## Step 5: Lock It In

```toml
[tool.pytest.ini_options]
filterwarnings = [
    "error",   # todo warning novo quebra o CI
    # geoutil 2.3.1 (outro time): remover ao migrar para 3.0.
    'ignore:geoutil\.region_weight uses a legacy:FutureWarning',
]
```

Without `"error"`, fixed warnings return with the next edit. Run `python -m pytest` and `python -W error <entry point>` before you report done.

## Red Flags: Stop

| Thought | Reality |
|---|---|
| "`np.errstate(ignore)` only on this one line is narrow" | Narrow scope still hides the bug in that line. Mask the rows instead. |
| "The zeros are there on purpose" | Did you verify that, or assume it? Log the count and ask. |
| "A real fix would change the F1" | Check it. A mask often gives identical numbers. If not, the F1 on the slide is wrong. |
| "`warnings.filterwarnings('ignore')` at the top, demo in 10 min" | Also hides tomorrow's `ConvergenceWarning`. Most fixes take one line each. |
| "It's only a FutureWarning" | It is a scheduled crash: `applymap`, `fillna(method=)`, `freq="H"` are already errors in pandas 3. |
| "No warnings in the run, done" | Imported modules hide `DeprecationWarning`. Run with `-W default`. |
| "Filter with `module='lib'`" | The module is the caller, not the library. Match the message. |

Dependency upgrades that trigger the warnings: REQUIRED SUB-SKILL: auditing-python-dependencies. Tests that guard the fix: REQUIRED SUB-SKILL: testing-ml-code.
