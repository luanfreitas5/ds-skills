"""Gera a seção "Mudanças nos dados" de um PR comparando o dataset de main com o da branch.

Uso
---
python summarize_data_change.py main.parquet branch.parquet --target churn --key customer_id
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path

import polars as pl
from rich.console import Console
from rich.logging import RichHandler

logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler(console=Console(stderr=True))])
logger = logging.getLogger(__name__)


def compute_file_hash(path: Path) -> str:
    """Calcula o SHA-256 completo de um arquivo, lendo em blocos.

    Parameters
    ----------
    path : Path
        Caminho do arquivo de dados.

    Returns
    -------
    str
        Hash hexadecimal completo (nunca truncado no PR).

    Raises
    ------
    FileNotFoundError
        Se o arquivo não existir.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def compare_schemas(before: pl.DataFrame, after: pl.DataFrame) -> list[str]:
    """Lista colunas adicionadas, removidas, com tipo alterado ou com nulos alterados.

    Parameters
    ----------
    before : pl.DataFrame
        Dataset gerado pela main.
    after : pl.DataFrame
        Dataset gerado pela branch.

    Returns
    -------
    list[str]
        Linhas de tabela markdown ``| coluna | main | branch |``.
    """
    rows: list[str] = []
    for col in sorted(set(before.columns) | set(after.columns)):
        old = f"{before.schema[col]} ({before[col].null_count()} nulos)" if col in before.columns else "—"
        new = f"{after.schema[col]} ({after[col].null_count()} nulos)" if col in after.columns else "—"
        if old != new:
            rows.append(f"| `{col}` | {old} | {new} |")
    return rows


def summarize_data_change(before_path: Path, after_path: Path, target: str, key: str) -> str:
    """Monta o markdown com linhas, entidades, prevalência do alvo, hashes e schema.

    Parameters
    ----------
    before_path, after_path : Path
        Parquets de main e da branch (mesma etapa do pipeline).
    target : str
        Coluna-alvo binária (0/1).
    key : str
        Chave da entidade (ex.: ``customer_id``).

    Returns
    -------
    str
        Seção markdown pronta para colar no PR.

    Examples
    --------
    >>> print(summarize_data_change(Path("main.parquet"), Path("branch.parquet"), "churn", "customer_id"))
    """
    before, after = pl.read_parquet(before_path), pl.read_parquet(after_path)
    lost = before.select(key).unique().join(after.select(key).unique(), on=key, how="anti").height

    def pct(n_after: float, n_before: float) -> str:
        return f"{(n_after - n_before) / n_before:+.1%}" if n_before else "n/a"

    lines = [
        "| | main | branch | Δ |",
        "|---|---|---|---|",
        f"| Linhas | {before.height:_} | {after.height:_} | {pct(after.height, before.height)} |",
        f"| `{key}` distintos | {before[key].n_unique():_} | {after[key].n_unique():_} | {lost:_} perdidos |",
        f"| Prevalência `{target}` | {before[target].mean():.2%} | {after[target].mean():.2%} | "
        f"{(after[target].mean() - before[target].mean()) * 100:+.2f} p.p. |",
        f"| SHA-256 | `{compute_file_hash(before_path)}` | `{compute_file_hash(after_path)}` | |",
    ]
    schema_rows = compare_schemas(before, after)
    if schema_rows:
        lines += ["", "| Coluna | main | branch |", "|---|---|---|", *schema_rows]
    logger.info(f"Resumo gerado: {len(schema_rows)} colunas alteradas, {lost} entidades perdidas")
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resumo de mudanças nos dados para o PR.")
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--target", required=True)
    parser.add_argument("--key", required=True)
    args = parser.parse_args()
    print(summarize_data_change(args.before, args.after, args.target, args.key))
