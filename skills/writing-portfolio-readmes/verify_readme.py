"""Executa os comandos do README em um clone limpo do repositório.

Uso::

    python verify_readme.py caminho/do/repo --python 3.10
    python verify_readme.py caminho/do/repo --include-uncommitted  # antes do 1º commit

Somente blocos ```bash``` / ```sh``` entre ``<!-- verify:start -->`` e
``<!-- verify:end -->`` são executados, na ordem, com ``bash -euo pipefail``.
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import subprocess  # nosec B404 - executa comandos do próprio README, por design
import sys
import tempfile
from pathlib import Path

logger = logging.getLogger("verify_readme")

REGION = re.compile(r"<!--\s*verify:start\s*-->(.*?)<!--\s*verify:end\s*-->", re.S)
BLOCK = re.compile(r"```(?:bash|sh|shell)\n(.*?)```", re.S)
SKIP_DIRS = {".venv", "venv", "__pycache__", ".pytest_cache", ".mypy_cache"}
LEAKY_ENV = ("VIRTUAL_ENV", "CONDA_PREFIX", "CONDA_DEFAULT_ENV", "PYTHONPATH", "PYTHONHOME")


def extract_commands(readme: Path) -> list[str]:
    """Extrai os blocos shell da região verificável do README.

    Parameters
    ----------
    readme : Path
        Caminho do README.md.

    Returns
    -------
    list[str]
        Blocos de comandos, na ordem em que aparecem.

    Raises
    ------
    ValueError
        Se não houver região ``verify:start``/``verify:end`` com blocos shell.
    """
    text = readme.read_text(encoding="utf-8")
    blocks = [b for region in REGION.findall(text) for b in BLOCK.findall(region)]
    if not blocks:
        raise ValueError(f"Nenhum bloco bash entre <!-- verify:start/end --> em {readme}")
    return blocks


def create_clean_copy(repo: Path, dest: Path, include_uncommitted: bool) -> Path:
    """Cria uma cópia só com o que um visitante do GitHub receberia.

    Parameters
    ----------
    repo : Path
        Repositório de origem.
    dest : Path
        Diretório temporário de destino.
    include_uncommitted : bool
        Se True, copia arquivos rastreados + não ignorados (antes do commit);
        caso contrário, faz ``git clone`` do HEAD (somente o que foi commitado).

    Returns
    -------
    Path
        Diretório da cópia limpa.
    """
    target = dest / repo.resolve().name
    if not include_uncommitted:
        subprocess.run(["git", "clone", "--quiet", str(repo.resolve()), str(target)], check=True)  # noqa: S603, S607  # nosec B603 B607
        return target
    files = subprocess.run(  # noqa: S603  # nosec B603 B607
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],  # noqa: S607
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split("\0")
    if not (repo / ".gitignore").exists():
        logger.warning("Sem .gitignore: .venv, dados e caches iriam para o primeiro commit.")
    for rel in filter(None, files):
        src = repo / rel
        if SKIP_DIRS.intersection(Path(rel).parts):
            continue  # ambiente local nunca faz parte de um clone limpo
        if src.is_file():
            (target / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target / rel)
    return target


def build_clean_env(python: str | None) -> dict[str, str]:
    """Remove do ambiente tudo que mascara dependências não declaradas.

    Parameters
    ----------
    python : str or None
        Versão mínima declarada (ex.: ``"3.10"``), exportada como ``UV_PYTHON``.

    Returns
    -------
    dict[str, str]
        Variáveis de ambiente para os comandos.
    """
    env = {k: v for k, v in os.environ.items() if k not in LEAKY_ENV}
    env["PYTHONNOUSERSITE"] = "1"  # ignora pacotes do site-packages do usuário
    if python:
        env["UV_PYTHON"] = python
    return env


def run_commands(blocks: list[str], cwd: Path, env: dict[str, str], shell: str) -> bool:
    """Executa cada bloco e para no primeiro erro.

    Returns
    -------
    bool
        True se todos os blocos terminarem com código 0.
    """
    for i, block in enumerate(blocks, start=1):
        logger.info("Bloco %d/%d:\n%s", i, len(blocks), block.strip())
        result = subprocess.run([shell, "-euo", "pipefail", "-c", block], cwd=cwd, env=env, check=False)  # noqa: S603  # nosec B603
        if result.returncode != 0:
            logger.error("Bloco %d falhou (código %d). O README não funciona como está.", i, result.returncode)
            return False
    return True


def main() -> int:
    """Ponto de entrada da CLI."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("repo", type=Path)
    parser.add_argument("--readme", default="README.md")
    parser.add_argument("--python", help="versão mínima declarada, ex.: 3.10")
    parser.add_argument("--include-uncommitted", action="store_true")
    parser.add_argument("--shell", default=shutil.which("bash") or "bash")
    parser.add_argument("--keep", action="store_true", help="não apaga a cópia temporária")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s \t %(message)s")

    blocks = extract_commands(args.repo / args.readme)
    tmp = Path(tempfile.mkdtemp(prefix="readme-check-"))
    try:
        clone = create_clean_copy(args.repo, tmp, args.include_uncommitted)
        logger.info("Cópia limpa em %s", clone)
        ok = run_commands(blocks, clone, build_clean_env(args.python), args.shell)
    finally:
        if not args.keep:
            shutil.rmtree(tmp, ignore_errors=True)
    logger.info("README verificado com sucesso." if ok else "Verificação do README falhou.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
