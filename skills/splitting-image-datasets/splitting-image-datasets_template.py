"""Split de imagens por grupo (paciente/objeto/cena) com fusão de quase-duplicatas.

Funções principais
------------------
compute_dihedral_phashes, find_near_duplicate_pairs, build_split_groups, split_by_group.

Verificado com imagehash 4.3.2, scikit-learn 1.9.1, scipy 1.18.1, numpy 2.5.4, pillow 13.0.0.
"""

import logging
from pathlib import Path

import imagehash
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import StratifiedGroupKFold

logger = logging.getLogger(__name__)

DIHEDRAL = [
    lambda im: im, ImageOps.mirror, ImageOps.flip, lambda im: im.rotate(180),
    lambda im: im.transpose(Image.Transpose.ROTATE_90), lambda im: im.transpose(Image.Transpose.ROTATE_270),
    lambda im: im.transpose(Image.Transpose.TRANSPOSE), lambda im: im.transpose(Image.Transpose.TRANSVERSE),
]


def compute_dihedral_phashes(paths: list[Path]) -> np.ndarray:
    """pHash 64 bits de cada imagem nas 8 orientações (flips e rotações de 90°).

    Parameters
    ----------
    paths : list[Path]
        Caminhos das imagens.

    Returns
    -------
    np.ndarray
        Matriz ``uint64`` (n_imagens, 8); a coluna 0 é a orientação original.

    Examples
    --------
    >>> hashes = compute_dihedral_phashes(df["patch_path"].map(Path).tolist())
    """
    out = np.zeros((len(paths), len(DIHEDRAL)), dtype=np.uint64)
    for i, path in enumerate(paths):
        with Image.open(path) as img:
            rgb = img.convert("RGB")
            for j, transform in enumerate(DIHEDRAL):
                out[i, j] = int(str(imagehash.phash(transform(rgb))), 16)
    return out


def find_near_duplicate_pairs(hashes: np.ndarray, max_distance: int = 8, chunk: int = 512) -> np.ndarray:
    """Pares (i, j), i < j, cuja distância de Hamming mínima entre orientações é <= max_distance."""
    original = hashes[:, 0]
    pairs = []
    for start in range(0, len(hashes), chunk):
        block = hashes[start : start + chunk]  # (b, 8)
        dist = np.bitwise_count(block[:, :, None] ^ original[None, None, :]).min(axis=1)  # (b, n)
        rows, cols = np.nonzero(dist <= max_distance)
        rows += start
        keep = rows < cols
        pairs.append(np.stack([rows[keep], cols[keep]], axis=1))
    return np.concatenate(pairs) if pairs else np.empty((0, 2), dtype=int)


def build_split_groups(df: pd.DataFrame, pairs: np.ndarray, group_col: str = "patient_id") -> pd.Series:
    """Funde pacientes ligados por quase-duplicatas num mesmo grupo de split.

    Parameters
    ----------
    df : pd.DataFrame
        Uma linha por imagem, na mesma ordem de ``hashes``; contém ``group_col``.
    pairs : np.ndarray
        Saída de :func:`find_near_duplicate_pairs`.
    group_col : str, optional
        Unidade independente (paciente, objeto, cena), by default "patient_id".

    Returns
    -------
    pd.Series
        Id de grupo por imagem: componentes conexos de (mesmo paciente) ∪ (quase-duplicata).
    """
    codes, _ = pd.factorize(df[group_col])
    n_groups = codes.max() + 1
    a, b = codes[pairs[:, 0]], codes[pairs[:, 1]]
    graph = coo_matrix((np.ones(len(a)), (a, b)), shape=(n_groups, n_groups))
    n_components, labels = connected_components(graph, directed=False)
    cross = int((a != b).sum())
    logger.info(f"Quase-duplicatas: {len(pairs):,} pares, {cross:,} entre pacientes diferentes; "
                f"{n_groups:,} pacientes → {n_components:,} grupos de split")
    return pd.Series(labels[codes], index=df.index, name="split_group")


def split_by_group(df: pd.DataFrame, groups: pd.Series, label_col: str = "label", seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Um fold de StratifiedGroupKFold (~20% teste) com checagem de disjunção."""
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_idx, test_idx = next(sgkf.split(df, df[label_col], groups))
    overlap = set(groups.iloc[train_idx]) & set(groups.iloc[test_idx])
    if overlap:
        raise ValueError(f"Vazamento: {len(overlap)} grupos em treino e teste")
    return train_idx, test_idx


