import hashlib
import logging

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm

logger = logging.getLogger(__name__)


def assign_variant(user_id: str, salt: str, treatment_share: float = 0.5) -> str:
    """Atribui o usuário (nunca a sessão) a "control" ou "treatment" de forma determinística.

    Parameters
    ----------
    user_id : str
        Identificador estável do usuário (a unidade de randomização).
    salt : str
        Salt único do experimento.
    treatment_share : float, optional
        Fração do tráfego no tratamento, by default 0.5.

    Returns
    -------
    str
        ``"treatment"`` ou ``"control"``; aumentar a fração só move usuários para o tratamento.

    Examples
    --------
    >>> assign_variant("u42", salt="rank_v2_2026_11")
    """
    bucket = int(hashlib.sha256(f"{salt}:{user_id}".encode()).hexdigest()[:15], 16) % 10_000
    return "treatment" if bucket < treatment_share * 10_000 else "control"


def interleave_team_draft(
    list_a: list[str], list_b: list[str], rng: np.random.Generator, length: int = 10
) -> tuple[list[str], dict[str, str]]:
    """Intercala duas listas por Team-Draft; devolve a lista mostrada e o time de cada item.

    Parameters
    ----------
    list_a, list_b : list[str]
        Rankings do modelo atual (A) e do novo (B).
    rng : np.random.Generator
        Gerador com semente; a ordem de escolha é sorteada a cada rodada.
    length : int, optional
        Tamanho da lista mostrada, by default 10.

    Returns
    -------
    tuple[list[str], dict[str, str]]
        ``(lista, time_por_item)`` com times ``"A"`` e ``"B"``.

    Examples
    --------
    >>> shown, team = interleave_team_draft(a, b, np.random.default_rng(0))
    """
    shown: list[str] = []
    team: dict[str, str] = {}
    count = {"A": 0, "B": 0}
    sources = {"A": list_a, "B": list_b}
    while len(shown) < length:
        order = ["A", "B"] if count["A"] < count["B"] else ["B", "A"] if count["B"] < count["A"] else list(rng.permutation(["A", "B"]))
        progressed = False
        for t in order:
            if len(shown) >= length:
                break
            pick = next((i for i in sources[t] if i not in team), None)
            if pick is None:
                continue
            shown.append(pick)
            team[pick] = t
            count[t] += 1
            progressed = True
        if not progressed:
            break
    return shown, team


def evaluate_interleaving(impressions: pd.DataFrame) -> dict[str, float]:
    """Vitórias por usuário (clique em itens do time B vs A) e teste binomial bicaudal.

    Parameters
    ----------
    impressions : pd.DataFrame
        Uma linha por item mostrado: ``user_id``, ``team`` ("A"/"B"), ``clicked`` (0/1).

    Returns
    -------
    dict[str, float]
        ``win_rate_b`` (empates excluídos), ``n_decisive`` e ``p_value``.

    Examples
    --------
    >>> evaluate_interleaving(df)["p_value"]
    """
    clicks = impressions.pivot_table(index="user_id", columns="team", values="clicked", aggfunc="sum", fill_value=0)
    wins_b, wins_a = int((clicks["B"] > clicks["A"]).sum()), int((clicks["A"] > clicks["B"]).sum())
    n = wins_a + wins_b
    return {
        "win_rate_b": wins_b / n if n else float("nan"),
        "n_decisive": n,
        "p_value": float(binomtest(wins_b, n, 0.5).pvalue) if n else float("nan"),
    }


def compute_ratio_test(units: pd.DataFrame, numerator: str, denominator: str) -> dict[str, float]:
    """Razão de somas por grupo com variância pelo método delta, uma linha por usuário.

    Parameters
    ----------
    units : pd.DataFrame
        Uma linha por USUÁRIO: ``variant``, ``numerator``, ``denominator``.
    numerator, denominator : str
        Colunas (ex.: ``clicks`` e ``impressions``).

    Returns
    -------
    dict[str, float]
        ``control``, ``treatment``, ``diff``, ``ci_low``, ``ci_high`` (IC 95%), ``p_value``.

    Examples
    --------
    >>> compute_ratio_test(units, "clicks", "impressions")
    """
    stats = {}
    for name, g in units.groupby("variant"):
        y, x = g[numerator].to_numpy(float), g[denominator].to_numpy(float)
        r = y.sum() / x.sum()
        resid = y - r * x
        stats[name] = (r, np.var(resid, ddof=1) / len(g) / x.mean() ** 2)
    diff = stats["treatment"][0] - stats["control"][0]
    se = float(np.sqrt(stats["treatment"][1] + stats["control"][1]))
    z = norm.ppf(0.975)
    return {
        "control": stats["control"][0], "treatment": stats["treatment"][0], "diff": diff,
        "ci_low": diff - z * se, "ci_high": diff + z * se, "p_value": float(2 * norm.sf(abs(diff / se))),
    }


def compute_ctr_by_position(log: pd.DataFrame, max_position: int = 10) -> pd.DataFrame:
    """CTR por posição e variante: compara posições iguais, não a lista inteira.

    Parameters
    ----------
    log : pd.DataFrame
        Uma linha por impressão: ``variant``, ``position`` (1 = topo), ``clicked``.
    max_position : int, optional
        Última posição considerada, by default 10.

    Returns
    -------
    pd.DataFrame
        Índice ``position``, colunas por variante com o CTR.

    Examples
    --------
    >>> compute_ctr_by_position(log)
    """
    sub = log[log["position"] <= max_position]
    return sub.pivot_table(index="position", columns="variant", values="clicked", aggfunc="mean")
