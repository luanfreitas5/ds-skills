"""Classificação de texto de redes sociais em pt-BR: rótulos, split e escada de baselines.

Verificado com scikit-learn 1.9.1, spaCy 3.8.18 e sentence-transformers 6.2.0 (dados sintéticos).
"""

from __future__ import annotations

import logging
import re

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, make_pipeline

logger = logging.getLogger(__name__)

RT_RE = re.compile(r"^RT @\w+:\s*")  # retweet vira o texto original
URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
ELONGATION_RE = re.compile(r"(\w)\1{3,}")  # "muitoooooo" -> "muitooo"
KAPPA_GATE = 0.6


def normalize_social_text(text: str) -> str:
    """Normaliza o mínimo: troca URL e @menção por marcadores e encurta alongamentos.

    Mantém emojis, caixa, pontuação e negação ("não"), que carregam sentimento.

    Parameters
    ----------
    text : str
        Texto bruto do post.

    Returns
    -------
    str
        Texto normalizado.

    Examples
    --------
    >>> normalize_social_text("@ana NÃO gosteiiiiii 😡 https://t.co/x")
    '@user NÃO gosteiii 😡 http'
    >>> normalize_social_text("RT @bia: que dia lindo")
    'que dia lindo'
    """
    text = RT_RE.sub("", text)
    text = URL_RE.sub("http", text)
    text = MENTION_RE.sub("@user", text)
    text = ELONGATION_RE.sub(r"\1\1\1", text)
    return re.sub(r"\s+", " ", text).strip()


def compute_agreement(df: pd.DataFrame, col_a: str, col_b: str) -> dict:
    """Relatório de concordância: o entregável quando o gate de kappa falha.

    Parameters
    ----------
    df : pd.DataFrame
        Posts com um rótulo por anotador.
    col_a, col_b : str
        Colunas dos dois anotadores.

    Returns
    -------
    dict
        ``kappa``, ``per_class`` (um-contra-todos), ``crosstab`` (A × B) e
        ``marginals`` (distribuição de rótulos por anotador).
    """
    classes = sorted(set(df[col_a]) | set(df[col_b]))
    report = {
        "kappa": float(cohen_kappa_score(df[col_a], df[col_b])),
        "per_class": {c: round(float(cohen_kappa_score(df[col_a] == c, df[col_b] == c)), 3) for c in classes},
        "crosstab": pd.crosstab(df[col_a], df[col_b], rownames=["A"], colnames=["B"]),
        "marginals": pd.DataFrame({"A": df[col_a].value_counts(normalize=True),
                                   "B": df[col_b].value_counts(normalize=True)}),
    }
    logger.info(f"Kappa de Cohen: {report['kappa']:.3f}; por classe: {report['per_class']}")
    logger.info(f"Matriz anotador A × B:\n{report['crosstab']}")
    return report


def build_gold_labels(df: pd.DataFrame, col_a: str, col_b: str) -> pd.DataFrame:
    """Aplica o gate de kappa e separa rótulos consensuais dos que pedem adjudicação.

    Parameters
    ----------
    df : pd.DataFrame
        Posts com um rótulo por anotador.
    col_a, col_b : str
        Colunas dos dois anotadores.

    Returns
    -------
    pd.DataFrame
        Cópia com ``label`` (só onde há consenso, NaN no resto) e ``needs_adjudication``.
        Filtre ``~needs_adjudication`` antes de ``split_by_author``.

    Raises
    ------
    ValueError
        Se o kappa ficar abaixo de ``KAPPA_GATE``: o guia de anotação é o problema, não o modelo.
        Entregue ``compute_agreement`` no lugar do modelo.
    """
    kappa = compute_agreement(df, col_a, col_b)["kappa"]
    if kappa < KAPPA_GATE:
        raise ValueError(f"Kappa {kappa:.3f} < {KAPPA_GATE}: revise o guia de anotação antes de modelar.")
    out = df.copy()
    agreed = out[col_a] == out[col_b]
    out["label"] = out[col_a].where(agreed)
    out["needs_adjudication"] = ~agreed
    logger.info(f"{int((~agreed).sum())} posts sem consenso enviados para adjudicação.")
    return out


def split_by_author(
    df: pd.DataFrame, group_col: str = "user_id", seed: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split ~80/20 sem autor (nem texto repetido) dos dois lados.

    Retweets e copypasta repetem o mesmo texto entre autores; o texto normalizado
    vira parte do grupo, unindo esses autores no mesmo lado do split.

    Parameters
    ----------
    df : pd.DataFrame
        Posts com ``text_norm``, ``label`` e ``group_col``.
    group_col : str, optional
        Coluna do autor, by default "user_id".
    seed : int, optional
        Semente do embaralhamento, by default 42.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        Treino e teste.

    Raises
    ------
    ValueError
        Se ``label`` tiver nulos (passe só os posts de consenso).
    """
    if df["label"].isna().any():
        raise ValueError("Há rótulos nulos: filtre os posts de consenso antes do split.")
    groups = _merge_author_and_text_groups(df[group_col], df["text_norm"])
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_idx, test_idx = next(cv.split(df, df["label"], groups=groups))
    train, test = df.iloc[train_idx], df.iloc[test_idx]
    assert not set(train[group_col]) & set(test[group_col]), "Autor nos dois lados do split"
    assert not set(train["text_norm"]) & set(test["text_norm"]), "Texto repetido nos dois lados"
    top_share = test[group_col].value_counts(normalize=True).iloc[0]
    logger.info(f"Teste: {len(test) / len(df):.1%} dos posts; autor mais frequente = {top_share:.1%} do teste")
    return train, test


def _merge_author_and_text_groups(authors: pd.Series, texts: pd.Series) -> np.ndarray:
    """Une autores que compartilham um texto idêntico (union-find)."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for author, text in zip(authors.astype(str), texts, strict=True):
        parent[find("u:" + author)] = find("t:" + text)
    return np.array([find("u:" + a) for a in authors.astype(str)])


def build_tfidf_baseline(seed: int = 42):
    """TF-IDF de palavras (1-2) + caracteres (2-5) com regressão logística.

    N-gramas de caractere absorvem gíria, erro de grafia e emoji sem dicionário.
    """
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)),
    ])
    return make_pipeline(features, LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed))


def evaluate_ladder(train: pd.DataFrame, test: pd.DataFrame, encoder=None) -> dict[str, float]:
    """F1-macro de cada degrau da escada no mesmo teste.

    Parameters
    ----------
    train, test : pd.DataFrame
        Splits com ``text_norm`` e ``label``.
    encoder : SentenceTransformer, optional
        Modelo de embeddings congelado; se None, o degrau é pulado.

    Returns
    -------
    dict[str, float]
        F1-macro por degrau.
    """
    y_tr, y_te = train["label"], test["label"]
    results = {}
    dummy = DummyClassifier(strategy="most_frequent").fit(train[["text_norm"]], y_tr)
    results["majority"] = f1_score(y_te, dummy.predict(test[["text_norm"]]), average="macro")
    tfidf = build_tfidf_baseline().fit(train["text_norm"], y_tr)
    results["tfidf_lr"] = f1_score(y_te, tfidf.predict(test["text_norm"]), average="macro")
    if encoder is not None:
        emb_tr = encoder.encode(train["text_norm"].tolist(), normalize_embeddings=True)
        emb_te = encoder.encode(test["text_norm"].tolist(), normalize_embeddings=True)
        clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(emb_tr, y_tr)
        results["embeddings_lr"] = f1_score(y_te, clf.predict(emb_te), average="macro")
    for name, value in results.items():
        logger.info(f"Degrau {name}: F1-macro={value:.4f}")
    return results
