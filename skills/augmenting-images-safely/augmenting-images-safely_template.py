"""Augmentation segura para imagens: validacao, auditoria de operacoes e mixers.

Copie para ``src/data/augmentation.py``. Testado com albumentations 2.x,
torchvision 0.29 (v2), numpy 2.5 em imagens sinteticas.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from torchvision.transforms import v2

logger = logging.getLogger(__name__)

# Operacoes que mudam o significado do rotulo, por tipo de tarefa.
FORBIDDEN_OPS: dict[str, tuple[type, ...]] = {
    "digits": (A.HorizontalFlip, A.VerticalFlip, A.Transpose, A.RandomRotate90),
    "text": (A.HorizontalFlip, A.VerticalFlip, A.Transpose, A.RandomRotate90),
    "laterality": (A.HorizontalFlip, A.Transpose, A.RandomRotate90),
}


def build_eval_transform(size: int, mean: Iterable[float], std: Iterable[float]) -> A.Compose:
    """Cria o transform deterministico de validacao/teste/servico.

    Parameters
    ----------
    size : int
        Lado da imagem de saida.
    mean, std : Iterable[float]
        Estatisticas do backbone (nao do dataset).

    Returns
    -------
    A.Compose
        Resize + Normalize + tensor, sem nenhuma operacao aleatoria.

    Examples
    --------
    >>> t = build_eval_transform(32, (0.5,) * 3, (0.5,) * 3)
    """
    return A.Compose(
        [A.Resize(size, size), A.Normalize(mean=tuple(mean), std=tuple(std)), ToTensorV2()]
    )


def _flatten(transform: A.BasicTransform | A.Compose) -> list[A.BasicTransform]:
    children = getattr(transform, "transforms", None)
    if children is None:
        return [transform]
    return [leaf for child in children for leaf in _flatten(child)]


def validate_forbidden_ops(compose: A.Compose, task: str) -> None:
    """Falha se o pipeline contem operacao que altera o rotulo da tarefa.

    Raises
    ------
    ValueError
        Se houver operacao proibida para ``task`` (inclusive em sub-Compose).
    KeyError
        Se ``task`` nao estiver em ``FORBIDDEN_OPS``.
    """
    forbidden = FORBIDDEN_OPS[task]
    found = sorted({type(t).__name__ for t in _flatten(compose) if isinstance(t, forbidden)})
    if found:
        raise ValueError(f"Operacoes que alteram o rotulo em '{task}': {found}")


def validate_eval_is_deterministic(transform: A.Compose, image: np.ndarray, n_runs: int = 5) -> None:
    """Falha se o transform de avaliacao retorna saidas diferentes para a mesma imagem.

    Raises
    ------
    ValueError
        Se alguma execucao divergir da primeira.
    """
    first = transform(image=image)["image"]
    for _ in range(n_runs - 1):
        if not torch.equal(first, transform(image=image)["image"]):
            logger.error("Transform de avaliacao e estocastico")
            raise ValueError("Transform de validacao/teste nao e deterministico")


def compute_object_survival(
    transform: A.Compose, image: np.ndarray, mask: np.ndarray, min_visible: float = 0.8,
    n_trials: int = 500,
) -> float:
    """Estima a fracao de amostras em que o objeto fica quase todo visivel.

    Parameters
    ----------
    transform : A.Compose
        Pipeline de treino a auditar (com saida em tensor ou numpy).
    image, mask : np.ndarray
        Imagem sintetica/real e mascara binaria do objeto (mesmo HxW).
    min_visible : float, optional
        Fracao minima da area do objeto que deve sobreviver, by default 0.8.
    n_trials : int, optional
        Numero de amostragens, by default 500.

    Returns
    -------
    float
        Proporcao de tentativas em que a fracao da imagem ocupada pelo objeto
        ficou >= ``min_visible`` x a fracao original (proxy: crop que corta o
        objeto reduz essa fracao; crop que o contem a aumenta).
    """
    # Referencia: area do objeto em fracao da imagem, medida sem crop aleatorio.
    base = float(mask.mean())
    ok = 0
    for _ in range(n_trials):
        out = transform(image=image, mask=mask)["mask"]
        out = np.asarray(out)
        ok += float(out.mean()) >= min_visible * base
    return ok / n_trials


def build_batch_mixer(num_classes: int, use_cutmix: bool = True, alpha: float = 1.0) -> v2.Transform:
    """Cria MixUp/CutMix para aplicar no lote (collate_fn), apenas no treino.

    Retorna rotulos suaves (``(B, num_classes)``): a perda deve aceitar alvo
    suave e as metricas de treino nao sao acuracia comum.

    Examples
    --------
    >>> mixer = build_batch_mixer(10)
    >>> x, y = mixer(images, labels)  # images (B,C,H,W), labels (B,)
    """
    ops: list[v2.Transform] = [v2.MixUp(alpha=alpha, num_classes=num_classes)]
    if use_cutmix:
        ops.append(v2.CutMix(alpha=alpha, num_classes=num_classes))
    return v2.RandomChoice(ops)
