"""Template de fine-tuning de modelos de visão (torchvision).

Copiar para ``src/training/finetune.py``. Funções: normalização do backbone,
baseline de extração de features + regressão logística, congelamento com BatchNorm
em eval, grupos de parâmetros com LR por camada e agenda warmup + cosseno.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch import nn
from torchvision import models, transforms

logger = logging.getLogger(__name__)


def get_backbone_normalization(weights: models.WeightsEnum) -> tuple[list[float], list[float]]:
    """Lê média, desvio e tamanho esperados pelo backbone a partir dos pesos.

    Parameters
    ----------
    weights : models.WeightsEnum
        Ex.: ``models.ResNet18_Weights.IMAGENET1K_V1``.

    Returns
    -------
    tuple[list[float], list[float]]
        ``(mean, std)`` usados no pré-treino.

    Examples
    --------
    >>> mean, std = get_backbone_normalization(models.ResNet18_Weights.IMAGENET1K_V1)
    """
    meta = weights.transforms()
    return list(meta.mean), list(meta.std)


def build_eval_transform(weights: models.WeightsEnum) -> transforms.Compose:
    """Cria o transform de validação/teste com a normalização do backbone.

    Parameters
    ----------
    weights : models.WeightsEnum
        Pesos pré-treinados que definem média e desvio.

    Returns
    -------
    transforms.Compose
        Transform ``ToTensor`` + ``Normalize`` (o resize fica a cargo do pipeline de dados).
    """
    mean, std = get_backbone_normalization(weights)
    return transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])


def replace_head(model: nn.Module, num_classes: int) -> nn.Module:
    """Troca a camada final (``fc``) de uma ResNet por uma nova com ``num_classes`` saídas.

    Parameters
    ----------
    model : nn.Module
        ResNet da torchvision.
    num_classes : int
        Número de classes do problema.

    Returns
    -------
    nn.Module
        O mesmo modelo, com ``fc`` nova.

    Raises
    ------
    AttributeError
        Se o modelo não tiver o atributo ``fc``.
    """
    if not hasattr(model, "fc"):
        raise AttributeError("O modelo não possui atributo 'fc'; ajuste replace_head para esta arquitetura.")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


@torch.no_grad()
def extract_features(model: nn.Module, loader: torch.utils.data.DataLoader, device: str = "cpu"):
    """Extrai embeddings do backbone (sem a cabeça) para o baseline linear.

    Parameters
    ----------
    model : nn.Module
        ResNet com ``fc`` já substituída por ``nn.Identity``.
    loader : DataLoader
        Loader sem shuffle e sem augmentation.
    device : str, optional
        Dispositivo, by default "cpu".

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(features, labels)``.
    """
    model.eval().to(device)
    feats, labels = [], []
    for x, y in loader:
        feats.append(model(x.to(device)).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def freeze_backbone(model: nn.Module, head_prefix: str = "fc") -> None:
    """Congela todos os parâmetros exceto a cabeça.

    Parameters
    ----------
    model : nn.Module
        Modelo a congelar.
    head_prefix : str, optional
        Prefixo dos parâmetros da cabeça, by default "fc".
    """
    for name, param in model.named_parameters():
        param.requires_grad = name.startswith(head_prefix)
    logger.info("Backbone congelado; treináveis: %d parâmetros", count_trainable(model))


def set_train_mode(model: nn.Module) -> None:
    """Coloca o modelo em ``train`` mantendo BatchNorm congelada em ``eval``.

    ``requires_grad=False`` NÃO impede a BatchNorm de atualizar ``running_mean/var``
    em ``model.train()``; só o modo eval impede.

    Parameters
    ----------
    model : nn.Module
        Modelo cujo modo será definido.
    """
    model.train()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm) and not any(
            p.requires_grad for p in module.parameters(recurse=False)
        ):
            module.eval()


def count_trainable(model: nn.Module) -> int:
    """Conta parâmetros com ``requires_grad=True``."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_param_groups(
    model: nn.Module,
    head_lr: float = 1e-3,
    backbone_lr: float = 1e-4,
    layer_decay: float = 0.5,
    head_prefix: str = "fc",
    weight_decay: float = 1e-4,
) -> list[dict]:
    """Cria grupos de parâmetros com LR por camada (maior na cabeça, menor no início).

    Os blocos ``layer4 > layer3 > ...`` recebem ``backbone_lr * layer_decay**k``,
    onde ``k`` cresce em direção à entrada. Bias e BatchNorm ficam sem weight decay.

    Parameters
    ----------
    model : nn.Module
        ResNet da torchvision.
    head_lr : float, optional
        LR da cabeça, by default 1e-3.
    backbone_lr : float, optional
        LR do último bloco do backbone, by default 1e-4.
    layer_decay : float, optional
        Fator multiplicativo por bloco em direção à entrada, by default 0.5.
    head_prefix : str, optional
        Prefixo da cabeça, by default "fc".
    weight_decay : float, optional
        Decaimento de pesos para tensores com mais de 1 dimensão, by default 1e-4.

    Returns
    -------
    list[dict]
        Grupos prontos para ``torch.optim.AdamW``; só parâmetros treináveis.
    """
    order = ["layer4", "layer3", "layer2", "layer1"]
    groups: dict[tuple[str, bool], list] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith(head_prefix):
            lr = head_lr
            key = (head_prefix, param.ndim > 1)
        else:
            block = next((b for b in order if name.startswith(b)), "stem")
            k = order.index(block) if block in order else len(order)
            lr = backbone_lr * layer_decay**k
            key = (block, param.ndim > 1)
        groups.setdefault((key[0], key[1], lr), []).append(param)  # type: ignore[arg-type]
    return [
        {"params": ps, "lr": lr, "base_lr": lr, "weight_decay": weight_decay if decay else 0.0}
        for (_, decay, lr), ps in groups.items()
    ]


def set_warmup_cosine_lr(optimizer: torch.optim.Optimizer, step: int, total_steps: int, warmup_steps: int) -> None:
    """Aplica warmup linear e depois decaimento cosseno, preservando a razão entre grupos.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Otimizador cujos grupos têm a chave ``base_lr``.
    step : int
        Passo atual (0-indexado).
    total_steps : int
        Total de passos de treino.
    warmup_steps : int
        Passos de warmup.
    """
    if step < warmup_steps:
        scale = (step + 1) / warmup_steps
    else:
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        scale = 0.5 * (1 + np.cos(np.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = group["base_lr"] * scale
