"""Checagens quantitativas para mapas de saliência (Grad-CAM) de classificadores de imagem.

Verificado com torch 2.15.0, pytorch-grad-cam (grad-cam) 1.5.7, captum 0.9.0, scikit-learn 1.9.1.

Funções principais
------------------
- ``build_region_mask``: máscara booleana (H, W) de uma região suspeita (rótulo, marcador, borda).
- ``compute_cam``: Grad-CAM com um alvo POR imagem e aviso de mapas todos-zero.
- ``compute_region_mass``: fração da massa do mapa dentro da região (compare com a área da região).
- ``compute_masked_auc``: AUC original vs. AUC com a região neutralizada, com IC bootstrap da queda.
- ``compute_insertion_effect``: variação da probabilidade ao colar a região em imagens negativas.
- ``compute_cam_randomization_similarity``: teste de sanidade (pesos aleatórios em todas as camadas).
- ``compute_occlusion_region_mass``: checagem cruzada com captum (Occlusion), outro método.
"""

import copy
import logging

import numpy as np
import torch
from captum.attr import Occlusion
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


def build_region_mask(shape: tuple[int, int], rows: slice, cols: slice) -> torch.Tensor:
    """Cria uma máscara booleana (H, W) com True na região indicada.

    Parameters
    ----------
    shape : tuple[int, int]
        Altura e largura da imagem de entrada do modelo.
    rows, cols : slice
        Faixas de linhas e colunas da região (ex.: canto superior esquerdo).

    Returns
    -------
    torch.Tensor
        Máscara booleana (H, W).

    Examples
    --------
    >>> build_region_mask((64, 64), slice(0, 10), slice(0, 16)).sum().item()
    160
    """
    mask = torch.zeros(shape, dtype=torch.bool)
    mask[rows, cols] = True
    return mask


def predict_proba(model: torch.nn.Module, x: torch.Tensor, batch_size: int = 128) -> np.ndarray:
    """Probabilidade da classe positiva (índice 1), em lotes e sem gradiente."""
    model.eval()
    with torch.no_grad():
        return np.concatenate(
            [torch.softmax(model(x[i : i + batch_size]), 1)[:, 1].numpy() for i in range(0, len(x), batch_size)]
        )


def compute_cam(model: torch.nn.Module, layer: torch.nn.Module, x: torch.Tensor, class_index: int = 1) -> np.ndarray:
    """Grad-CAM da classe ``class_index`` com um alvo por imagem.

    ``targets=[alvo]`` com uma só entrada cobre apenas a primeira imagem do lote: as demais saem
    com mapa todo zero, sem erro. Por isso a lista é replicada e os mapas vazios são contados.

    Parameters
    ----------
    model : torch.nn.Module
        Modelo em modo ``eval``.
    layer : torch.nn.Module
        Última camada convolucional (saída 4D), por exemplo ``model.features[-1]`` no DenseNet
        da torchvision (antes do ReLU/pooling final, confira o shape).
    x : torch.Tensor
        Lote (N, C, H, W) já normalizado como no treino.
    class_index : int, optional
        Classe explicada, by default 1.

    Returns
    -------
    np.ndarray
        Mapas (N, H, W) em [0, 1].

    Examples
    --------
    >>> cams = compute_cam(model, model[7], x_val[:32])
    """
    model.eval()
    with GradCAM(model=model, target_layers=[layer]) as cam:
        cams = cam(input_tensor=x, targets=[ClassifierOutputTarget(class_index)] * len(x))
    n_zero = int((cams.reshape(len(cams), -1).max(axis=1) == 0).sum())
    if n_zero:
        logger.warning(f"{n_zero} de {len(cams)} mapas Grad-CAM ficaram todos zero (sem evidência da classe).")
    return cams


def compute_region_mass(cams: np.ndarray, region: torch.Tensor) -> np.ndarray:
    """Fração da massa de cada mapa que cai dentro da região (mapas todo-zero retornam 0)."""
    mask = region.numpy()
    total = cams.sum(axis=(1, 2))
    return np.where(total > 0, cams[:, mask].sum(axis=1) / np.maximum(total, 1e-12), 0.0)


def compute_masked_auc(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: np.ndarray,
    region: torch.Tensor,
    fill_value: float,
    n_boot: int = 500,
    seed: int = 42,
) -> dict[str, float]:
    """AUC original vs. AUC com a região preenchida por ``fill_value``, com IC 95% da queda.

    Parameters
    ----------
    fill_value : float
        Valor neutro (média do conjunto de treino no canal), nunca zero por conveniência.
    n_boot : int, optional
        Reamostragens bootstrap por imagem, by default 500. Com várias imagens por paciente,
        reamostre PACIENTES (REQUIRED SUB-SKILL: splitting-image-datasets).

    Returns
    -------
    dict[str, float]
        ``auc``, ``auc_masked``, ``drop``, ``drop_ci_low``, ``drop_ci_high``.
    """
    x_masked = x.clone()
    x_masked[:, :, region] = fill_value
    p, p_masked = predict_proba(model, x), predict_proba(model, x_masked)
    rng = np.random.default_rng(seed)
    drops = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        drops.append(roc_auc_score(y[idx], p[idx]) - roc_auc_score(y[idx], p_masked[idx]))
    auc, auc_masked = roc_auc_score(y, p), roc_auc_score(y, p_masked)
    return {
        "auc": auc,
        "auc_masked": auc_masked,
        "drop": auc - auc_masked,
        "drop_ci_low": float(np.percentile(drops, 2.5)),
        "drop_ci_high": float(np.percentile(drops, 97.5)),
    }


def compute_insertion_effect(
    model: torch.nn.Module, x_negative: torch.Tensor, donor: torch.Tensor, region: torch.Tensor
) -> dict[str, float]:
    """Variação média da probabilidade positiva ao colar a região de ``donor`` em imagens negativas.

    Parameters
    ----------
    x_negative : torch.Tensor
        Imagens da classe negativa (N, C, H, W).
    donor : torch.Tensor
        Uma imagem (C, H, W) que contém o marcador/região suspeita.
    """
    x_new = x_negative.clone()
    x_new[:, :, region] = donor[:, region]
    before, after = predict_proba(model, x_negative), predict_proba(model, x_new)
    return {"p_before": float(before.mean()), "p_after": float(after.mean()), "delta": float((after - before).mean())}


def compute_cam_randomization_similarity(
    model: torch.nn.Module, layer_name: str, x: torch.Tensor, seed: int = 0
) -> float:
    """Correlação de Spearman média entre o mapa do modelo treinado e o de um modelo com pesos aleatórios.

    Todas as camadas são reinicializadas: reinicializar só a camada final pode deixar o mapa quase
    idêntico (nos testes, correlação 0,9999) e passar a impressão de que o método é inofensivo.
    Correlação alta com o modelo aleatório significa que o mapa reflete a imagem, não o modelo.
    """
    torch.manual_seed(seed)
    random_model = copy.deepcopy(model)
    for module in random_model.modules():
        if hasattr(module, "reset_parameters"):
            module.reset_parameters()
    trained = compute_cam(model, model.get_submodule(layer_name), x)
    random = compute_cam(random_model, random_model.get_submodule(layer_name), x)
    corrs = [
        spearmanr(a.ravel(), b.ravel()).statistic for a, b in zip(trained, random) if a.std() > 0 and b.std() > 0
    ]
    return float(np.nanmean(corrs)) if corrs else float("nan")


def compute_occlusion_region_mass(
    model: torch.nn.Module, x: torch.Tensor, region: torch.Tensor, fill_value: float, class_index: int = 1,
    window: int = 8, stride: int = 4,
) -> float:
    """Fração da atribuição positiva (captum ``Occlusion``) dentro da região: confirma o Grad-CAM por outro método."""
    model.eval()
    occlusion = Occlusion(model)
    attr = occlusion.attribute(
        x, target=class_index, sliding_window_shapes=(x.shape[1], window, window),
        strides=(x.shape[1], stride, stride), baselines=fill_value,
    ).sum(dim=1).clamp(min=0)
    total = attr.sum(dim=(1, 2))
    inside = attr[:, region].sum(dim=1)
    return float((inside / total.clamp(min=1e-12))[total > 0].mean())


if __name__ == "__main__":
    # Demonstração em dados sintéticos: sinal pulmonar fraco + marcador correlacionado ao rótulo.
    import torch.nn as nn

    torch.manual_seed(0)
    size = 64
    tag = build_region_mask((size, size), slice(0, 10), slice(0, 16))

    def make_data(n: int, p_tag_pos: float, p_tag_neg: float, seed: int) -> tuple[torch.Tensor, np.ndarray]:
        rng = np.random.default_rng(seed)
        y = rng.integers(0, 2, n)
        x = rng.normal(0.3, 0.15, (n, 1, size, size)).astype("float32")
        for i in range(n):
            if y[i]:
                x[i, 0, 28:40, 30:44] += 0.04 * rng.uniform(0.5, 1.0)
            if rng.random() < (p_tag_pos if y[i] else p_tag_neg):
                x[i, 0][tag.numpy()] += 1.0
        return torch.tensor(x), y

    x_tr, y_tr = make_data(800, 0.95, 0.05, 0)
    x_te, y_te = make_data(600, 0.95, 0.05, 1)  # mesmo viés do treino (como um split por imagem)
    x_ext, y_ext = make_data(600, 0.5, 0.5, 2)  # "outro hospital": marcador independente do rótulo
    net = nn.Sequential(
        nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(16, 2),
    )
    opt = torch.optim.Adam(net.parameters(), 1e-2)
    y_t = torch.tensor(y_tr)
    for _ in range(8):
        for idx in torch.randperm(len(x_tr)).split(64):
            opt.zero_grad()
            nn.functional.cross_entropy(net(x_tr[idx]), y_t[idx]).backward()
            opt.step()
    net.eval()
    logging.basicConfig(level=logging.INFO)

    fill = float(x_tr.mean())
    print("AUC teste interno:", round(roc_auc_score(y_te, predict_proba(net, x_te)), 3))
    print("AUC externo:", round(roc_auc_score(y_ext, predict_proba(net, x_ext)), 3))
    print("mascarado:", compute_masked_auc(net, x_te, y_te, tag, fill))
    pos = x_te[y_te == 1][:100]
    print("massa CAM no marcador:", compute_region_mass(compute_cam(net, net[7], pos), tag).mean(), "| área:", tag.float().mean().item())
    print("massa Occlusion no marcador:", compute_occlusion_region_mass(net, pos[:20], tag, fill))
    neg = x_te[y_te == 0][:100]
    donor = x_te[(y_te == 1) & (x_te[:, 0][:, tag].mean(1).numpy() > 1.0)][0]
    print("inserção:", compute_insertion_effect(net, neg, donor, tag))
    print("Spearman treinado vs aleatório:", compute_cam_randomization_similarity(net, "7", pos))
