"""Exportação para ONNX com paridade, quantização com portão e benchmark.

Funções: ``export_sklearn``, ``export_torch``, ``assert_onnx_parity``,
``benchmark_latency`` e ``quantize_with_gate``.
"""

import logging
import time
from pathlib import Path
from typing import Callable

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)

OPSET = 17  # fixe e registre junto do modelo
PROVIDERS = ["CPUExecutionProvider"]  # explícito: nada de GPU "por acaso"


def create_session(path: Path) -> ort.InferenceSession:
    """Cria uma sessão ONNX Runtime com provedores explícitos.

    Parameters
    ----------
    path : Path
        Arquivo ``.onnx``.

    Returns
    -------
    ort.InferenceSession
        Sessão pronta; crie uma vez na inicialização, nunca por requisição.

    Examples
    --------
    >>> session = create_session(Path("models/churn.onnx"))  # doctest: +SKIP
    """
    return ort.InferenceSession(str(path), providers=PROVIDERS)


def predict_proba_onnx(session: ort.InferenceSession, x: np.ndarray) -> np.ndarray:
    """Retorna P(classe 1) de uma sessão (sklearn: 2 saídas; torch: 1).

    Parameters
    ----------
    session : ort.InferenceSession
        Sessão ONNX.
    x : np.ndarray
        Matriz (n_linhas, n_features).

    Returns
    -------
    np.ndarray
        Vetor (n_linhas,) de probabilidades.

    Examples
    --------
    >>> predict_proba_onnx(session, x[:5])  # doctest: +SKIP
    """
    outputs = session.run(None, {"input": x.astype(np.float32)})
    proba = outputs[-1] if len(outputs) > 1 else outputs[0]
    return proba[:, 1] if proba.ndim == 2 and proba.shape[1] == 2 else proba.reshape(-1)


def export_sklearn(pipeline, n_features: int, path: Path) -> Path:
    """Exporta um ``Pipeline`` sklearn completo para ONNX com batch dinâmico.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Pipeline treinado (pré-processamento + modelo).
    n_features : int
        Número de colunas de entrada.
    path : Path
        Destino do arquivo ``.onnx``.

    Returns
    -------
    Path
        Caminho do arquivo gravado.

    Examples
    --------
    >>> export_sklearn(pipe, 30, Path("models/churn.onnx"))  # doctest: +SKIP
    """
    final_step = pipeline.steps[-1][1]
    onx = convert_sklearn(
        pipeline,
        initial_types=[("input", FloatTensorType([None, n_features]))],  # None = batch dinâmico
        options={id(final_step): {"zipmap": False}},  # saída em array, não lista de dicts
        target_opset={"": OPSET, "ai.onnx.ml": 3},
    )
    path.write_bytes(onx.SerializeToString())
    logger.info("Modelo sklearn exportado para %s", path)
    return path


def export_torch(model, n_features: int, path: Path) -> Path:
    """Exporta um ``nn.Module`` para ONNX com eixo de batch dinâmico.

    Parameters
    ----------
    model : torch.nn.Module
        Modelo treinado; colocado em ``eval()`` aqui.
    n_features : int
        Número de colunas de entrada.
    path : Path
        Destino do arquivo ``.onnx``.

    Returns
    -------
    Path
        Caminho do arquivo gravado.

    Examples
    --------
    >>> export_torch(net, 30, Path("models/mlp.onnx"))  # doctest: +SKIP
    """
    import torch

    model.eval()  # sem isso Dropout/BatchNorm exportam em modo de treino
    torch.onnx.export(
        model,
        torch.zeros(2, n_features),  # batch de exemplo != 1, para não especializar em 1
        str(path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=OPSET,
        dynamo=False,  # exportador TorchScript; dynamo=True exige onnxscript
    )
    logger.info("Modelo torch exportado para %s", path)
    return path


def assert_onnx_parity(
    path: Path,
    reference_fn: Callable[[np.ndarray], np.ndarray],
    x_test: np.ndarray,
    batch_sizes: tuple[int, ...] = (1, 7, 64, 500),
    atol: float = 1e-5,
    threshold: float = 0.5,
) -> float:
    """Compara probabilidades e rótulos do ONNX com o modelo original.

    Parameters
    ----------
    path : Path
        Arquivo ``.onnx`` (fp32).
    reference_fn : Callable
        Função ``x -> P(classe 1)`` do modelo original.
    x_test : np.ndarray
        Linhas de teste (não de treino), em número maior que o maior batch.
    batch_sizes : tuple of int, optional
        Tamanhos de batch a testar, by default (1, 7, 64, 500).
    atol : float, optional
        Diferença máxima de probabilidade, by default 1e-5.
    threshold : float, optional
        Limiar para comparar rótulos, by default 0.5.

    Returns
    -------
    float
        Maior diferença absoluta de probabilidade observada.

    Raises
    ------
    AssertionError
        Se a diferença exceder ``atol`` ou algum rótulo mudar.

    Examples
    --------
    >>> assert_onnx_parity(Path("m.onnx"), pipe_proba, x_test)  # doctest: +SKIP
    """
    session = create_session(path)
    worst = 0.0
    for size in batch_sizes:
        chunk = x_test[:size].astype(np.float32)
        got, expected = predict_proba_onnx(session, chunk), reference_fn(chunk)
        diff = float(np.max(np.abs(got - expected)))
        flips = int(np.sum((got >= threshold) != (expected >= threshold)))
        assert diff <= atol, f"batch={size}: diferença {diff:.2e} > {atol:.0e}"
        assert flips == 0, f"batch={size}: {flips} rótulos mudaram após a exportação"
        worst = max(worst, diff)
    logger.info("Paridade ok em %s: diferença máxima %.2e", path.name, worst)
    return worst


def benchmark_latency(
    predict_fn: Callable[[np.ndarray], object],
    x: np.ndarray,
    batch_size: int,
    runs: int = 200,
) -> dict[str, float]:
    """Mede p50 e p99 (ms) de uma função de predição, após aquecimento.

    Parameters
    ----------
    predict_fn : Callable
        Função que recebe um batch e prediz (original ou ONNX).
    x : np.ndarray
        Linhas de teste.
    batch_size : int
        Linhas por chamada.
    runs : int, optional
        Repetições medidas, by default 200.

    Returns
    -------
    dict of str to float
        ``{"p50_ms": ..., "p99_ms": ...}``.

    Examples
    --------
    >>> benchmark_latency(pipe.predict_proba, x, 1)  # doctest: +SKIP
    """
    batch = x[:batch_size].astype(np.float32)
    for _ in range(20):  # aquecimento: a primeira chamada paga alocações
        predict_fn(batch)
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        predict_fn(batch)
        times.append((time.perf_counter() - start) * 1000)
    return {"p50_ms": float(np.percentile(times, 50)), "p99_ms": float(np.percentile(times, 99))}


def quantize_with_gate(
    fp32_path: Path,
    x_test: np.ndarray,
    y_test: np.ndarray,
    max_auc_drop: float = 0.005,
    max_flip_rate: float = 0.002,
    threshold: float = 0.5,
    n_boot: int = 500,
    seed: int = 42,
) -> Path | None:
    """Quantiza para int8 e só aprova se métrica e latência justificarem.

    Parameters
    ----------
    fp32_path : Path
        Modelo ``.onnx`` fp32 de uma rede com MatMul/Gemm.
    x_test, y_test : np.ndarray
        Conjunto de teste para medir a perda de AUC.
    max_auc_drop : float, optional
        Queda máxima aceita de AUC, definida antes de medir, by default 0.005.
    max_flip_rate : float, optional
        Fração máxima de rótulos que mudam no limiar de serviço, by default 0.002.
    threshold : float, optional
        Limiar de decisão usado em produção, by default 0.5.
    n_boot : int, optional
        Reamostragens bootstrap pareadas, by default 500.
    seed : int, optional
        Semente do bootstrap, by default 42.

    Returns
    -------
    Path or None
        Caminho do int8 se aprovado; ``None`` se reprovado (mantenha o fp32).

    Raises
    ------
    RuntimeError
        Se o modelo não tem pesos quantizáveis (ex.: árvores).

    Examples
    --------
    >>> quantize_with_gate(Path("mlp.onnx"), x_test, y_test)  # doctest: +SKIP
    """
    int8_path = fp32_path.with_name(fp32_path.stem + "_int8.onnx")
    try:
        quantize_dynamic(str(fp32_path), str(int8_path), weight_type=QuantType.QInt8)
    except Exception as exc:  # noqa: BLE001 - o tipo do erro varia entre versões do ORT
        logger.error("Quantização falhou em %s: %s", fp32_path.name, exc)
        raise RuntimeError(f"{fp32_path.name} não tem pesos quantizáveis (árvores?)") from exc

    session32, session8 = create_session(fp32_path), create_session(int8_path)
    p32, p8 = predict_proba_onnx(session32, x_test), predict_proba_onnx(session8, x_test)
    rng = np.random.default_rng(seed)
    drops = []
    for _ in range(n_boot):  # bootstrap pareado: mesma amostra nos dois modelos
        idx = rng.integers(0, len(y_test), len(y_test))
        if len(np.unique(y_test[idx])) < 2:
            continue
        drops.append(roc_auc_score(y_test[idx], p32[idx]) - roc_auc_score(y_test[idx], p8[idx]))
    drop_upper = float(np.percentile(drops, 97.5))  # pior caso plausível da queda
    lat32 = benchmark_latency(lambda b: predict_proba_onnx(session32, b), x_test, 1)
    lat8 = benchmark_latency(lambda b: predict_proba_onnx(session8, b), x_test, 1)
    flip_rate = float(np.mean((p32 >= threshold) != (p8 >= threshold)))
    size_ratio = int8_path.stat().st_size / fp32_path.stat().st_size
    logger.info(
        "int8: queda de AUC (IC95%% superior) %.4f, limite %.4f; rótulos alterados %.4f; p99 %.2f ms contra %.2f ms; tamanho x%.2f",
        drop_upper, max_auc_drop, flip_rate, lat8["p99_ms"], lat32["p99_ms"], size_ratio,
    )
    faster = lat8["p99_ms"] < lat32["p99_ms"]
    smaller = size_ratio < 0.8
    if drop_upper > max_auc_drop or flip_rate > max_flip_rate or not (faster or smaller):
        logger.warning("int8 reprovado: mantenha o fp32")
        return None
    return int8_path
