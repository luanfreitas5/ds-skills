"""Pré-voo de servir LLM: orçamento de VRAM e verificação de saúde por inferência."""

from __future__ import annotations

import json
import logging
import urllib.request
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Bits efetivos por peso (aproximados; confirme o tamanho real do arquivo/log de carga).
BITS_PER_WEIGHT: dict[str, float] = {
    "fp16": 16.0,
    "fp8": 8.0,
    "q8_0": 8.5,
    "q4_k_m": 4.85,
    "awq_int4": 4.5,
    "q2_k": 3.0,
}


@dataclass(frozen=True)
class ModelSpec:
    """Arquitetura mínima para estimar memória.

    Parameters
    ----------
    params_b : float
        Parâmetros em bilhões.
    n_layers : int
        Número de camadas do transformer.
    n_kv_heads : int
        Cabeças de chave/valor (GQA).
    head_dim : int
        Dimensão por cabeça.
    """

    params_b: float
    n_layers: int
    n_kv_heads: int
    head_dim: int


LLAMA31_8B = ModelSpec(8.0, 32, 8, 128)
LLAMA31_70B = ModelSpec(70.6, 80, 8, 128)


def estimate_vram_gb(
    spec: ModelSpec,
    quant: str,
    context_tokens: int,
    concurrent_seqs: int,
    kv_bytes: int = 2,
    overhead_gb: float = 1.5,
) -> dict[str, float]:
    """Estima VRAM como pesos + KV cache + sobrecarga.

    Parameters
    ----------
    spec : ModelSpec
        Arquitetura do modelo.
    quant : str
        Chave de ``BITS_PER_WEIGHT``.
    context_tokens : int
        Contexto máximo POR sequência (prompt + resposta).
    concurrent_seqs : int
        Sequências simultâneas (vLLM ``--max-num-seqs``; Ollama ``NUM_PARALLEL``).
    kv_bytes : int, optional
        Bytes por elemento do KV cache (2 = fp16, 1 = q8/fp8), by default 2.
    overhead_gb : float, optional
        Ativações, grafos CUDA e fragmentação, by default 1.5.

    Returns
    -------
    dict[str, float]
        ``weights``, ``kv_cache``, ``overhead`` e ``total`` em GB.

    Raises
    ------
    ValueError
        Se ``quant`` for desconhecida ou os tamanhos forem não positivos.

    Examples
    --------
    >>> estimate_vram_gb(LLAMA31_8B, "q4_k_m", 4096, 50, kv_bytes=1)["total"] < 24
    True
    >>> estimate_vram_gb(LLAMA31_70B, "q2_k", 4096, 1)["total"] < 24
    False
    """
    if quant not in BITS_PER_WEIGHT:
        raise ValueError(f"Quantização desconhecida: {quant}")
    if min(context_tokens, concurrent_seqs) <= 0:
        raise ValueError("Contexto e concorrência devem ser positivos.")
    weights = spec.params_b * BITS_PER_WEIGHT[quant] / 8
    per_token = 2 * spec.n_layers * spec.n_kv_heads * spec.head_dim * kv_bytes
    kv_cache = per_token * context_tokens * concurrent_seqs / 1e9
    total = weights + kv_cache + overhead_gb
    return {"weights": weights, "kv_cache": kv_cache, "overhead": overhead_gb, "total": total}


def check_fits(estimate: dict[str, float], vram_gb: float, headroom: float = 0.9) -> bool:
    """Retorna True se o total estimado cabe em ``headroom`` da VRAM."""
    fits = estimate["total"] <= vram_gb * headroom
    logger.info("Estimativa %.1f GB de %.1f GB: %s", estimate["total"], vram_gb, "cabe" if fits else "NÃO cabe")
    return fits


def check_generation_health(base_url: str, model: str, timeout_s: float = 30.0) -> bool:
    """Saúde por inferência: gera 1 token via API compatível com OpenAI.

    ``/health`` (vLLM, llama-server) só diz que o processo está de pé; ``ollama list``
    não carrega o modelo. Esta chamada prova que o modelo responde.

    Parameters
    ----------
    base_url : str
        Ex.: ``http://localhost:8000/v1`` (vLLM) ou ``http://localhost:11434/v1`` (Ollama).
    model : str
        Nome do modelo servido.
    timeout_s : float, optional
        Tempo máximo, by default 30.0.

    Returns
    -------
    bool
        True se a resposta HTTP 200 contiver ``choices``.
    """
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": "ok"}], "max_tokens": 1}).encode()
    request = urllib.request.Request(  # noqa: S310 - URL interna controlada pelo operador
        f"{base_url}/chat/completions", data=body, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:  # noqa: S310
            return response.status == 200 and "choices" in json.load(response)
    except OSError:
        logger.error("Falha na verificação de geração em %s", base_url)
        return False


def run_load_test(
    base_url: str, model: str, concurrency: int, requests_total: int, max_p95_s: float, timeout_s: float = 120.0
) -> dict[str, float]:
    """Teste de carga: dispara requisições concorrentes e reporta p50/p95 e erros.

    Parameters
    ----------
    base_url : str
        Endpoint compatível com OpenAI (ex.: ``http://localhost:8000/v1``).
    model : str
        Nome do modelo servido.
    concurrency : int
        Usuários simultâneos alvo (use o número real, ex.: 50).
    requests_total : int
        Total de requisições (>= concurrency).
    max_p95_s : float
        Limite de aprovação para a latência p95, em segundos.
    timeout_s : float, optional
        Timeout por requisição, by default 120.0.

    Returns
    -------
    dict[str, float]
        ``p50_s``, ``p95_s``, ``errors`` e ``passed`` (1.0 se p95 <= limite e zero erros).

    Raises
    ------
    ValueError
        Se ``requests_total`` < ``concurrency``.
    """
    import statistics
    import time
    from concurrent.futures import ThreadPoolExecutor

    if requests_total < concurrency:
        raise ValueError("requests_total deve ser >= concurrency.")

    def one_call(_: int) -> float | None:
        started = time.perf_counter()
        ok = check_generation_health(base_url, model, timeout_s)
        return time.perf_counter() - started if ok else None

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(one_call, range(requests_total)))
    latencies = sorted(r for r in results if r is not None)
    errors = requests_total - len(latencies)
    p50 = statistics.median(latencies) if latencies else float("inf")
    p95 = latencies[min(len(latencies) - 1, int(0.95 * len(latencies)))] if latencies else float("inf")
    logger.info("Carga: p50=%.2fs p95=%.2fs erros=%d", p50, p95, errors)
    return {"p50_s": p50, "p95_s": p95, "errors": float(errors), "passed": float(errors == 0 and p95 <= max_p95_s)}
