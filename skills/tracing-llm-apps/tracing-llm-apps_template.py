"""Rastreamento de chamadas LLM com OpenTelemetry (opentelemetry-sdk 1.45.1, verificado em 2026-10).

Um span por chamada, só com metadados: versão do prompt, modelo, tokens, custo, latência.
O conteúdo (entrada/saída) fica desligado por padrão.
"""

import json
import logging
import os
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.trace import Status, StatusCode

logger = logging.getLogger(__name__)
tracer = trace.get_tracer("llm-app")

_CPF = re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_DIGITS = re.compile(r"\d{3}[\d.\-]*")  # fragmentos de CPF/telefone que o regex completo perde


@dataclass(frozen=True)
class PriceTable:
    """Preço por 1M de tokens (USD), com versão para auditar o custo gravado."""

    version: str  # inclua origem e data de verificação, ex.: "2026-10-27 docs-do-provedor"
    prices: dict[str, tuple[float, float]]  # modelo servido -> (entrada, saída)


def compute_cost(table: PriceTable, model_served: str, tokens_in: int, tokens_out: int) -> float:
    """Calcula o custo em USD; modelo ausente na tabela levanta KeyError (nunca custo zero)."""
    price_in, price_out = table.prices[model_served]
    return (tokens_in * price_in + tokens_out * price_out) / 1_000_000


def redact_text(text: str) -> str:
    """Mascara CPF, e-mail e qualquer sequência de 3+ dígitos. Não cobre nomes: conteúdo é opt-in."""
    return _DIGITS.sub("[NUM]", _EMAIL.sub("[EMAIL]", _CPF.sub("[CPF]", text)))


def traced_llm_call(
    call_llm: Callable[..., dict],
    *,
    prompt_id: str,
    prompt_sha256: str,
    model: str,
    system: str,
    user: str,
    price_table: PriceTable,
    capture_content: bool = False,
) -> str:
    """Executa a chamada dentro de um span com os atributos do contrato de rastreamento.

    Parameters
    ----------
    call_llm : Callable[..., dict]
        Função do provedor; devolve ``text``, ``model`` (servido) e ``usage``.
    prompt_id, prompt_sha256 : str
        Identidade imutável do prompt (ver versioning-prompts).
    capture_content : bool, optional
        Se True, grava entrada/saída já mascaradas; por padrão False.

    Returns
    -------
    str
        Texto da resposta.

    Notes
    -----
    Modelo servido fora da tabela de preços marca ``app.cost.unpriced`` no span e segue
    devolvendo a resposta; o custo nunca é gravado como zero.
    """
    with tracer.start_as_current_span("llm.generate") as span:
        span.set_attributes(
            {
                "gen_ai.operation.name": "chat",
                "gen_ai.request.model": model,
                "app.prompt.id": prompt_id,
                "app.prompt.sha256": prompt_sha256,
                "app.price_table.version": price_table.version,
            }
        )
        start = time.perf_counter()
        try:
            answer = call_llm(model, system, user)
        except Exception as exc:
            span.record_exception(exc)
            span.set_status(Status(StatusCode.ERROR, type(exc).__name__))
            logger.error("Chamada LLM falhou: %s", type(exc).__name__)
            raise
        usage = answer["usage"]
        span.set_attributes(
            {
                "gen_ai.response.model": answer["model"],
                "gen_ai.usage.input_tokens": usage["prompt_tokens"],
                "gen_ai.usage.output_tokens": usage["completion_tokens"],
                "app.latency_ms": round((time.perf_counter() - start) * 1000, 1),
            }
        )
        try:
            cost = compute_cost(price_table, answer["model"], usage["prompt_tokens"], usage["completion_tokens"])
            span.set_attribute("app.cost.usd", cost)
        except KeyError:  # observabilidade nunca derruba o atendimento
            span.set_attribute("app.cost.unpriced", True)
            logger.warning("Modelo %s ausente na tabela de preços; custo não registrado", answer["model"])
        if capture_content:
            span.set_attribute("app.input", redact_text(user))
            span.set_attribute("app.output", redact_text(answer["text"]))
        return answer["text"]


class JsonlSpanExporter(SpanExporter):
    """Exporta spans para um arquivo JSONL local: o 'print' que dá para consultar depois."""

    def __init__(self, path: Path) -> None:
        self._path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        with self._path.open("a", encoding="utf-8") as handle:
            for span in spans:
                row = {
                    "trace_id": format(span.context.trace_id, "032x"),
                    "name": span.name,
                    "status": span.status.status_code.name,
                    **dict(span.attributes or {}),
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return SpanExportResult.SUCCESS


def configure_tracing(jsonl_path: Path) -> TracerProvider:
    """Configura o provider uma vez por processo, no entrypoint (chamadas seguintes são ignoradas). Adiciona OTLP se OTEL_EXPORTER_OTLP_ENDPOINT existir.

    Langfuse e LangSmith aceitam OTLP/HTTP: basta definir OTEL_EXPORTER_OTLP_ENDPOINT e
    OTEL_EXPORTER_OTLP_HEADERS no ambiente (chaves só no .env).
    """
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(JsonlSpanExporter(jsonl_path)))
    if os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    return provider


def summarize_by_prompt(jsonl_path: Path) -> dict[str, dict[str, float]]:
    """Agrupa os spans por versão do prompt: chamadas, custo médio e latência mediana."""
    rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line]
    summary: dict[str, dict[str, float]] = {}
    for sha in {r["app.prompt.sha256"] for r in rows if "app.prompt.sha256" in r}:
        group = [r for r in rows if r.get("app.prompt.sha256") == sha and r["status"] != "ERROR"]
        if not group:
            continue
        latencies = sorted(r["app.latency_ms"] for r in group)
        costs = [r["app.cost.usd"] for r in group if "app.cost.usd" in r]  # ignora spans sem preço
        summary[sha] = {
            "calls": len(group),
            "mean_cost_usd": sum(costs) / len(costs) if costs else float("nan"),
            "p50_latency_ms": latencies[len(latencies) // 2],
        }
    return summary
