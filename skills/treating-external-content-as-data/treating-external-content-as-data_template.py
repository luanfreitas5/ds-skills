"""Portão de ferramentas para LLMs que leem conteúdo não confiável (RAG, e-mails, páginas).

O modelo propõe; o código decide. Toda chamada de ferramenta passa por: allowlist,
validação Pydantic estrita, argumentos sensíveis fixados pelo servidor e confirmação
humana para efeitos externos. Verificado com pydantic 2.14.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field, ValidationError

logger = logging.getLogger(__name__)


class SendEmailArgs(BaseModel):
    """Argumentos que o modelo PODE escolher. Destinatário não está aqui: vem da sessão."""

    model_config = ConfigDict(extra="forbid")  # "to" vindo do modelo é rejeitado, não ignorado

    subject: str = Field(min_length=1, max_length=150)
    body: str = Field(min_length=1, max_length=4000)


@dataclass(frozen=True)
class Session:
    """Contexto autenticado do servidor; nunca derivado do texto do modelo ou dos documentos."""

    user_email: str
    user_asked_for_email: bool  # intenção detectada na mensagem do USUÁRIO, não no contexto


@dataclass(frozen=True)
class ToolSpec:
    """Ferramenta permitida: schema de argumentos, executor e se exige confirmação humana."""

    args_model: type[BaseModel]
    run: Callable[[Session, BaseModel], str]
    needs_confirmation: bool


@dataclass(frozen=True)
class ToolDecision:
    """Resultado do portão: executada, bloqueada ou aguardando confirmação."""

    status: str  # "executed" | "blocked" | "pending_confirmation"
    detail: str


def run_tool_call(
    name: str,
    raw_arguments: str,
    session: Session,
    registry: dict[str, ToolSpec],
    confirm: Callable[[str, BaseModel], bool],
) -> ToolDecision:
    """Valida e executa (ou bloqueia) uma chamada de ferramenta proposta pelo modelo.

    Parameters
    ----------
    name : str
        Nome da ferramenta proposto pelo modelo.
    raw_arguments : str
        JSON de argumentos proposto pelo modelo.
    session : Session
        Contexto autenticado; fonte única de destinatários, ids e permissões.
    registry : dict[str, ToolSpec]
        Allowlist explícita de ferramentas.
    confirm : Callable[[str, BaseModel], bool]
        Mostra ao usuário a ação exata e retorna sua decisão.

    Returns
    -------
    ToolDecision
        Decisão auditável; o `detail` volta ao modelo como observação.

    Examples
    --------
    >>> run_tool_call("delete_all", "{}", session, REGISTRY, confirm).status
    'blocked'
    """
    spec = registry.get(name)
    if spec is None:
        logger.warning(f"Ferramenta fora da allowlist bloqueada: {name}")
        return ToolDecision("blocked", f"ferramenta não permitida: {name}")
    try:
        args = spec.args_model.model_validate(json.loads(raw_arguments))
    except (json.JSONDecodeError, ValidationError) as exc:
        logger.warning(f"Argumentos inválidos para {name}: {type(exc).__name__}")
        return ToolDecision("blocked", "argumentos inválidos")
    if name == "send_email" and not session.user_asked_for_email:
        logger.warning("send_email proposto sem pedido do usuário; possível injeção no contexto")
        return ToolDecision("blocked", "o usuário não pediu envio de e-mail")
    if spec.needs_confirmation and not confirm(name, args):
        return ToolDecision("pending_confirmation", "aguardando confirmação do usuário")
    logger.info(f"Ferramenta executada: {name}")  # sem corpo nem destinatário no log (LGPD)
    return ToolDecision("executed", spec.run(session, args))


def wrap_untrusted(chunks: list[str]) -> str:
    """Delimita conteúdo externo como dado. Reduz, não impede, a injeção: o portão acima é a defesa."""
    def strip_tags(text: str) -> str:
        return text.replace("</doc>", "").replace("</untrusted_documents>", "")

    body = "\n".join(f"<doc>{strip_tags(c)}</doc>" for c in chunks)
    return f"<untrusted_documents>\n{body}\n</untrusted_documents>"
