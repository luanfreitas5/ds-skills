"""Agente LangGraph com orçamento explícito, estado tipado e ferramentas classificadas por risco.

Verificado com langgraph 1.2.18 e langchain-core 1.7.0. O modelo é injetado já com
`bind_tools` aplicado, para que os testes de trajetória usem um modelo roteirizado.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt

logger = logging.getLogger(__name__)

StopReason = Literal["answered", "max_steps", "max_tool_calls", "max_cost", "deadline", "repeated_call"]


@dataclass(frozen=True)
class Budget:
    """Limites duros de uma execução; ao atingir qualquer um, o agente encerra com resposta parcial."""

    max_steps: int = 8
    max_tool_calls: int = 12
    max_cost_usd: float = 0.50
    max_seconds: float = 120.0
    max_repeats: int = 2  # mesma tool com os mesmos argumentos


class AgentState(TypedDict):
    """Estado por execução (thread), nunca global."""

    messages: Annotated[list[AnyMessage], add_messages]
    steps: int
    tool_calls: int
    cost_usd: float
    started_at: float
    seen_calls: dict[str, int]
    stop_reason: StopReason | None


def create_initial_state(question: str) -> AgentState:
    """Cria o estado inicial de uma execução.

    Parameters
    ----------
    question : str
        Pergunta do usuário.

    Returns
    -------
    AgentState
        Estado com contadores zerados e relógio iniciado.

    Examples
    --------
    >>> create_initial_state("concorrentes do produto X")["steps"]
    0
    """
    return AgentState(
        messages=[("user", question)],  # type: ignore[typeddict-item]
        steps=0,
        tool_calls=0,
        cost_usd=0.0,
        started_at=time.monotonic(),
        seen_calls={},
        stop_reason=None,
    )


def compute_call_cost(message: AIMessage, usd_per_1k_tokens: float) -> float:
    """Custo aproximado de uma chamada a partir de `usage_metadata` (0 se ausente)."""
    usage = message.usage_metadata or {}
    return usage.get("total_tokens", 0) / 1000 * usd_per_1k_tokens


def check_budget(state: AgentState, budget: Budget) -> StopReason | None:
    """Retorna o motivo de parada se algum limite foi atingido, senão None."""
    if state["steps"] >= budget.max_steps:
        return "max_steps"
    if state["tool_calls"] >= budget.max_tool_calls:
        return "max_tool_calls"
    if state["cost_usd"] >= budget.max_cost_usd:
        return "max_cost"
    if time.monotonic() - state["started_at"] >= budget.max_seconds:
        return "deadline"
    if any(n > budget.max_repeats for n in state["seen_calls"].values()):
        return "repeated_call"
    return None


def build_agent(
    model: Runnable,
    tools: list[BaseTool],
    side_effect_tools: set[str],
    budget: Budget,
    usd_per_1k_tokens: float = 0.01,
    checkpointer: Any = None,
):
    """Monta o grafo do agente.

    Parameters
    ----------
    model : Runnable
        Chat model já com `bind_tools(tools)` aplicado (ou modelo roteirizado nos testes).
    tools : list[BaseTool]
        Ferramentas disponíveis.
    side_effect_tools : set[str]
        Nomes das ferramentas que escrevem, enviam ou gastam; exigem aprovação humana.
    budget : Budget
        Limites duros da execução.
    usd_per_1k_tokens : float, optional
        Preço usado para o orçamento de custo, by default 0.01.
    checkpointer : Any, optional
        Persistência por thread; obrigatório quando há ferramentas com efeito colateral.

    Returns
    -------
    CompiledStateGraph
        Grafo compilado; invoque com `config={"configurable": {"thread_id": ...}, "recursion_limit": ...}`.

    Raises
    ------
    ValueError
        Se houver ferramenta com efeito colateral e nenhum checkpointer (a aprovação não teria como retomar).
    """
    tools_by_name = {t.name: t for t in tools}
    if side_effect_tools and checkpointer is None:
        raise ValueError("Ferramentas com efeito colateral exigem checkpointer para aprovação humana.")

    def call_model(state: AgentState) -> dict:
        reply = model.invoke(state["messages"])
        return {
            "messages": [reply],
            "steps": state["steps"] + 1,
            "cost_usd": state["cost_usd"] + compute_call_cost(reply, usd_per_1k_tokens),
        }

    def run_tools(state: AgentState) -> dict:
        results, seen = [], dict(state["seen_calls"])
        calls = state["messages"][-1].tool_calls  # type: ignore[union-attr]
        for call in calls:
            key = f"{call['name']}:{sorted(call['args'].items())}"
            seen[key] = seen.get(key, 0) + 1
            tool = tools_by_name.get(call["name"])
            if tool is None:
                content = f"Erro: ferramenta desconhecida '{call['name']}'."
            elif call["name"] in side_effect_tools and not interrupt({"tool": call["name"], "args": call["args"]}):
                content = "Ação recusada pelo revisor humano."
            else:
                try:
                    content = str(tool.invoke(call["args"]))
                except Exception as exc:  # noqa: BLE001 - erro da tool vira observação, não derruba a execução
                    logger.warning(f"Falha na ferramenta {call['name']}: {exc}")
                    content = f"Erro na ferramenta: {type(exc).__name__}"
            results.append(ToolMessage(content=content, tool_call_id=call["id"], name=call["name"]))
        return {"messages": results, "tool_calls": state["tool_calls"] + len(calls), "seen_calls": seen}

    def finalize(state: AgentState) -> dict:
        last = state["messages"][-1]
        answered = isinstance(last, AIMessage) and not last.tool_calls
        reason: StopReason = "answered" if answered else (check_budget(state, budget) or "max_steps")
        if reason != "answered":
            logger.warning(f"Agente interrompido por orçamento: {reason}")
            note = AIMessage(content=f"[parcial: execução interrompida por {reason}]")
            return {"messages": [note], "stop_reason": reason}
        return {"stop_reason": reason}

    def route_after_model(state: AgentState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls and check_budget(state, budget) is None:
            return "tools"
        return "finalize"

    def route_after_tools(state: AgentState) -> str:
        return "agent" if check_budget(state, budget) is None else "finalize"

    graph = StateGraph(AgentState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", run_tools)
    graph.add_node("finalize", finalize)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route_after_model, ["tools", "finalize"])
    graph.add_conditional_edges("tools", route_after_tools, ["agent", "finalize"])
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer)


def compute_recursion_limit(budget: Budget) -> int:
    """Limite de super-steps do LangGraph como rede de segurança acima do orçamento (2 nós por passo + folga)."""
    return 2 * budget.max_steps + 4
