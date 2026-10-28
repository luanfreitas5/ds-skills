"""Auditoria de workflows N8N exportados (JSON) e helpers de idempotência.

Uso: ``python building-n8n-workflows_template.py workflow.json``
Retorna código 1 se houver achados. Verificado com Python 3.13 e dados sintéticos.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

SECRET_PATTERN = re.compile(
    r"(sk-[A-Za-z0-9_-]{16,}|Bearer\s+[A-Za-z0-9._-]{16,}|AKIA[0-9A-Z]{16}|"
    r"(?i:api[_-]?key|password|secret|token)['\"]?\s*[:=]\s*['\"][^'\"={]{8,})"
)
SEND_TYPES = ("gmail", "emailsend", "slack", "telegram", "httprequest")


def build_idempotency_key(*parts: str) -> str:
    """Gera chave determinística por efeito (ex.: lead + campanha), independente da execução.

    Parameters
    ----------
    *parts : str
        Campos que identificam o efeito (não use ``$execution.id`` nem timestamps).

    Returns
    -------
    str
        SHA-256 hexadecimal de 64 caracteres.

    Examples
    --------
    >>> build_idempotency_key("a@x.com", "camp-1") == build_idempotency_key("a@x.com", "camp-1")
    True
    """
    return hashlib.sha256("|".join(p.strip().lower() for p in parts).encode()).hexdigest()


def audit_workflow(workflow: dict) -> list[str]:
    """Lista problemas de um workflow N8N exportado.

    Parameters
    ----------
    workflow : dict
        JSON exportado (chaves ``nodes`` e ``settings``).

    Returns
    -------
    list of str
        Achados; lista vazia significa que nenhuma regra foi violada.

    Raises
    ------
    ValueError
        Se ``nodes`` estiver ausente.
    """
    if "nodes" not in workflow:
        raise ValueError("JSON sem a chave 'nodes': exporte o workflow pelo N8N.")
    findings: list[str] = []
    nodes = workflow["nodes"]
    for node in nodes:
        name, kind = node.get("name", "?"), node.get("type", "").lower()
        # Segredos em parâmetros (credenciais ficam em "credentials", fora de "parameters")
        if SECRET_PATTERN.search(json.dumps(node.get("parameters", {}))):
            findings.append(f"{name}: segredo gravado no nó; use Credentials")
        if any(t in kind for t in SEND_TYPES) and node.get("onError") is None:
            findings.append(f"{name}: efeito externo sem 'onError' definido")
    names = " ".join(n.get("type", "").lower() for n in nodes)
    if not workflow.get("settings", {}).get("errorWorkflow"):
        findings.append("workflow sem errorWorkflow (alerta de falha ausente)")
    has_send = any(t in names for t in SEND_TYPES)
    if has_send and "removeduplicates" not in names and "datatable" not in names:
        findings.append("efeito externo sem deduplicação/registro de enviados")
    return findings


def run_audit(path: Path) -> int:
    """Audita um arquivo e registra os achados em pt-BR."""
    findings = audit_workflow(json.loads(path.read_text(encoding="utf-8")))
    for item in findings:
        logger.error("Achado: %s", item)
    logger.info("Auditoria concluída: %d achado(s)", len(findings))
    return 1 if findings else 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    sys.exit(run_audit(Path(sys.argv[1])))
