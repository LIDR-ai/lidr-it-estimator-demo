"""Live per-agent activity log for the Session 13 graph (didactic UI feed).

The multi-agent graph normally runs to the next human gate inside one blocking
``ainvoke`` — nothing about *which agent is doing what* reaches the frontend. To
power a live "watch the agents work" panel, the streaming endpoints drive the graph
with ``graph.astream(..., stream_mode="updates")`` (one event per node as it
finishes) and append a short, human-readable line per event here.

Two interchangeable backends behind one contract (``reset`` / ``append`` / ``read``),
mirroring :mod:`app.generation.rag.idempotency`:

* **Redis** when ``REDIS_URL`` is configured — a list ``graph:activity:{id}`` with a
  TTL, shared across workers.
* **In-process dict** otherwise — so the feature works in tests / single-process dev
  without Redis.

``describe_node`` is a pure function: LangGraph node name + its state update → the
didactic line(s). It reads the very fields each agent already produces (see
``app/domain/graph/agents/*.py``). It never raises — an unknown shape degrades to a
generic line — because it runs inside the streaming loop.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone

import structlog

log = structlog.get_logger()

_KEY_PREFIX = "graph:activity:"

# LangGraph node function name → the stable key the UI groups by (matches the Rails
# ``Agents::GraphFlow::NODES`` keys so the panel can light the right row).
_NODE_KEYS: dict[str, str] = {
    "classifier_agent": "classifier",
    "structure_agent": "structure",
    "human_gate_structure": "gate_structure",
    "estimate_task_hours": "hours",
    "recover_and_handover": "recover",
    "analysis_agent": "analysis",
    "human_gate_analysis": "gate_analysis",
    "proposal_agent": "proposal",
}

_NODE_LABELS: dict[str, str] = {
    "classifier": "Classifier",
    "structure": "Structure",
    "gate_structure": "Gate 1",
    "hours": "Hours",
    "recover": "Recover",
    "analysis": "Analysis",
    "gate_analysis": "Gate 2",
    "proposal": "Proposal",
}


def _as_updates(update) -> list[dict]:
    """Normalise a node update to a list of dicts (fan-out yields a list)."""
    if isinstance(update, list):
        return [u for u in update if isinstance(u, dict)]
    if isinstance(update, dict):
        return [update]
    return []


def describe_node(node_name: str, update) -> list[dict]:
    """Map one ``astream`` chunk entry to didactic activity lines.

    Returns a list of ``{node, label, message}`` (usually one; the hours fan-out
    yields one per task). Best-effort and exception-free.
    """
    try:
        if node_name == "__interrupt__":
            return [{"node": "gate", "label": "Pausa", "message": "⏸ esperando revisión humana"}]

        key = _NODE_KEYS.get(node_name, node_name)
        label = _NODE_LABELS.get(key, key)
        updates = _as_updates(update)

        if node_name == "classifier_agent":
            complexity = _first(updates, "complexity") or "?"
            return [{"node": key, "label": label, "message": f"Complejidad: {complexity}"}]

        if node_name == "structure_agent":
            structure = _first(updates, "structure") or {}
            modules = structure.get("modules") or []
            tasks = sum(len(m.get("tasks") or []) for m in modules)
            return [
                {"node": key, "label": label, "message": f"{len(modules)} módulos · {tasks} tareas"}
            ]

        if node_name == "estimate_task_hours":
            lines: list[dict] = []
            for u in updates:
                for est in u.get("task_hours") or []:
                    task = est.get("task", "tarea")
                    if est.get("has_match") and est.get("estimated_hours") is not None:
                        msg = f"{task}: {est['estimated_hours']} h"
                    else:
                        msg = f"{task}: SIN ANÁLOGO"
                    lines.append({"node": key, "label": label, "message": msg})
            return lines or [{"node": key, "label": label, "message": "estimando horas…"}]

        if node_name == "recover_and_handover":
            estimate = _first(updates, "estimate") or {}
            days = estimate.get("total_engineer_days")
            tail = f" · {days} jornadas" if days is not None else ""
            jev = _first(updates, "jev_decisions") or []
            if jev:
                accepted = sum(
                    1
                    for t in (_first(updates, "task_hours") or [])
                    if t.get("accepted_by") == "jev"
                )
                tail += f" · Jev aceptó {accepted}/{len(jev)}"
            return [{"node": key, "label": label, "message": f"Estimación construida{tail}"}]

        if node_name == "analysis_agent":
            report = _first(updates, "analysis_report") or {}
            conf = report.get("overall_confidence", "?")
            ratio = report.get("grounded_task_ratio")
            tail = f" · fundamentadas {round(ratio * 100)}%" if isinstance(ratio, (int, float)) else ""
            return [{"node": key, "label": label, "message": f"Confianza {conf}{tail}"}]

        if node_name == "proposal_agent":
            proposal = _first(updates, "proposal") or ""
            return [
                {"node": key, "label": label, "message": f"Propuesta redactada ({len(proposal)} car.)"}
            ]

        if node_name == "human_gate_structure":
            return [{"node": key, "label": label, "message": "Estructura aprobada"}]

        if node_name == "human_gate_analysis":
            status = _first(updates, "status") or "validado"
            return [{"node": key, "label": label, "message": f"Validado ({status})"}]

        return [{"node": key, "label": label, "message": "hecho"}]
    except Exception as exc:  # noqa: BLE001 — never break the streaming loop.
        log.warning("describe_node_failed", node=node_name, error=str(exc)[:200])
        return [{"node": node_name, "label": node_name, "message": "…"}]


def _first(updates: list[dict], field: str):
    for u in updates:
        if field in u:
            return u[field]
    return None


class GraphActivityLog:
    """Redis-list-or-dict store of per-run didactic activity lines."""

    def __init__(self, redis_client=None, ttl: int = 3600) -> None:
        self._redis = redis_client
        self._ttl = ttl
        self._mem: dict[str, list[dict]] = {}
        self._lock = threading.Lock()

    @classmethod
    def from_settings(cls, settings) -> "GraphActivityLog":
        """Build the store, preferring Redis and degrading to the dict fallback."""
        redis_client = None
        if settings.REDIS_URL:
            try:
                import redis

                redis_client = redis.from_url(settings.REDIS_URL, decode_responses=True)
                redis_client.ping()
            except Exception as exc:  # noqa: BLE001 — fall back to in-memory.
                log.warning(
                    "graph_activity_redis_unavailable",
                    reason="setup_failed",
                    error_type=type(exc).__name__,
                )
                redis_client = None
        return cls(redis_client=redis_client)

    def reset(self, estimation_id: str) -> None:
        """Clear the log for a fresh START (resume appends, so it is not reset)."""
        if self._redis is not None:
            self._redis.delete(_KEY_PREFIX + estimation_id)
            return
        with self._lock:
            self._mem.pop(estimation_id, None)

    def append(self, estimation_id: str, node: str, label: str, message: str) -> None:
        """Append one activity line (seq = current length)."""
        entry = {
            "node": node,
            "label": label,
            "message": message,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        if self._redis is not None:
            key = _KEY_PREFIX + estimation_id
            pipe = self._redis.pipeline()
            entry["seq"] = self._redis.llen(key)
            pipe.rpush(key, json.dumps(entry))
            pipe.expire(key, self._ttl)
            pipe.execute()
            return
        with self._lock:
            bucket = self._mem.setdefault(estimation_id, [])
            entry["seq"] = len(bucket)
            bucket.append(entry)

    def read(self, estimation_id: str) -> list[dict]:
        """Return every activity line for a run, in order."""
        if self._redis is not None:
            raw = self._redis.lrange(_KEY_PREFIX + estimation_id, 0, -1)
            return [json.loads(item) for item in raw]
        with self._lock:
            return list(self._mem.get(estimation_id, []))
