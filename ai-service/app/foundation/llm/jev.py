"""Jev (TypeSafe AI) — the external classifier that triages doubtful task hours.

Jev is NOT an OpenAI-compatible model (its docs explicitly say not to call
``/chat/completions``), so it cannot go through LiteLLM / ``LLMWrapper``: this module
is a thin, typed HTTP client over its **native decisions** endpoint.

Contract (https://www.jevai.org/docs):

* ``POST https://www.jevai.org/api/v1/decisions``
* headers ``Authorization: Bearer <key>`` + ``Content-Type: application/json``
* body ``{"model": "typesafe-ai/jev", "state": {...}, "questions": {...}}``, capped at
  32 KiB. A ``choice`` question takes a ``criteria`` object keyed by option and is
  answered with ``choice`` / ``probabilities`` / ``confidence``.
* response envelope ``{"code": 0, "message": "ok", "data": {...}}`` — ``code != 0``
  is an error; the per-question answers ride in ``data.answers.<question_id>``.

The docs do NOT publish a native response example, batching, rate limits or error
codes, so the answer parsing is deliberately defensive and every failure degrades to
``escalate`` (the caller's current behaviour: hand the task to the recovery agent).
Jev must never be able to break a run. One request per task, under a semaphore: the
contract has no batch endpoint, and packing N tasks into one ``state`` would mix their
contexts and risk the 32 KiB cap.

Only the task's name/description and its numbers are sent — never the transcript —
per Jev's own guidance not to send unrelated private data.
"""

from __future__ import annotations

import asyncio
import json
from typing import Literal, Protocol

import httpx
import structlog
from pydantic import BaseModel, ConfigDict, ValidationError

log = structlog.get_logger()

DECISIONS_PATH = "/api/v1/decisions"
MAX_BODY_BYTES = 32 * 1024
QUESTION_ID = "triage"

_POLICY = (
    "Accept the retrieved hours only if the historical analog is clearly comparable "
    "to this task; otherwise escalate it to a deeper agentic review."
)
_QUESTION = {
    "type": "choice",
    "instructions": "Decide whether the retrieved hour estimate can be accepted as is.",
    "criteria": {
        "accept": "The retrieved hours are trustworthy enough to use directly.",
        "escalate": "The estimate is doubtful; send it to the recovery agent.",
    },
}


class JevTaskInput(BaseModel):
    """One doubtful task handed to Jev (what goes into the request ``state``)."""

    module: str
    task: str
    description: str | None = None
    estimated_hours: float | None = None
    reliability: float | None = None
    hours_range: dict | None = None
    flag_reason: str | None = None


class JevVerdict(BaseModel):
    """Jev's triage decision for one task. ``error`` set ⇒ forced ``escalate``."""

    module: str
    task: str
    decision: Literal["accept", "escalate"]
    confidence: float | None = None
    probabilities: dict[str, float] = {}
    error: str | None = None


class JevChoiceAnswer(BaseModel):
    """``data.answers.triage`` — undocumented shape, so every field is optional."""

    model_config = ConfigDict(extra="allow")

    choice: str | None = None
    probabilities: dict[str, float] | None = None
    confidence: float | None = None


class JevEnvelope(BaseModel):
    """The ``{code, message, data}`` wrapper every Jev response uses."""

    model_config = ConfigDict(extra="allow")

    code: int
    message: str | None = None
    data: dict | None = None


class JevClassifier(Protocol):
    """Anything that can triage a batch of doubtful tasks (the test seam)."""

    async def classify(self, tasks: list[JevTaskInput]) -> list[JevVerdict]: ...


def build_decision_request(task: JevTaskInput, *, model: str) -> dict:
    """The native-decisions body for one task, trimmed to fit the 32 KiB cap.

    Only the free-text ``description`` can blow the cap, so it is the one field cut.
    """
    state = {**task.model_dump(), "policy": _POLICY}
    body = {"model": model, "state": state, "questions": {QUESTION_ID: _QUESTION}}
    size = len(json.dumps(body).encode())
    if size > MAX_BODY_BYTES and state.get("description"):
        overflow = size - MAX_BODY_BYTES + 64  # headroom for the ellipsis + escaping
        desc = state["description"].encode()
        state["description"] = desc[: max(0, len(desc) - overflow)].decode(errors="ignore") + "…"
    return body


def parse_decision_response(task: JevTaskInput, payload: dict) -> JevVerdict:
    """Envelope → verdict. Raises ``ValueError`` on anything not a clean answer."""
    envelope = JevEnvelope.model_validate(payload)
    if envelope.code != 0:
        raise ValueError(f"jev code {envelope.code}: {envelope.message}")
    raw = ((envelope.data or {}).get("answers") or {}).get(QUESTION_ID)
    if not isinstance(raw, dict):
        raise ValueError("jev response has no answers.triage")
    answer = JevChoiceAnswer.model_validate(raw)
    if answer.choice not in ("accept", "escalate"):
        raise ValueError(f"jev returned an unknown choice {answer.choice!r}")
    return JevVerdict(
        module=task.module,
        task=task.task,
        decision=answer.choice,
        confidence=answer.confidence,
        probabilities=answer.probabilities or {},
    )


class HttpJevClassifier:
    """``JevClassifier`` over the native decisions endpoint (one call per task)."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://www.jevai.org",
        model: str = "typesafe-ai/jev",
        timeout: float = 10.0,
        max_concurrency: int = 8,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._max_concurrency = max(1, max_concurrency)
        self._transport = transport  # httpx.MockTransport in tests

    async def classify(self, tasks: list[JevTaskInput]) -> list[JevVerdict]:
        if not tasks:
            return []
        semaphore = asyncio.Semaphore(self._max_concurrency)
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        async with httpx.AsyncClient(
            base_url=self._base_url,
            headers=headers,
            timeout=self._timeout,
            transport=self._transport,
        ) as client:

            async def _one(task: JevTaskInput) -> JevVerdict:
                async with semaphore:
                    return await self._classify_one(client, task)

            return list(await asyncio.gather(*(_one(t) for t in tasks)))

    async def _classify_one(self, client: httpx.AsyncClient, task: JevTaskInput) -> JevVerdict:
        try:
            response = await client.post(
                DECISIONS_PATH, json=build_decision_request(task, model=self._model)
            )
            response.raise_for_status()
            return parse_decision_response(task, response.json())
        except (httpx.HTTPError, ValueError, ValidationError) as exc:
            # Fail safe: an unreachable/garbled Jev means "not accepted" → recovery agent.
            log.warning("jev_classify_failed", module=task.module, task=task.task, error=str(exc))
            return JevVerdict(
                module=task.module, task=task.task, decision="escalate", error=str(exc)
            )
