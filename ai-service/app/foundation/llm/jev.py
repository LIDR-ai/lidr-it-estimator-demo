"""Jev (TypeSafe AI) — the external classifier that triages doubtful task hours.

Jev is a TypeSafe AI "System One" decision model, NOT an OpenAI-compatible chat model,
so it cannot go through LiteLLM / ``LLMWrapper``: this module is a thin, typed HTTP
client over TypeSafe's official API.

Contract (https://api.typesafe.ai/openapi.json, docs.typesafe.ai):

* ``POST https://api.typesafe.ai/v1/systemone`` with ``Authorization: Bearer <key>``
  (keys from https://console.typesafe.ai/keys).
* body ``{"model": "jev-latest", "state": {...}, "questions": {"<name>": {...}}}``; a
  ``choice`` question takes ``instructions`` + a ``criteria`` object keyed by option.
* response ``{"model": "jev-1.14.0", "answers": {"<name>": {"type": "choice", "choice",
  "confidence", "probabilities"}}, "usage": {"input_tokens", "output_tokens"}}`` — no
  envelope. Billed per INPUT token; output tokens are free.
* errors: 400/401/413/415/422 are contract errors (never retried); 408, 429, 529 and
  other 5xx are retried with exponential backoff (the official SDK's default policy).

Every failure degrades to ``escalate`` (hand the task to the recovery agent): Jev must
never be able to break a run. One request per task, under a semaphore: each task is its
own ``state``, so tasks never mix contexts.

Only the task's name/description, its numbers and its neighbours' hours/distances are
sent — never the transcript nor any corpus id.
"""

from __future__ import annotations

import asyncio
from typing import Literal, Protocol

import httpx
import structlog
from pydantic import BaseModel, ConfigDict, ValidationError

log = structlog.get_logger()

SYSTEMONE_PATH = "/v1/systemone"
QUESTION_ID = "triage"
# 408 timeout, 429 rate limit, 5xx (529 = TypeSafe overloaded): worth another attempt.
RETRYABLE_STATUSES = frozenset({408, 429, *range(500, 600)})
MAX_BACKOFF_SECONDS = 5.0

# Jev can only judge the consensus if it SEES the evidence behind it: without the
# neighbours it escalates everything with p=1.0 (measured against the live API). It
# also needs the distance SCALE: told only "lower = closer", it read every real match
# (0.30–0.45) as distant and escalated even "four agree + one outlier" (p≈0.25–0.42);
# with the scale it accepts those (p≈0.65–0.82) and still escalates analogs that split
# into two hour levels (p≈0.07). 0.45 mirrors TASK_HOURS_DISTANCE_THRESHOLD.
MAX_NEIGHBORS = 5
_QUESTION = {
    "type": "choice",
    "instructions": (
        "Decide whether estimated_hours can be accepted as is. It is a distance-weighted "
        "consensus of the historical analogs in neighbors. Distances are cosine distances "
        "on a scale where every analog below 0.45 already counts as a match and below 0.35 "
        "is a close match. One outlier among otherwise agreeing analogs is normal in "
        "historical data."
    ),
    "criteria": {
        "accept": (
            "Most analogs agree on similar hours (an isolated outlier is fine), so the "
            "consensus is usable as is."
        ),
        "escalate": (
            "The analogs split into clearly different hour levels, so the estimate needs a "
            "deeper review."
        ),
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
    # The evidence: the nearest historical tasks' hours and cosine distance (no ids).
    neighbors: list[dict] | None = None


class JevVerdict(BaseModel):
    """Jev's triage decision for one task. ``error`` set ⇒ forced ``escalate``.

    ``model`` is the version that answered (``jev-latest`` resolves to e.g.
    ``jev-1.14.0``) and ``usage`` its token counts. ``source="simulated"`` marks a
    verdict from the labelled heuristic stand-in (``JEV_FALLBACK=simulated``);
    ``fallback_reason`` keeps why the real Jev was not used.
    """

    module: str
    task: str
    decision: Literal["accept", "escalate"]
    confidence: float | None = None
    probabilities: dict[str, float] = {}
    error: str | None = None
    model: str | None = None
    usage: dict[str, int] | None = None
    source: Literal["jev", "simulated"] = "jev"
    fallback_reason: str | None = None


class ChoiceAnswer(BaseModel):
    """``answers.triage`` of a ``choice`` question (OpenAPI ``ChoiceAnswer``)."""

    model_config = ConfigDict(extra="allow")

    type: str
    choice: str
    confidence: float | None = None
    probabilities: dict[str, float] = {}


class SystemOneResponse(BaseModel):
    """``POST /v1/systemone`` 200 body (OpenAPI ``SystemOneResponse``)."""

    model_config = ConfigDict(extra="allow")

    model: str | None = None
    answers: dict[str, dict]
    usage: dict[str, int] | None = None


class JevClassifier(Protocol):
    """Anything that can triage a batch of doubtful tasks (the test seam)."""

    async def classify(self, tasks: list[JevTaskInput]) -> list[JevVerdict]: ...


def build_request(task: JevTaskInput, *, model: str) -> dict:
    """The ``/v1/systemone`` body for one task: its data as ``state``, one choice question."""
    state = task.model_dump(exclude_none=True)
    if task.neighbors:
        state["neighbors"] = [
            {"estimated_hours": n.get("estimated_hours"), "distance": n.get("distance")}
            for n in task.neighbors[:MAX_NEIGHBORS]
        ]
    return {"model": model, "state": state, "questions": {QUESTION_ID: _QUESTION}}


def parse_response(task: JevTaskInput, payload: dict) -> JevVerdict:
    """200 body → verdict. Raises ``ValueError`` on anything not a clean choice answer."""
    response = SystemOneResponse.model_validate(payload)
    raw = response.answers.get(QUESTION_ID)
    if not isinstance(raw, dict):
        raise ValueError("jev response has no answers.triage")
    answer = ChoiceAnswer.model_validate(raw)
    if answer.type != "choice":
        raise ValueError(f"jev answered a {answer.type!r} question, expected 'choice'")
    if answer.choice not in ("accept", "escalate"):
        raise ValueError(f"jev returned an unknown choice {answer.choice!r}")
    return JevVerdict(
        module=task.module,
        task=task.task,
        decision=answer.choice,
        confidence=answer.confidence,
        probabilities=answer.probabilities,
        model=response.model,
        usage=response.usage,
    )


class HttpJevClassifier:
    """``JevClassifier`` over TypeSafe's ``/v1/systemone`` (one call per task)."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.typesafe.ai",
        model: str = "jev-latest",
        timeout: float = 30.0,
        max_concurrency: int = 8,
        retries: int = 2,
        retry_backoff: float = 0.5,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._api_key = api_key
        self._retries = max(0, retries)
        self._retry_backoff = retry_backoff
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout
        self._max_concurrency = max(1, max_concurrency)
        self._transport = transport  # httpx.MockTransport in tests

    async def classify(self, tasks: list[JevTaskInput]) -> list[JevVerdict]:
        if not tasks:
            return []
        semaphore = asyncio.Semaphore(self._max_concurrency)
        # Circuit breaker for THIS batch: once one task proves Jev is down (retryable
        # status / timeout / connection error) after its retries, the rest stop calling.
        outage: list[str] = []
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
                    if outage:
                        return _escalated(task, f"jev unavailable (circuit open): {outage[0]}")
                    return await self._classify_one(client, task, outage)

            return list(await asyncio.gather(*(_one(t) for t in tasks)))

    async def _classify_one(
        self, client: httpx.AsyncClient, task: JevTaskInput, outage: list[str]
    ) -> JevVerdict:
        body = build_request(task, model=self._model)
        for attempt in range(self._retries + 1):
            try:
                response = await client.post(SYSTEMONE_PATH, json=body)
                response.raise_for_status()
                return parse_response(task, response.json())
            except (httpx.HTTPError, ValueError, ValidationError) as exc:
                error = _short_error(exc)
                if not _is_retryable(exc):
                    # Contract errors (4xx, garbled body): retrying would not change them.
                    log.warning(
                        "jev_classify_failed", module=task.module, task=task.task, error=error
                    )
                    return _escalated(task, error)
                if attempt < self._retries:
                    await asyncio.sleep(self._delay(exc, attempt))
                    continue
                if not outage:
                    outage.append(error)
                    log.warning("jev_circuit_open", error=error, retries=self._retries)
                return _escalated(task, error)
        raise AssertionError("unreachable")  # pragma: no cover

    def _delay(self, exc: Exception, attempt: int) -> float:
        """Exponential backoff, or the server's ``Retry-After`` (seconds) when it sends one."""
        if isinstance(exc, httpx.HTTPStatusError):
            retry_after = exc.response.headers.get("retry-after")
            try:
                if retry_after is not None:
                    return min(MAX_BACKOFF_SECONDS, max(0.0, float(retry_after)))
            except ValueError:
                pass  # an HTTP-date Retry-After: fall back to the exponential delay
        return min(MAX_BACKOFF_SECONDS, self._retry_backoff * 2**attempt)


def _escalated(task: JevTaskInput, error: str) -> JevVerdict:
    """Fail safe: an unreachable/garbled Jev means "not accepted" → recovery agent."""
    return JevVerdict(module=task.module, task=task.task, decision="escalate", error=error)


def _is_retryable(exc: Exception) -> bool:
    """Is this TypeSafe being busy/down (worth a retry) rather than a contract error?"""
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRYABLE_STATUSES
    return isinstance(exc, httpx.TransportError)  # timeouts, connection errors


def _short_error(exc: Exception) -> str:
    """One readable line (httpx appends a multi-line MDN hint to status errors).

    TypeSafe errors are JSON ``{"detail": "..." | {"error_type", "message"} | [...]}``;
    the message is kept because it is what tells a bad key from a bad request.
    """
    if not isinstance(exc, httpx.HTTPStatusError):
        return (str(exc).splitlines() or [type(exc).__name__])[0]
    response = exc.response
    head = f"HTTP {response.status_code} {response.reason_phrase}".strip()
    try:
        detail = (response.json() or {}).get("detail")
    except (ValueError, AttributeError):
        detail = None
    if isinstance(detail, dict):
        detail = detail.get("message") or detail.get("error_type")
    elif isinstance(detail, list) and detail:
        first = detail[0]
        detail = first.get("msg") if isinstance(first, dict) else str(first)
    return f"{head}: {str(detail)[:200]}" if detail else head


class HeuristicJevClassifier:
    """Labelled, deterministic stand-in for Jev — NOT Jev.

    Used only as ``JEV_FALLBACK=simulated`` when the real API cannot answer, so a demo
    still shows the triage step. The decision is explainable rather than random:
    ``p(accept)`` grows with the retrieval reliability inside Jev's band, and a
    contradictory range (``hours_range``) counts against it. Every verdict says
    ``source="simulated"``.
    """

    def __init__(self, *, min_reliability: float = 0.15) -> None:
        self._min_reliability = min_reliability

    def p_accept(self, task: JevTaskInput) -> float:
        p = 0.5 + ((task.reliability or 0.0) - self._min_reliability) * 2.5
        if task.hours_range is not None:
            p -= 0.3
        return round(min(0.95, max(0.05, p)), 3)

    async def classify(self, tasks: list[JevTaskInput]) -> list[JevVerdict]:
        verdicts = []
        for task in tasks:
            p = self.p_accept(task)
            verdicts.append(
                JevVerdict(
                    module=task.module,
                    task=task.task,
                    decision="accept" if p >= 0.5 else "escalate",
                    confidence=max(p, 1 - p),
                    probabilities={"accept": p, "escalate": round(1 - p, 3)},
                    source="simulated",
                )
            )
        return verdicts


class FallbackJevClassifier:
    """Real Jev first; ONLY the verdicts it could not give come from ``fallback``."""

    def __init__(self, primary: JevClassifier, fallback: JevClassifier) -> None:
        self._primary = primary
        self._fallback = fallback

    async def classify(self, tasks: list[JevTaskInput]) -> list[JevVerdict]:
        verdicts = await self._primary.classify(tasks)
        failed = [i for i, v in enumerate(verdicts) if v.error]
        if not failed:
            return verdicts
        substitutes = await self._fallback.classify([tasks[i] for i in failed])
        for i, sub in zip(failed, substitutes, strict=True):
            verdicts[i] = sub.model_copy(update={"fallback_reason": verdicts[i].error})
        log.info("jev_fallback_simulated", simulated=len(failed), total=len(tasks))
        return verdicts
