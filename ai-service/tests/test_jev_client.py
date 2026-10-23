"""The Jev (TypeSafe AI) HTTP client — network-free via ``httpx.MockTransport``.

Pins the request contract (Bearer auth, ``model``, one ``choice`` question named
``triage``, 32 KiB cap) and the fail-safe: every failure mode degrades to an
``escalate`` verdict carrying ``error``, never an exception.
"""

from __future__ import annotations

import json

import httpx

from app.foundation.llm.jev import (
    MAX_BODY_BYTES,
    HttpJevClassifier,
    JevTaskInput,
    build_decision_request,
)

TASK = JevTaskInput(
    module="Backend",
    task="Payments gateway",
    description="Integrate the card gateway",
    estimated_hours=32.0,
    reliability=0.28,
    flag_reason="low reliability (0.28)",
)


def _ok(choice="accept", accept=0.9):
    return {
        "code": 0,
        "message": "ok",
        "data": {
            "answers": {
                "triage": {
                    "choice": choice,
                    "probabilities": {"accept": accept, "escalate": round(1 - accept, 3)},
                    "confidence": max(accept, 1 - accept),
                }
            }
        },
    }


def _classifier(handler):
    return HttpJevClassifier(api_key="k-123", transport=httpx.MockTransport(handler))


async def test_request_follows_the_native_decisions_contract():
    seen: list[httpx.Request] = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=_ok())

    [verdict] = await _classifier(handler).classify([TASK])

    req = seen[0]
    assert req.method == "POST"
    assert str(req.url) == "https://www.jevai.org/api/v1/decisions"
    assert req.headers["Authorization"] == "Bearer k-123"
    body = json.loads(req.content)
    assert body["model"] == "typesafe-ai/jev"
    assert body["state"]["task"] == "Payments gateway"
    assert body["state"]["reliability"] == 0.28
    assert body["questions"]["triage"]["type"] == "choice"
    assert set(body["questions"]["triage"]["criteria"]) == {"accept", "escalate"}
    assert verdict.decision == "accept"
    assert verdict.probabilities["accept"] == 0.9
    assert verdict.error is None


async def test_one_request_per_task():
    calls: list[str] = []

    def handler(request):
        calls.append(json.loads(request.content)["state"]["task"])
        return httpx.Response(200, json=_ok(choice="escalate", accept=0.2))

    tasks = [TASK.model_copy(update={"task": f"t{i}"}) for i in range(5)]
    verdicts = await _classifier(handler).classify(tasks)

    assert sorted(calls) == [f"t{i}" for i in range(5)]
    assert [v.task for v in verdicts] == [f"t{i}" for i in range(5)]
    assert all(v.decision == "escalate" for v in verdicts)


async def test_nonzero_code_escalates_with_error():
    def handler(request):
        return httpx.Response(200, json={"code": 4001, "message": "quota exceeded"})

    [verdict] = await _classifier(handler).classify([TASK])
    assert verdict.decision == "escalate"
    assert "quota exceeded" in verdict.error


async def test_http_error_escalates_with_error():
    [verdict] = await _classifier(lambda r: httpx.Response(503)).classify([TASK])
    assert verdict.decision == "escalate"
    assert verdict.error


async def test_timeout_escalates_with_error():
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    [verdict] = await _classifier(handler).classify([TASK])
    assert verdict.decision == "escalate"
    assert verdict.error


async def test_unexpected_shapes_escalate_with_error():
    bodies = [
        {"code": 0, "data": {}},  # no answers
        {"code": 0, "data": {"answers": {"triage": {"choice": "maybe"}}}},  # unknown choice
        {"unexpected": True},  # no envelope
    ]
    for body in bodies:
        [verdict] = await _classifier(lambda r, b=body: httpx.Response(200, json=b)).classify(
            [TASK]
        )
        assert verdict.decision == "escalate", body
        assert verdict.error, body

    [verdict] = await _classifier(lambda r: httpx.Response(200, text="<html>")).classify([TASK])
    assert verdict.decision == "escalate"


async def test_empty_batch_makes_no_calls():
    def handler(request):  # pragma: no cover - must not be reached
        raise AssertionError("no request expected")

    assert await _classifier(handler).classify([]) == []


def test_oversized_description_is_trimmed_under_the_cap():
    big = TASK.model_copy(update={"description": "x" * (MAX_BODY_BYTES * 2)})
    body = build_decision_request(big, model="typesafe-ai/jev")
    assert len(json.dumps(body).encode()) <= MAX_BODY_BYTES
    assert body["state"]["description"].endswith("…")
