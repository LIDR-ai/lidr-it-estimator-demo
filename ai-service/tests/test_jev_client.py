"""The Jev (TypeSafe AI) HTTP client — network-free via ``httpx.MockTransport``.

Pins the official contract (``POST https://api.typesafe.ai/v1/systemone``, Bearer auth,
``model="jev-latest"``, one ``choice`` question named ``triage``; envelope-less
response with ``model`` + ``usage``), the retry policy (408/429/5xx retried, contract
errors not) and the fail-safe: every failure mode degrades to an ``escalate`` verdict
carrying ``error``, never an exception.
"""

from __future__ import annotations

import json

import httpx

from app.foundation.llm.jev import (
    MAX_NEIGHBORS,
    FallbackJevClassifier,
    HeuristicJevClassifier,
    HttpJevClassifier,
    JevTaskInput,
    build_request,
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
    """A real ``/v1/systemone`` 200 body (shape captured from the live API)."""
    return {
        "model": "jev-1.14.0",
        "answers": {
            "triage": {
                "type": "choice",
                "choice": choice,
                "confidence": max(accept, 1 - accept),
                "probabilities": {"accept": accept, "escalate": round(1 - accept, 3)},
            }
        },
        "usage": {"input_tokens": 434, "output_tokens": 36},
    }


def _classifier(handler, **kwargs):
    kwargs.setdefault("retry_backoff", 0)
    return HttpJevClassifier(api_key="k-123", transport=httpx.MockTransport(handler), **kwargs)


def _tasks(n):
    return [TASK.model_copy(update={"task": f"t{i}"}) for i in range(n)]


# --- contract ------------------------------------------------------------------- #


async def test_request_follows_the_official_systemone_contract():
    seen: list[httpx.Request] = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json=_ok())

    [verdict] = await _classifier(handler).classify([TASK])

    req = seen[0]
    assert req.method == "POST"
    assert str(req.url) == "https://api.typesafe.ai/v1/systemone"
    assert req.headers["Authorization"] == "Bearer k-123"
    body = json.loads(req.content)
    assert body["model"] == "jev-latest"
    assert body["state"]["task"] == "Payments gateway"
    assert body["state"]["reliability"] == 0.28
    assert "hours_range" not in body["state"]  # None fields are not sent
    assert body["questions"]["triage"]["type"] == "choice"
    assert body["questions"]["triage"]["instructions"]
    assert set(body["questions"]["triage"]["criteria"]) == {"accept", "escalate"}
    assert verdict.decision == "accept"
    assert verdict.probabilities["accept"] == 0.9
    assert verdict.model == "jev-1.14.0"  # the version that answered, not the alias
    assert verdict.usage == {"input_tokens": 434, "output_tokens": 36}
    assert verdict.error is None


def test_neighbors_are_sent_as_evidence_without_ids_and_capped():
    neighbors = [
        {"source_id": i, "budget_id": f"b{i}", "estimated_hours": 40 + i, "distance": 0.2 + i / 100}
        for i in range(8)
    ]
    body = build_request(TASK.model_copy(update={"neighbors": neighbors}), model="jev-latest")

    sent = body["state"]["neighbors"]
    assert len(sent) == MAX_NEIGHBORS
    assert sent[0] == {"estimated_hours": 40, "distance": 0.2}  # no corpus ids leave
    assert "neighbors" in body["questions"]["triage"]["instructions"]


async def test_one_request_per_task():
    calls: list[str] = []

    def handler(request):
        calls.append(json.loads(request.content)["state"]["task"])
        return httpx.Response(200, json=_ok(choice="escalate", accept=0.2))

    verdicts = await _classifier(handler).classify(_tasks(5))

    assert sorted(calls) == [f"t{i}" for i in range(5)]
    assert [v.task for v in verdicts] == [f"t{i}" for i in range(5)]
    assert all(v.decision == "escalate" for v in verdicts)


async def test_unexpected_shapes_escalate_with_error():
    noul = {"type": "noul", "noul": 0.9}
    bodies = [
        {"model": "jev-1.14.0", "answers": {}},  # no triage answer
        {"model": "jev-1.14.0", "answers": {"triage": noul}},  # wrong question type
        {"answers": {"triage": {"type": "choice", "choice": "maybe"}}},  # unknown choice
        {"unexpected": True},  # not a SystemOneResponse
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


# --- errors and retries ---------------------------------------------------------- #


async def test_a_bad_key_is_not_retried_and_keeps_typesafes_message():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(
            401,
            json={
                "detail": {
                    "error_type": "authentication_error",
                    "message": "Cannot authenticate with the server.",
                }
            },
        )

    [verdict] = await _classifier(handler, retries=3).classify([TASK])

    assert len(calls) == 1
    assert verdict.decision == "escalate"
    assert verdict.error == "HTTP 401 Unauthorized: Cannot authenticate with the server."


async def test_a_validation_error_is_not_retried():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(422, json={"detail": [{"msg": "Field required"}]})

    [verdict] = await _classifier(handler, retries=3).classify([TASK])

    assert len(calls) == 1
    assert verdict.error == "HTTP 422 Unprocessable Entity: Field required"


async def test_overloaded_and_rate_limited_are_retried_and_can_recover():
    responses = iter(
        [
            httpx.Response(529),
            httpx.Response(429, headers={"retry-after": "0"}),
            httpx.Response(200, json=_ok()),
        ]
    )

    [verdict] = await _classifier(lambda r: next(responses), retries=2).classify([TASK])

    assert verdict.decision == "accept"
    assert verdict.error is None


async def test_timeout_is_retried_then_escalates():
    calls = []

    def handler(request):
        calls.append(request)
        raise httpx.ReadTimeout("slow", request=request)

    [verdict] = await _classifier(handler, retries=1).classify([TASK])

    assert len(calls) == 2
    assert verdict.decision == "escalate"
    assert verdict.error == "slow"


def test_backoff_is_exponential_capped_and_honours_retry_after():
    jev = HttpJevClassifier(api_key="k", retry_backoff=0.5)
    request = httpx.Request("POST", "https://api.typesafe.ai/v1/systemone")

    def status_error(headers=None):
        response = httpx.Response(529, headers=headers or {}, request=request)
        return httpx.HTTPStatusError("x", request=request, response=response)

    assert [jev._delay(status_error(), a) for a in range(5)] == [0.5, 1.0, 2.0, 4.0, 5.0]
    assert jev._delay(status_error({"retry-after": "2"}), 0) == 2.0
    assert jev._delay(status_error({"retry-after": "120"}), 0) == 5.0  # capped


async def test_an_outage_opens_the_circuit_instead_of_calling_every_task():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(529)

    verdicts = await _classifier(handler, max_concurrency=2, retries=1).classify(_tasks(20))

    assert len(calls) <= 2 * (1 + 1)  # only the tasks already in flight, with their retry
    assert all(v.decision == "escalate" and v.error for v in verdicts)
    assert sum("circuit open" in v.error for v in verdicts) >= 18
    assert verdicts[0].error.startswith("HTTP 529")  # one readable line, no MDN hint


# --- labelled fallback ------------------------------------------------------------ #


def test_heuristic_is_deterministic_and_explainable():
    jev = HeuristicJevClassifier(min_reliability=0.15)
    assert jev.p_accept(TASK.model_copy(update={"reliability": 0.30})) == 0.875
    assert jev.p_accept(TASK.model_copy(update={"reliability": 0.20})) == 0.625
    ranged = TASK.model_copy(update={"reliability": 0.30, "hours_range": {"low": 1, "high": 9}})
    assert jev.p_accept(ranged) == 0.575  # a contradictory range counts against it


async def test_fallback_replaces_only_the_failed_verdicts_and_labels_them():
    def handler(request):
        task = json.loads(request.content)["state"]["task"]
        if task == "t0":
            return httpx.Response(200, json=_ok("escalate", 0.1))
        return httpx.Response(400)

    fallback = FallbackJevClassifier(_classifier(handler), HeuristicJevClassifier())
    real, simulated = await fallback.classify(_tasks(2))

    assert (real.source, real.decision, real.fallback_reason) == ("jev", "escalate", None)
    assert real.usage == {"input_tokens": 434, "output_tokens": 36}
    assert simulated.source == "simulated"
    assert simulated.error is None
    assert simulated.usage is None  # the stand-in makes no call and costs nothing
    assert simulated.fallback_reason == "HTTP 400 Bad Request"
    assert simulated.probabilities["accept"] == HeuristicJevClassifier().p_accept(TASK)
