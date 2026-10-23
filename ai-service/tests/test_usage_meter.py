"""The per-run token meter (``app/foundation/llm/usage_meter.py``)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from app.foundation.llm.usage_meter import (
    _estimate_cost,
    metered,
    record,
    usage_from_chat_completion,
    usage_from_responses_api,
)


def test_record_without_ledger_is_a_noop():
    assert record("x", "gpt-4o", input_tokens=10, output_tokens=10) is None


def test_ledger_accumulates_and_groups_with_list_prices():
    with metered() as ledger:
        record(
            "trinity_recovery", "gpt-5", input_tokens=1_000, output_tokens=500, reasoning_tokens=300
        )
        record("trinity_recovery", "gpt-5", input_tokens=2_000, output_tokens=100)
        record("ReliabilityReport", "gpt-4o", input_tokens=400, output_tokens=200)
        record("jev", "typesafe-ai/jev", cost_usd=0.01)

    total = ledger.totals()
    assert (total.calls, total.input_tokens, total.output_tokens) == (4, 3_400, 800)
    trinity = ledger.by_source()["trinity_recovery"]
    assert (trinity.calls, trinity.reasoning_tokens) == (2, 300)
    assert trinity.cost_usd == _estimate_cost("gpt-5", 3_000, 600)
    assert ledger.by_source()["jev"].cost_usd == 0.01  # flat fee, no tokens
    assert ledger.by_model()["jev"].calls == 1  # model prefix normalised
    assert total.cost_usd == round(
        _estimate_cost("gpt-5", 3_000, 600) + _estimate_cost("gpt-4o", 400, 200) + 0.01, 6
    )


def test_nested_ledgers_are_isolated():
    with metered() as outer:
        record("a", "gpt-4o", input_tokens=1)
        with metered() as inner:
            record("b", "gpt-4o", input_tokens=1)
        record("c", "gpt-4o", input_tokens=1)
    assert [r.source for r in outer.records] == ["a", "c"]
    assert [r.source for r in inner.records] == ["b"]


async def test_ledger_survives_to_thread_and_gather():
    def sync_call():
        record("thread", "gpt-4o", input_tokens=5)

    async def async_call(i):
        record(f"task{i}", "gpt-4o", input_tokens=1)

    with metered() as ledger:
        await asyncio.to_thread(sync_call)
        await asyncio.gather(*(async_call(i) for i in range(3)))
    assert sorted(r.source for r in ledger.records) == ["task0", "task1", "task2", "thread"]


def test_usage_extractors_are_defensive():
    assert usage_from_responses_api(SimpleNamespace()) == {}
    assert usage_from_chat_completion(None) == {}
    resp = SimpleNamespace(
        usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=50,
            output_tokens_details=SimpleNamespace(reasoning_tokens=30),
            input_tokens_details=SimpleNamespace(cached_tokens=20),
        )
    )
    assert usage_from_responses_api(resp) == {
        "input_tokens": 100,
        "output_tokens": 50,
        "reasoning_tokens": 30,
        "cached_input_tokens": 20,
    }
    completion = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3))
    assert usage_from_chat_completion(completion)["input_tokens"] == 7
