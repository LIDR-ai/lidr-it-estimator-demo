"""``scripts/compare_jev_tokens.py`` end to end, network-free.

Fakes every LLM (classifier/analysis wrapper, gpt-5 structure, Trinity) and makes
Trinity's token usage proportional to the tasks it receives, so the WITH-Jev branch
must come out cheaper by exactly the tasks Jev accepted.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from app.domain.graph import build as build_mod
from app.domain.schemas.agent_trace import AgentTrace
from app.foundation.llm.usage_meter import _estimate_cost, record
from app.generation.agentic.agent_schemas import AgentTaskDerivation, AgentTaskHoursRun
from app.generation.rag.schemas import TaskHoursEstimate
from tests.domain.graph.test_graph import _FakeWrapper, _structure

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
TOKENS_PER_TASK = 10_000


def _load_script():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(
        "compare_jev_tokens", SCRIPTS / "compare_jev_tokens.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _MeteredWrapper(_FakeWrapper):
    def complete_structured(self, *, response_model, **kwargs):
        record(response_model.__name__, "gpt-4o", input_tokens=500, output_tokens=100)
        return super().complete_structured(response_model=response_model, **kwargs)


RELIABILITY = {"API": 0.9, "Auth": 0.3, "Cache": 0.25, "Search": 0.2}


async def _estimate_one(module, name, description, *, top_k, distance_threshold, **kwargs):
    if name == "Legacy":
        return TaskHoursEstimate(module=module, task=name, has_match=False)
    return TaskHoursEstimate(
        module=module,
        task=name,
        estimated_hours=24,
        reliability=RELIABILITY[name],
        has_match=True,
        dispersion=0.1,
        neighbors=[],
    )


async def _trinity(flagged, *, model, **kwargs):
    record("trinity_recovery", model, input_tokens=TOKENS_PER_TASK * len(flagged), output_tokens=0)
    return AgentTaskHoursRun(
        derivations=[
            AgentTaskDerivation(
                module=f.module, task=f.task, estimated_hours=64, reliability=0.7, has_match=True
            )
            for f in flagged
        ],
        trace=AgentTrace(),
        iterations=1,
        stopped_reason="completed",
    )


@pytest.mark.asyncio
async def test_with_jev_branch_saves_exactly_the_accepted_tasks(monkeypatch):
    script = _load_script()
    # Register the originals so the script's global swaps are undone after the test.
    for name in ("classifier_agent", "structure_agent"):
        monkeypatch.setattr(build_mod, name, getattr(build_mod, name))
    monkeypatch.setattr("app.dependencies.get_jev_classifier", lambda: None)

    monkeypatch.setattr("app.dependencies.get_llm_wrapper", lambda: _MeteredWrapper())
    monkeypatch.setattr("app.dependencies.get_async_openai_client", lambda: object())
    monkeypatch.setattr(
        "app.domain.graph.agents.structure.run_structure_agent",
        _structure({"Backend": ["API", "Auth", "Cache", "Search", "Legacy"]}),
    )
    monkeypatch.setattr("app.domain.graph.agents.hours.estimate_one", _estimate_one)
    monkeypatch.setattr("app.domain.graph.agents.hours.run_task_hours_recovery_agent", _trinity)

    common_values, common = await script.run_common_leg("A" * 200)
    assert common.by_source()["ComplexityClassification"].calls == 1
    script._replay_common(common_values)

    jev = script.SimulatedJev(1.0)  # accept every candidate
    off = await script.run_branch("off", "A" * 200, None)
    on = await script.run_branch("on", "A" * 200, jev)

    assert off["retrieved"] == on["retrieved"]  # same structure, same retrieval
    assert off["doubtful"] == 4  # Auth, Cache, Search (band) + Legacy (no match)
    assert (on["jev_candidates"], on["jev_accepted"]) == (3, 3)
    assert (off["trinity_tasks"], on["trinity_tasks"]) == (4, 1)  # only Legacy left

    trin_off = off["ledger"].by_source()["trinity_recovery"]
    trin_on = on["ledger"].by_source()["trinity_recovery"]
    assert trin_off.input_tokens - trin_on.input_tokens == 3 * TOKENS_PER_TASK
    # The upstream agents are replayed, not re-run: no classifier tokens in the branches.
    assert "ComplexityClassification" not in on["ledger"].by_source()
    assert on["ledger"].by_source()["jev"].calls == 3

    report = script.render(common, off, on, simulated=True, jev_fee=0.0)
    saved = _estimate_cost("gpt-5", 3 * TOKENS_PER_TASK, 0)
    assert f"${saved:.4f}" in report
    assert "SIMULATED" in report


def _fake_run(name, trinity_usd, *, accepted, calls=3):
    from app.foundation.llm.usage_meter import UsageLedger, UsageRecord

    ledger = UsageLedger(
        [UsageRecord("trinity_recovery", "gpt-5", cost_usd=trinity_usd, calls=calls)]
    )
    return {
        "name": name,
        "ledger": ledger,
        "retrieved": {},
        "tasks": 10,
        "doubtful": 5,
        "jev_candidates": 3,
        "jev_accepted": accepted,
        "jev_errors": 0,
        "trinity_tasks": 5 - accepted,
        "days": 10,
    }


def test_render_flags_noise_cap_and_overlapping_repeats():
    from app.foundation.llm.usage_meter import UsageLedger

    script = _load_script()
    # Jev accepted nothing → any delta is noise, and must be labelled as such.
    report = script.render(
        UsageLedger(),
        _fake_run("off", 0.07, accepted=0),
        _fake_run("on", 0.24, accepted=0),
        simulated=False,
        jev_fee=0.0,
        max_iterations=3,
    )
    assert "pure gpt-5 run-to-run noise" in report
    assert "hit the cap" in report  # 3 round-trips == cap 3

    # Repeats: means drive the saving; overlapping ranges are called out.
    offs = [_fake_run("off", c, accepted=2) for c in (0.50, 0.60)]
    ons = [_fake_run("on", c, accepted=2) for c in (0.40, 0.55)]
    report = script.render(UsageLedger(), offs, ons, simulated=True, jev_fee=0.0, max_iterations=10)
    assert "Repeats (2 runs per branch" in report
    assert "$0.0750" in report  # mean saving: 0.55 - 0.475
    assert "OVERLAP" in report
    assert "hit the cap" not in report
