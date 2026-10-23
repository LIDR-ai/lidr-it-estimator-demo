"""Unit tests for the two-phase agent loop, driven by a fake AsyncOpenAI client.

No network and no API key: a scripted fake client returns canned Responses API
outputs so we can assert the loop's control flow — phase-1 structure parsing, and
phase-2 recovery (search→derive per task, call_id echoing, the max-iterations
safeguard, the derivation capture and the trace shape).
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from app.generation.agentic.agent_loop import (
    run_structure_agent,
    run_task_hours_recovery_agent,
)
from app.generation.agentic.agent_schemas import (
    AgentModuleNode,
    AgentStructure,
    AgentTaskNode,
    AgentTaskRef,
)
from app.generation.rag.task_hours import distance_weighted_consensus


def _function_call(name: str, call_id: str, arguments: dict):
    return SimpleNamespace(
        type="function_call", name=name, call_id=call_id, arguments=json.dumps(arguments)
    )


def _reasoning(text: str):
    return SimpleNamespace(type="reasoning", summary=[SimpleNamespace(text=text)])


def _message():
    return SimpleNamespace(type="message", role="assistant", content=[])


class _FakeResponses:
    """Scripted ``responses.create`` / ``responses.parse`` double."""

    def __init__(self, scripted_outputs: list[list] | None = None, parsed=None):
        self._scripted = scripted_outputs or []
        self._parsed = parsed
        self._i = 0
        self.create_calls: list[dict] = []
        self.parse_calls: list[dict] = []

    async def create(self, **kwargs):
        self.create_calls.append(kwargs)
        output = self._scripted[min(self._i, len(self._scripted) - 1)]
        self._i += 1
        return SimpleNamespace(output=output, id=f"resp_{self._i}")

    async def parse(self, **kwargs):
        self.parse_calls.append(kwargs)
        # The parse double echoes any reasoning items the caller wants surfaced.
        return SimpleNamespace(
            output_parsed=self._parsed,
            output=[_reasoning("Split the project into auth and reporting.")],
        )


class _FakeClient:
    def __init__(self, responses: _FakeResponses):
        self.responses = responses


async def _stub_backend(query: str, sectors: list[str] | None) -> list[dict]:
    return [
        {"id": 1, "estimated_hours": 100, "content_preview": "x", "distance": 0.1},
        {"id": 2, "estimated_hours": 140, "content_preview": "y", "distance": 0.3},
    ]


# --- phase 1: structure ---------------------------------------------------- #
async def test_structure_agent_returns_tree_and_thin_trace():
    parsed = AgentStructure(
        modules=[
            AgentModuleNode(
                name="Auth",
                description="Access",
                tasks=[AgentTaskNode(name="OAuth backend"), AgentTaskNode(name="RBAC")],
            )
        ],
        confidence="high",
        reasoning="Standard SaaS shape.",
    )
    fake = _FakeResponses(parsed=parsed)
    structure, trace = await run_structure_agent(
        "a project brief", client=_FakeClient(fake), model="gpt-5-mini"
    )
    assert len(structure.modules) == 1
    assert [t.name for t in structure.modules[0].tasks] == ["OAuth backend", "RBAC"]
    # No tools in phase 1: it goes through parse(), not create().
    assert fake.parse_calls and not fake.create_calls
    # Thin one-step trace carries the count and the reasoning summary.
    assert len(trace.steps) == 1
    assert trace.steps[0].tool == "propose_structure"
    assert "2 tasks" in trace.steps[0].observation
    assert trace.steps[0].reasoning_summary is not None


# --- phase 2: hours recovery ----------------------------------------------- #
def _recovery_script():
    """search auth → search reporting → derive auth → derive reporting → stop."""
    return [
        [
            _reasoning("Search analogs for each flagged task."),
            _function_call("search_budgets", "s1", {"query": "oauth backend", "filters": None}),
            _function_call("search_budgets", "s2", {"query": "reporting", "filters": None}),
        ],
        [
            _function_call(
                "derive_task_hours",
                "d1",
                {
                    "module": "Auth",
                    "task": "OAuth backend",
                    "neighbors": [
                        {"estimated_hours": 100, "distance": 0.1, "source_id": 1, "budget_id": None},
                        {"estimated_hours": 140, "distance": 0.3, "source_id": 2, "budget_id": None},
                    ],
                },
            ),
        ],
        [_message()],
    ]


def _flagged():
    return [
        AgentTaskRef(module="Auth", task="OAuth backend", reason="no analog"),
        AgentTaskRef(module="Reporting", task="Dashboards", reason="low reliability"),
    ]


async def test_recovery_runs_search_then_derive_and_captures_derivations():
    fake = _FakeResponses(_recovery_script())
    run = await run_task_hours_recovery_agent(
        _flagged(),
        client=_FakeClient(fake),
        model="gpt-5-mini",
        max_iterations=10,
        retrieval_backend=_stub_backend,
        consensus_fn=distance_weighted_consensus,
    )
    tools = [s.tool for s in run.trace.steps]
    assert tools.count("search_budgets") == 2
    assert "derive_task_hours" in tools
    assert run.stopped_reason == "completed"

    # The derive_task_hours output was captured as a merge-ready derivation.
    assert len(run.derivations) == 1
    d = run.derivations[0]
    assert (d.module, d.task) == ("Auth", "OAuth backend")
    assert d.has_match is True
    expected_hours, expected_reliability, _ = distance_weighted_consensus([(100, 0.1), (140, 0.3)])
    assert d.estimated_hours == expected_hours
    assert d.reliability == expected_reliability


async def test_recovery_empty_flagged_list_short_circuits():
    fake = _FakeResponses()
    run = await run_task_hours_recovery_agent(
        [],
        client=_FakeClient(fake),
        model="gpt-5-mini",
        retrieval_backend=_stub_backend,
        consensus_fn=distance_weighted_consensus,
    )
    assert run.iterations == 0
    assert run.derivations == []
    assert not fake.create_calls  # the loop never started


async def test_recovery_call_ids_are_echoed_back():
    fake = _FakeResponses(_recovery_script())
    await run_task_hours_recovery_agent(
        _flagged(),
        client=_FakeClient(fake),
        model="gpt-5-mini",
        retrieval_backend=_stub_backend,
        consensus_fn=distance_weighted_consensus,
    )
    second_call_input = fake.create_calls[1]["input"]
    echoed = {item["call_id"] for item in second_call_input}
    assert echoed == {"s1", "s2"}
    for item in second_call_input:
        assert item["type"] == "function_call_output"
        assert isinstance(item["output"], str)


async def test_recovery_max_iterations_safeguard_stops_loop():
    never_stops = [[_function_call("search_budgets", "x", {"query": "loop", "filters": None})]]
    fake = _FakeResponses(never_stops)
    run = await run_task_hours_recovery_agent(
        _flagged(),
        client=_FakeClient(fake),
        model="gpt-5-mini",
        max_iterations=3,
        retrieval_backend=_stub_backend,
        consensus_fn=distance_weighted_consensus,
    )
    assert run.stopped_reason == "max_iterations"
    assert run.iterations == 3


async def test_recovery_bad_tool_args_do_not_crash_the_loop():
    script = [
        [_function_call("derive_task_hours", "d1", {"module": "M"})],  # missing task/neighbors
        [_message()],
    ]
    fake = _FakeResponses(script)
    run = await run_task_hours_recovery_agent(
        _flagged(),
        client=_FakeClient(fake),
        model="gpt-5-mini",
        retrieval_backend=_stub_backend,
        consensus_fn=distance_weighted_consensus,
    )
    assert run.trace.steps[0].tool == "derive_task_hours"
    assert "error" in run.trace.steps[0].observation.lower()
    assert run.derivations == []  # a failed derive is not captured


async def test_recovery_sums_responses_api_usage_into_run_and_ledger():
    from app.foundation.llm.usage_meter import metered

    class _UsageResponses(_FakeResponses):
        async def create(self, **kwargs):
            resp = await super().create(**kwargs)
            resp.usage = SimpleNamespace(
                input_tokens=1_000,
                output_tokens=400,
                output_tokens_details=SimpleNamespace(reasoning_tokens=300),
                input_tokens_details=SimpleNamespace(cached_tokens=0),
            )
            return resp

    fake = _UsageResponses(_recovery_script())
    with metered() as ledger:
        run = await run_task_hours_recovery_agent(
            _flagged(),
            client=_FakeClient(fake),
            model="gpt-5",
            max_iterations=10,
            retrieval_backend=_stub_backend,
            consensus_fn=distance_weighted_consensus,
        )
    rounds = len(fake.create_calls)
    assert run.usage["input_tokens"] == 1_000 * rounds
    assert run.usage["reasoning_tokens"] == 300 * rounds
    trinity = ledger.by_source()["trinity_recovery"]
    assert (trinity.calls, trinity.output_tokens) == (rounds, 400 * rounds)


async def test_recovery_without_usage_reports_none():
    run = await run_task_hours_recovery_agent(
        _flagged(),
        client=_FakeClient(_FakeResponses(_recovery_script())),
        model="gpt-5-mini",
        max_iterations=10,
        retrieval_backend=_stub_backend,
        consensus_fn=distance_weighted_consensus,
    )
    assert run.usage is None
