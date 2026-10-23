"""End-to-end multi-agent graph run, network-free (Session 13, live).

The compiled graph is driven with a ``MemorySaver`` checkpointer and fakes for every
network dependency — the ``LLMWrapper`` (scripted structured outputs), the S12
structure agent, the per-task ``estimate_one`` and the recovery loop. No network, no
API key, no database.

We assert the whole multi-agent contract: the run PAUSES at the two human gates and
resumes with ``Command(resume=...)``; the ``Command`` handovers wire classifier →
structure and recover → analysis; the ``Send`` fan-out accumulates one row per task
without duplicating; a flagged task triggers the agentic recovery; and the gate-2
route into the bonus proposal is honoured.
"""

from __future__ import annotations

import pytest
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from app.domain.graph.build import build_graph, fan_out_hours, route_after_gate2
from app.domain.graph.schemas import (
    CommercialProposal,
    ComplexityClassification,
    ReliabilityReport,
)
from app.domain.schemas.agent_trace import AgentTrace
from app.generation.agentic.agent_schemas import (
    AgentModuleNode,
    AgentStructure,
    AgentTaskDerivation,
    AgentTaskHoursRun,
    AgentTaskNode,
)
from app.generation.rag.schemas import TaskHoursEstimate

TRANSCRIPT = "A" * 200  # only its presence matters; the classifier is faked.


class _FakeWrapper:
    """Scripted ``complete_structured`` double keyed on the response model."""

    def __init__(self, *, complexity="high"):
        self._complexity = complexity
        self.calls: list[str] = []

    def complete_structured(self, *, system_prompt, user_message, response_model, **kwargs):
        self.calls.append(response_model.__name__)
        meta = {"model": "fake", "provider": "fake", "latency_ms": 1}
        if response_model is ComplexityClassification:
            return (
                ComplexityClassification(
                    complexity=self._complexity,
                    reformulated_transcript="Build a backend, a mobile app and an ERP integration.",
                    reasoning="several dispares components",
                ),
                meta,
            )
        if response_model is ReliabilityReport:
            return (
                ReliabilityReport(
                    overall_confidence="medium",
                    grounded_task_ratio=1.0,
                    weak_points=[],
                    summary="looks reasonable",
                ),
                meta,
            )
        if response_model is CommercialProposal:
            return (
                CommercialProposal(
                    title="RUTA",
                    executive_summary="A logistics platform.",
                    scope=["Backend", "Mobile"],
                    total_engineer_days=20,
                    body_markdown="# Proposal\n...",
                ),
                meta,
            )
        raise AssertionError(f"unexpected response_model {response_model!r}")


def _structure(modules):
    """Build a fake ``run_structure_agent`` returning the given module→task tree."""

    async def _run(brief, *, client, model, reasoning_effort="medium", persona=None):
        struct = AgentStructure(
            modules=[
                AgentModuleNode(
                    name=m,
                    tasks=[AgentTaskNode(name=t, description=f"{t} scope") for t in tasks],
                )
                for m, tasks in modules.items()
            ],
            confidence="high",
            reasoning="decomposed",
        )
        return struct, AgentTrace()

    return _run


def _estimate_one(hours_by_task, *, no_match=()):
    """Fake ``estimate_one``: grounded hours per task, or ``has_match=False``."""

    async def _one(module, name, description, *, top_k, distance_threshold, **kwargs):
        if name in no_match:
            return TaskHoursEstimate(module=module, task=name, has_match=False)
        return TaskHoursEstimate(
            module=module,
            task=name,
            estimated_hours=hours_by_task.get(name, 40),
            reliability=0.85,
            has_match=True,
            dispersion=0.1,
            neighbors=[],
        )

    return _one


def _wire(monkeypatch, *, wrapper, structure_fn, estimate_one_fn, recovery_fn=None):
    monkeypatch.setattr("app.dependencies.get_llm_wrapper", lambda: wrapper)
    # A non-None async client so the gpt-5 agents proceed (they are themselves faked).
    monkeypatch.setattr("app.dependencies.get_async_openai_client", lambda: object())
    monkeypatch.setattr("app.domain.graph.agents.structure.run_structure_agent", structure_fn)
    monkeypatch.setattr("app.domain.graph.agents.hours.estimate_one", estimate_one_fn)
    if recovery_fn is not None:
        monkeypatch.setattr(
            "app.domain.graph.agents.hours.run_task_hours_recovery_agent", recovery_fn
        )


CONFIG = {"configurable": {"thread_id": "t1"}}


async def _start(graph):
    return await graph.ainvoke({"transcript": TRANSCRIPT, "estimation_id": "t1"}, CONFIG)


@pytest.mark.asyncio
async def test_gate2_estimate_overrides_recompute_totals(monkeypatch):
    """At gate 2 the human edits per-task hours; the node recomputes the totals from
    the overridden modules so days/hours/ratio/confidence stay consistent."""
    _wire(
        monkeypatch,
        wrapper=_FakeWrapper(),
        structure_fn=_structure({"Backend": ["API", "Auth"]}),
        estimate_one_fn=_estimate_one({"API": 80, "Auth": 40}),
    )
    graph = build_graph(MemorySaver())
    await _start(graph)
    await graph.ainvoke(Command(resume={"approved": True}), CONFIG)  # → gate 2
    snap = await graph.aget_state(CONFIG)
    assert snap.values["estimate"]["total_engineer_hours"] == 120.0  # 80 + 40

    # Human doubles API (80 → 160) and validates with an override of the full tree.
    edited = [
        {"name": "Backend", "tasks": [
            {"name": "API", "estimated_hours": 160, "has_match": True},
            {"name": "Auth", "estimated_hours": 40, "has_match": True},
        ]}
    ]
    result = await graph.ainvoke(
        Command(resume={"validated": True, "estimate_overrides": {"modules": edited},
                        "want_proposal": False}),
        CONFIG,
    )
    assert result["status"] == "validated"
    assert result["estimate"]["total_engineer_hours"] == 200.0  # recomputed, not 120
    assert result["estimate"]["total_engineer_days"] == 25       # round(200/8)
    assert result["estimate"]["confidence"] == "high"


@pytest.mark.asyncio
async def test_full_flow_pauses_at_both_gates_and_completes(monkeypatch):
    wrapper = _FakeWrapper()
    _wire(
        monkeypatch,
        wrapper=wrapper,
        structure_fn=_structure({"Backend": ["API", "Auth"], "Mobile": ["App"]}),
        estimate_one_fn=_estimate_one({"API": 80, "Auth": 40, "App": 120}),
    )
    graph = build_graph(MemorySaver())

    # START → pauses at human gate 1 (structure review).
    await _start(graph)
    snap = await graph.aget_state(CONFIG)
    assert snap.next == ("human_gate_structure",)
    assert snap.interrupts[0].value["gate"] == "structure_review"
    assert snap.values["complexity"] == "high"  # classifier ran
    assert snap.values["structure"]["modules"]  # structure_agent ran (handover 1)

    # RESUME gate 1 → fan-out + recover + analysis → pauses at human gate 2.
    await graph.ainvoke(Command(resume={"approved": True}), CONFIG)
    snap = await graph.aget_state(CONFIG)
    assert snap.next == ("human_gate_analysis",)
    assert snap.interrupts[0].value["gate"] == "final_review"
    # Fan-out accumulated exactly one row per task (no duplication).
    assert len(snap.values["task_hours"]) == 3
    # Handover 2 built the estimate; analysis produced a report.
    assert snap.values["estimate"]["total_engineer_days"] == round((80 + 40 + 120) / 8)
    assert snap.values["analysis_report"]["overall_confidence"] == "medium"

    # RESUME gate 2 (validated + want proposal) → proposal_agent → END.
    result = await graph.ainvoke(
        Command(resume={"validated": True, "want_proposal": True}), CONFIG
    )
    snap = await graph.aget_state(CONFIG)
    assert snap.next == ()  # completed
    assert result["status"] == "validated"
    assert result["proposal"].startswith("# Proposal")
    # Three structured LLM calls: classifier, analysis, proposal.
    assert wrapper.calls == ["ComplexityClassification", "ReliabilityReport", "CommercialProposal"]


@pytest.mark.asyncio
async def test_full_flow_with_jev_triage_end_to_end(monkeypatch):
    """START → gate 1 → fan-out → Jev → Trinity → analysis → gate 2 → proposal → END.

    Four tasks hit the three bands: "API" is solid (nobody asked), "Auth" and "Cache"
    sit in the Jev band (Jev accepts "Auth", escalates "Cache"), "Legacy" has no match.
    Only "Cache" + "Legacy" may reach Trinity, and the Jev acceptance must survive the
    gate-2 resume all the way to the final estimate.
    """
    from app.foundation.llm.jev import JevVerdict

    reliability = {"API": 0.9, "Auth": 0.3, "Cache": 0.25}

    async def _one(module, name, description, *, top_k, distance_threshold, **kwargs):
        if name == "Legacy":
            return TaskHoursEstimate(module=module, task=name, has_match=False)
        return TaskHoursEstimate(
            module=module,
            task=name,
            estimated_hours=24,
            reliability=reliability[name],
            has_match=True,
            dispersion=0.1,
            neighbors=[],
        )

    class _Jev:
        seen: list[str] = []

        async def classify(self, tasks):
            self.seen.extend(t.task for t in tasks)
            return [
                JevVerdict(
                    module=t.module,
                    task=t.task,
                    decision="accept" if t.task == "Auth" else "escalate",
                    probabilities={"accept": 0.95 if t.task == "Auth" else 0.1},
                )
                for t in tasks
            ]

    trinity_calls: list[list[str]] = []

    async def _recovery(flagged, **kwargs):
        trinity_calls.append(sorted(f.task for f in flagged))
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

    jev = _Jev()
    monkeypatch.setattr("app.dependencies.get_jev_classifier", lambda: jev)
    _wire(
        monkeypatch,
        wrapper=_FakeWrapper(),
        structure_fn=_structure({"Backend": ["API", "Auth", "Cache", "Legacy"]}),
        estimate_one_fn=_one,
        recovery_fn=_recovery,
    )
    graph = build_graph(MemorySaver())

    await _start(graph)
    await graph.ainvoke(Command(resume={"approved": True}), CONFIG)
    snap = await graph.aget_state(CONFIG)
    assert snap.next == ("human_gate_analysis",)
    assert sorted(jev.seen) == ["Auth", "Cache"]  # only the Jev band was asked
    assert trinity_calls == [["Cache", "Legacy"]]  # Jev's accept never reached Trinity

    result = await graph.ainvoke(
        Command(resume={"validated": True, "want_proposal": True}), CONFIG
    )
    assert result["status"] == "validated"
    assert result["proposal"]
    tasks = {t["name"]: t for t in result["estimate"]["modules"][0]["tasks"]}
    assert tasks["Auth"]["accepted_by"] == "jev"
    assert tasks["Auth"]["estimated_hours"] == 24  # retrieved hours, untouched
    assert tasks["Cache"]["estimated_hours"] == 64  # recovered by Trinity
    assert tasks["Legacy"]["estimated_hours"] == 64
    assert tasks["API"]["accepted_by"] is None
    assert len(result["jev_decisions"]) == 2
    assert len(result["task_hours"]) == 4  # keyed reducer: no duplicates across resumes


@pytest.mark.asyncio
async def test_gate2_without_proposal_ends_without_proposal(monkeypatch):
    wrapper = _FakeWrapper()
    _wire(
        monkeypatch,
        wrapper=wrapper,
        structure_fn=_structure({"Backend": ["API"]}),
        estimate_one_fn=_estimate_one({"API": 40}),
    )
    graph = build_graph(MemorySaver())
    await _start(graph)
    await graph.ainvoke(Command(resume={"approved": True}), CONFIG)
    result = await graph.ainvoke(Command(resume={"validated": True, "want_proposal": False}), CONFIG)

    snap = await graph.aget_state(CONFIG)
    assert snap.next == ()
    assert result["status"] == "validated"
    assert result.get("proposal") is None
    assert "CommercialProposal" not in wrapper.calls


@pytest.mark.asyncio
async def test_flagged_task_triggers_agentic_recovery(monkeypatch):
    wrapper = _FakeWrapper()

    recovery_calls: list[int] = []

    async def _recovery(flagged, **kwargs):
        recovery_calls.append(len(flagged))
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

    _wire(
        monkeypatch,
        wrapper=wrapper,
        structure_fn=_structure({"Backend": ["API", "Legacy"]}),
        # "Legacy" has no deterministic match → gets flagged → recovered by the agent.
        estimate_one_fn=_estimate_one({"API": 40}, no_match={"Legacy"}),
        recovery_fn=_recovery,
    )
    graph = build_graph(MemorySaver())
    await _start(graph)
    await graph.ainvoke(Command(resume={"approved": True}), CONFIG)

    snap = await graph.aget_state(CONFIG)
    assert recovery_calls == [1]  # exactly one flagged task handed to recovery
    hours = {t["task"]: t["estimated_hours"] for t in snap.values["task_hours"]}
    assert hours == {"API": 40, "Legacy": 64}  # recovered hours merged in
    # Still exactly two rows — the keyed reducer replaced, never appended.
    assert len(snap.values["task_hours"]) == 2


@pytest.mark.asyncio
async def test_recovery_search_cut_comes_from_the_runtime_config(monkeypatch):
    """The recovery agent's distance cut is the knob that decides the unmatched count.

    The fan-out cut only decides who is HANDED to the agent; this one decides who the
    agent can rescue. With a large corpus a loose value here means no task is ever left
    without hours, which is exactly the failure this is guarding against — so it must be
    live-tunable, not frozen at process start.
    """
    captured: dict = {}

    async def _recovery(flagged, **kwargs):
        captured["backend"] = kwargs["retrieval_backend"]
        return AgentTaskHoursRun(
            derivations=[], trace=AgentTrace(), iterations=1, stopped_reason="completed"
        )

    def _fake_backend(top_k, distance_threshold):
        captured["top_k"] = top_k
        captured["distance_threshold"] = distance_threshold
        return object()

    class FakeRuntime:
        def effective_task_hours_top_k(self):
            return 5

        def effective_task_hours_distance_threshold(self):
            return 0.44

        def effective_agent_search_distance_threshold(self):
            return 0.45

    monkeypatch.setattr("app.dependencies.get_runtime_retrieval_config", lambda: FakeRuntime())
    monkeypatch.setattr("app.domain.graph.agents.hours.make_retrieval_backend", _fake_backend)
    _wire(
        monkeypatch,
        wrapper=_FakeWrapper(),
        structure_fn=_structure({"Backend": ["API", "Legacy"]}),
        estimate_one_fn=_estimate_one({"API": 40}, no_match={"Legacy"}),
        recovery_fn=_recovery,
    )
    graph = build_graph(MemorySaver())
    await _start(graph)
    await graph.ainvoke(Command(resume={"approved": True}), CONFIG)

    # The recovery cut is its own knob, NOT the fan-out's 0.44.
    assert captured["distance_threshold"] == 0.45


def test_fan_out_hours_emits_one_send_per_task():
    state = {
        "approved_modules": [
            {"name": "Backend", "tasks": [{"name": "API"}, {"name": "Auth"}]},
            {"name": "Mobile", "tasks": [{"name": "App"}]},
        ]
    }
    sends = fan_out_hours(state)
    assert [s.arg["task"] for s in sends] == ["API", "Auth", "App"]
    assert all(s.node == "estimate_task_hours" for s in sends)


def test_fan_out_hours_with_no_tasks_routes_to_join():
    assert fan_out_hours({"approved_modules": []}) == "recover_and_handover"


def test_fan_out_hours_resolves_the_retrieval_knobs_once(monkeypatch):
    """The runtime config is read in the edge, not in each of the N parallel branches.

    Resolving per branch would mean one Redis round-trip per task and would let an
    operator's mid-fan-out change give different tasks different thresholds.
    """
    calls = []

    class FakeRuntime:
        def effective_task_hours_top_k(self):
            calls.append("top_k")
            return 7

        def effective_task_hours_distance_threshold(self):
            calls.append("threshold")
            return 0.4

    monkeypatch.setattr("app.dependencies.get_runtime_retrieval_config", lambda: FakeRuntime())

    state = {
        "approved_modules": [
            {"name": "Backend", "tasks": [{"name": "API"}, {"name": "Auth"}]},
            {"name": "Mobile", "tasks": [{"name": "App"}]},
        ]
    }
    sends = fan_out_hours(state)

    assert calls == ["top_k", "threshold"]  # once each, for three tasks
    assert all(s.arg["top_k"] == 7 for s in sends)
    assert all(s.arg["distance_threshold"] == 0.4 for s in sends)


@pytest.mark.asyncio
async def test_estimate_task_hours_falls_back_to_settings(monkeypatch):
    """A Send built without the knobs (offline runner, older tests) still works."""
    from app.config import get_settings
    from app.domain.graph.agents import hours as hours_module

    seen = {}

    async def fake_estimate_one(module, task, description, *, top_k, distance_threshold, **kw):
        seen["top_k"] = top_k
        seen["distance_threshold"] = distance_threshold
        return TaskHoursEstimate(module=module, task=task, has_match=False)

    monkeypatch.setattr(hours_module, "estimate_one", fake_estimate_one)

    await hours_module.estimate_task_hours({"module": "Backend", "task": "API"})
    settings = get_settings()
    assert seen["top_k"] == settings.TASK_HOURS_TOP_K
    assert seen["distance_threshold"] == settings.TASK_HOURS_DISTANCE_THRESHOLD

    seen.clear()
    await hours_module.estimate_task_hours(
        {"module": "Backend", "task": "API", "top_k": 9, "distance_threshold": 0.33}
    )
    assert seen == {"top_k": 9, "distance_threshold": 0.33}


def test_route_after_gate2_honours_want_proposal():
    assert route_after_gate2({"gate2_decision": {"want_proposal": True}}) == "proposal"
    assert route_after_gate2({"gate2_decision": {"want_proposal": False}}) == "end"
    assert route_after_gate2({}) == "end"
