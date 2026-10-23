"""Jev triage inside the recovery join — network-free.

Jev sits between the per-task retrieval and Trinity's gpt-5 recovery loop: it may
accept outright a doubtful task that already HAS hours; whatever it does not accept
(plus the rows it cannot judge) still goes to recovery. These tests drive the
``recover_and_handover`` node directly with a fake classifier and a fake recovery.
"""

from __future__ import annotations

import pytest

from app.config import get_settings
from app.domain.graph.agents._common import jev_accepts, jev_candidate
from app.domain.graph.agents.hours import recover_and_handover
from app.domain.schemas.agent_trace import AgentTrace
from app.foundation.llm.jev import JevVerdict
from app.generation.agentic.agent_schemas import AgentTaskDerivation, AgentTaskHoursRun

MIN = 0.15


def _row(task, *, hours=40.0, reliability=0.85, has_match=True, hours_range=None):
    return {
        "module": "Backend",
        "task": task,
        "estimated_hours": hours if has_match else None,
        "reliability": reliability if has_match else None,
        "has_match": has_match,
        "hours_range": hours_range,
    }


# --- pure rules ------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        (_row("ok", reliability=0.85), False),  # not doubtful: accepted without Jev
        (_row("band", reliability=0.25), True),  # doubtful, inside the Jev band
        (_row("edge", reliability=MIN), True),  # band is inclusive at the floor
        (_row("weak", reliability=0.10), False),  # too weak: straight to Trinity
        (_row("none", has_match=False), False),  # no hours to accept
        (_row("range", hours_range={"low": 20, "high": 80, "reason": "x"}), True),
    ],
)
def test_jev_candidate_bands(row, expected):
    assert jev_candidate(row, MIN) is expected


def test_jev_accepts_needs_our_threshold_not_just_the_choice():
    assert jev_accepts({"decision": "accept", "probabilities": {"accept": 0.9}}, 0.7)
    assert not jev_accepts({"decision": "accept", "probabilities": {"accept": 0.6}}, 0.7)
    assert jev_accepts({"decision": "accept", "confidence": 0.8}, 0.7)  # fallback
    assert not jev_accepts({"decision": "accept"}, 0.7)  # no score → no trust
    assert not jev_accepts({"decision": "escalate", "probabilities": {"accept": 0.9}}, 0.7)
    assert not jev_accepts(
        {"decision": "accept", "probabilities": {"accept": 0.9}, "error": "boom"}, 0.7
    )


# --- the node --------------------------------------------------------------- #


class _FakeJev:
    def __init__(self, decisions: dict[str, tuple[str, float]] | None = None, *, fail=False):
        self.decisions = decisions or {}
        self.fail = fail
        self.seen: list[str] = []

    async def classify(self, tasks):
        self.seen.extend(t.task for t in tasks)
        out = []
        for t in tasks:
            if self.fail:
                out.append(
                    JevVerdict(module=t.module, task=t.task, decision="escalate", error="down")
                )
                continue
            decision, accept = self.decisions.get(t.task, ("escalate", 0.1))
            out.append(
                JevVerdict(
                    module=t.module,
                    task=t.task,
                    decision=decision,
                    probabilities={"accept": accept, "escalate": 1 - accept},
                )
            )
        return out


class _Runtime:
    def effective_agent_search_distance_threshold(self):
        return 0.6


def _state(rows):
    tasks = [{"name": r["task"], "description": f"{r['task']} scope"} for r in rows]
    return {"approved_modules": [{"name": "Backend", "tasks": tasks}], "task_hours": rows}


@pytest.fixture
def recovery(monkeypatch):
    calls: list[list[str]] = []

    async def _recovery(flagged, **kwargs):
        calls.append([f.task for f in flagged])
        return AgentTaskHoursRun(
            derivations=[
                AgentTaskDerivation(
                    module=f.module,
                    task=f.task,
                    estimated_hours=64,
                    reliability=0.7,
                    has_match=True,
                )
                for f in flagged
            ],
            trace=AgentTrace(),
            iterations=1,
            stopped_reason="completed",
        )

    monkeypatch.setattr("app.domain.graph.agents.hours.run_task_hours_recovery_agent", _recovery)
    monkeypatch.setattr("app.domain.graph.agents.hours.make_retrieval_backend", lambda *a: None)
    monkeypatch.setattr("app.dependencies.get_async_openai_client", lambda: object())
    monkeypatch.setattr("app.dependencies.get_runtime_retrieval_config", lambda: _Runtime())
    return calls


def _with_jev(monkeypatch, jev):
    monkeypatch.setattr("app.dependencies.get_jev_classifier", lambda: jev)


ROWS = [
    _row("solid", reliability=0.9),  # not doubtful
    _row("good-enough", hours=24.0, reliability=0.3),  # Jev accepts
    _row("shaky", reliability=0.2),  # Jev escalates
    _row("weak", reliability=0.05),  # below the band → Trinity directly
    _row("unmatched", has_match=False),  # no hours → Trinity directly
]


async def test_jev_accepts_some_and_only_the_rest_reach_trinity(monkeypatch, recovery):
    jev = _FakeJev({"good-enough": ("accept", 0.92), "shaky": ("escalate", 0.2)})
    _with_jev(monkeypatch, jev)

    cmd = await recover_and_handover(_state(ROWS))

    assert sorted(jev.seen) == ["good-enough", "shaky"]  # only the band is asked
    assert sorted(recovery[0]) == ["shaky", "unmatched", "weak"]
    rows = {t["task"]: t for t in cmd.update["task_hours"]}
    assert rows["good-enough"]["accepted_by"] == "jev"
    assert rows["good-enough"]["estimated_hours"] == 24.0  # retrieved hours kept
    assert rows["good-enough"]["jev_confidence"] == 0.92
    assert rows["shaky"]["estimated_hours"] == 64  # recovered by Trinity
    assert len(cmd.update["jev_decisions"]) == 2
    tasks = {t["name"]: t for t in cmd.update["estimate"]["modules"][0]["tasks"]}
    assert tasks["good-enough"]["accepted_by"] == "jev"
    assert tasks["shaky"]["accepted_by"] is None
    assert "errors" not in cmd.update


async def test_low_accept_probability_is_escalated(monkeypatch, recovery):
    threshold = get_settings().GRAPH_JEV_ACCEPT_CONFIDENCE
    _with_jev(monkeypatch, _FakeJev({"good-enough": ("accept", threshold - 0.05)}))

    await recover_and_handover(_state(ROWS))

    assert "good-enough" in recovery[0]


async def test_all_accepted_skips_trinity(monkeypatch, recovery):
    rows = [_row("a", reliability=0.3), _row("b", reliability=0.25)]
    _with_jev(monkeypatch, _FakeJev({"a": ("accept", 0.95), "b": ("accept", 0.8)}))

    cmd = await recover_and_handover(_state(rows))

    assert recovery == []  # nothing left for the gpt-5 loop
    assert all(t["accepted_by"] == "jev" for t in cmd.update["task_hours"])


async def test_jev_failure_escalates_everything_and_records_an_error(monkeypatch, recovery):
    _with_jev(monkeypatch, _FakeJev(fail=True))

    cmd = await recover_and_handover(_state(ROWS))

    assert sorted(recovery[0]) == ["good-enough", "shaky", "unmatched", "weak"]
    assert cmd.update["errors"] == ["jev: 2 task(s) escalated after a Jev error"]


async def test_jev_disabled_is_the_pre_jev_behaviour(monkeypatch, recovery):
    _with_jev(monkeypatch, None)

    cmd = await recover_and_handover(_state(ROWS))

    assert sorted(recovery[0]) == ["good-enough", "shaky", "unmatched", "weak"]
    assert cmd.update["jev_decisions"] == []
    assert not any(t.get("accepted_by") for t in cmd.update["task_hours"])
