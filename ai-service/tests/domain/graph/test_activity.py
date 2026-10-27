"""The live per-agent activity feed (Session 13 live) — network-free.

``describe_node`` maps a LangGraph ``astream`` chunk (node name + state update) to the
didactic line(s) the UI shows, reading the exact fields each agent already produces.
``GraphActivityLog`` accumulates those lines; here we exercise the in-process fallback
(``redis_client=None``) so no Redis is needed.
"""

from __future__ import annotations

from app.domain.graph.activity import GraphActivityLog, describe_node


def test_describe_classifier_reads_complexity():
    lines = describe_node("classifier_agent", {"complexity": "high"})
    assert lines == [{"node": "classifier", "label": "Classifier", "message": "Complejidad: high"}]


def test_describe_structure_counts_modules_and_tasks():
    update = {"structure": {"modules": [{"tasks": [1, 2]}, {"tasks": [3]}]}}
    (line,) = describe_node("structure_agent", update)
    assert line["node"] == "structure"
    assert line["message"] == "2 módulos · 3 tareas"


def test_describe_hours_fanout_one_line_per_task_with_and_without_match():
    # Fan-out yields a LIST of per-branch updates in one chunk.
    update = [
        {"task_hours": [{"task": "A", "has_match": True, "estimated_hours": 37}]},
        {"task_hours": [{"task": "B", "has_match": False, "estimated_hours": None}]},
    ]
    lines = describe_node("estimate_task_hours", update)
    assert [line["message"] for line in lines] == ["A: 37 h", "B: SIN ANÁLOGO"]
    assert all(line["node"] == "hours" for line in lines)


def test_describe_recover_reads_total_days():
    update = {"estimate": {"total_engineer_days": 292}, "task_hours": []}
    (line,) = describe_node("recover_and_handover", update)
    assert line["node"] == "recover"
    assert "292 jornadas" in line["message"]


def test_describe_recover_shows_the_demo_sized_trinity_run():
    update = {
        "estimate": {"total_engineer_days": 100},
        "trinity_run": {
            "doubtful": 32,
            "tasks": 10,
            "recovered": 4,
            "iterations": 3,
            "max_iterations": 3,
        },
    }
    (line,) = describe_node("recover_and_handover", update)
    assert "Trinity recuperó 4/10 (de 32 dudosas, 3/3 vueltas)" in line["message"]


def test_describe_analysis_reads_confidence_and_ratio():
    update = {"analysis_report": {"overall_confidence": "medium", "grounded_task_ratio": 0.59}}
    (line,) = describe_node("analysis_agent", update)
    assert line["node"] == "analysis"
    assert "medium" in line["message"] and "59%" in line["message"]


def test_describe_interrupt_and_unknown_never_raise():
    assert describe_node("__interrupt__", None)[0]["message"].startswith("⏸")
    # A shape it does not know about degrades to a generic line, no exception.
    assert describe_node("mystery_node", {"weird": 1})[0]["node"] == "mystery_node"


def test_activity_log_in_process_fallback_appends_reads_and_resets():
    log = GraphActivityLog(redis_client=None)
    log.append("run-1", node="classifier", label="Classifier", message="Complejidad: high")
    log.append("run-1", node="structure", label="Structure", message="11 módulos · 123 tareas")
    entries = log.read("run-1")
    assert [e["seq"] for e in entries] == [0, 1]
    assert entries[1]["node"] == "structure"
    # A separate run is isolated.
    assert log.read("run-2") == []
    # Reset clears the run (fresh START).
    log.reset("run-1")
    assert log.read("run-1") == []


def test_describe_recover_labels_a_simulated_jev():
    update = {
        "estimate": {"total_engineer_days": 10},
        "jev_decisions": [
            {"task": "a", "source": "simulated"},
            {"task": "b", "source": "simulated"},
        ],
        "task_hours": [{"task": "a", "accepted_by": "jev_simulated"}, {"task": "b"}],
    }
    (line,) = describe_node("recover_and_handover", update)
    assert "Jev (simulado) aceptó 1/2 · Jev no disponible" in line["message"]
