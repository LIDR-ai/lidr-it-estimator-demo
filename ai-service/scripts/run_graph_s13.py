#!/usr/bin/env python3
"""Session 13 (live) — run the MULTI-AGENT estimation graph end to end.

Drives the compiled LangGraph ``StateGraph`` (``app/domain/graph``) through the full
agent pipeline, AUTO-APPROVING the two human gates so a whole run completes without a
person in the loop:

    classifier_agent → structure_agent → [HUMAN GATE 1] → estimate_task_hours × N
      → recover_and_handover → analysis_agent → [HUMAN GATE 2] → proposal_agent

Each ``interrupt()`` pauses the graph; this script resumes it with a canned
``Command(resume=...)`` (accept the structure at gate 1; validate + ask for a proposal
at gate 2). In production the business backend supplies those decisions from the UI.

Persistence + observability:

* By default it opens the SAME Postgres the project uses (pgvector) as the
  checkpointer; pass ``--memory`` to use an in-process ``MemorySaver`` instead.
* Set ``LOGFIRE_TOKEN`` to export one span per agent/gate to Pydantic Logfire and get
  the trace link. With no token the spans still run locally (nothing exported).

Run variants::

    # Deliverable run: real retrieval + gpt-5 agents + Postgres checkpointer + Logfire
    docker compose exec ai-service python scripts/run_graph_s13.py \\
        --out exercises/session-13/example_run_complex.txt

    # Partial-offline smoke: no DB, canned per-task hours (still needs OPENAI_API_KEY
    # for the classifier / structure / analysis / proposal agents)
    uv run python scripts/run_graph_s13.py --memory --stub

    # Same, but with a mix of reliabilities so the Jev triage + Trinity both run
    uv run python scripts/run_graph_s13.py --memory --stub-mixed

``--stub`` swaps the real S9/S10 per-task retrieval (``estimate_one``) for a canned
offline estimate so the fan-out needs no database. The real path needs the
historical-task corpus ingested (``scripts/build_task_corpus.py --ingest``).
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from langgraph.types import Command  # noqa: E402

from app.domain.graph.build import build_graph  # noqa: E402
from app.domain.graph.observability import configure_logfire  # noqa: E402
from app.foundation.llm.usage_meter import metered  # noqa: E402

DEFAULT_TRANSCRIPT = REPO_ROOT / "exercises" / "session-12" / "sample_transcript_complex.txt"

# The canned resume decisions the runner feeds at each gate (auto-approval).
_GATE_DECISIONS = {
    "structure_review": {"approved": True},  # accept the structure as proposed
    "final_review": {"validated": True, "want_proposal": True},  # validate + draft proposal
}


def _install_stub_hours() -> None:
    """Monkeypatch the per-task hours retrieval with a canned offline estimate.

    Keeps the fan-out DB-free: every task gets a deterministic grounded estimate, so
    no task is flagged and the gpt-5 recovery loop never runs.
    """
    from app.domain.graph.agents import hours as hours_mod
    from app.generation.rag.schemas import TaskHoursEstimate

    async def _stub_estimate_one(module, name, description, *, top_k, distance_threshold, **kwargs):
        # Deterministic hours from the task name so a run is stable and offline.
        hours = 8 + (abs(hash((module, name))) % 10) * 8  # 8..80h
        return TaskHoursEstimate(
            module=module,
            task=name,
            estimated_hours=hours,
            reliability=0.82,
            has_match=True,
            dispersion=0.1,
            neighbors=[],
        )

    hours_mod.estimate_one = _stub_estimate_one  # type: ignore[assignment]


def _install_stub_mixed_hours() -> None:
    """Like ``_install_stub_hours`` but with a deterministic MIX of reliabilities.

    Cycles every task through four buckets so a demo always exercises both deciders
    of the recovery join: solid (accepted, no one asked), the Jev band (0.15–0.35 →
    Jev decides), very weak (straight to Trinity) and no match (straight to Trinity).
    The Jev band carries synthetic neighbours — half close and agreeing, half distant
    and disagreeing — so the real Jev has evidence to accept OR escalate.
    """
    from app.domain.graph.agents import hours as hours_mod
    from app.generation.rag.schemas import TaskHoursEstimate, TaskNeighbor

    def _neighbors(seed: int, hours: int) -> list[TaskNeighbor]:
        if (seed // 4) % 2 == 0:  # close analogs that agree
            spread = [(-2, 0.22), (0, 0.25), (2, 0.27)]
        else:  # distant analogs that disagree
            spread = [(-hours + 4, 0.55), (hours, 0.58)]
        return [
            TaskNeighbor(source_id=i, estimated_hours=max(1, hours + dh), distance=d)
            for i, (dh, d) in enumerate(spread)
        ]

    async def _stub_estimate_one(module, name, description, *, top_k, distance_threshold, **kwargs):
        seed = sum(ord(c) for c in f"{module}/{name}")
        hours = 8 + (seed % 10) * 8  # 8..80h
        bucket = seed % 4
        if bucket == 3:
            return TaskHoursEstimate(module=module, task=name, has_match=False)
        reliability = {0: 0.82, 1: 0.28, 2: 0.08}[bucket]
        return TaskHoursEstimate(
            module=module,
            task=name,
            estimated_hours=hours,
            reliability=reliability,
            has_match=True,
            dispersion=0.1,
            neighbors=_neighbors(seed, hours) if bucket == 1 else [],
        )

    hours_mod.estimate_one = _stub_estimate_one  # type: ignore[assignment]


def _render_usage(ledger) -> str:
    """Per-agent token usage of the whole run (list prices from MODEL_COSTS)."""
    lines = ["TOKEN USAGE (whole run · list prices)"]
    for source, rec in sorted(ledger.by_source().items(), key=lambda kv: -kv[1].cost_usd):
        lines.append(
            f"  {source:<26} {rec.calls:>3} calls  {rec.input_tokens:>9,} in  "
            f"{rec.output_tokens:>8,} out ({rec.reasoning_tokens:,} reasoning)  ${rec.cost_usd:.4f}"
        )
    total = ledger.totals()
    lines.append(
        f"  {'TOTAL':<26} {total.calls:>3} calls  {total.input_tokens:>9,} in  "
        f"{total.output_tokens:>8,} out  ${total.cost_usd:.4f}"
    )
    return "\n".join(lines)


def _render(state: dict) -> str:
    lines = [
        "=" * 78,
        "SESSION 13 — MULTI-AGENT ESTIMATION GRAPH RUN",
        "=" * 78,
        f"estimation_id : {state.get('estimation_id')}",
        f"complexity    : {state.get('complexity')}",
        f"status        : {state.get('status')}",
        "",
        "STRUCTURE (structure_agent → gate 1)",
    ]
    for module in (state.get("structure") or {}).get("modules") or []:
        lines.append(f"  ▸ {module.get('name')}")
        for task in module.get("tasks") or []:
            lines.append(f"      - {task.get('name')}")

    lines += ["", "ESTIMATE (hours agent → analysis → gate 2)"]
    estimate = state.get("estimate") or {}
    for module in estimate.get("modules") or []:
        lines.append(f"  ▸ {module.get('name')}")
        for task in module.get("tasks") or []:
            hours = task.get("estimated_hours")
            hours_text = f"{hours}h" if hours is not None else "NO MATCH"
            flag = "" if task.get("has_match") else "  ⚠ flagged"
            if task.get("accepted_by") == "jev":
                flag += "  ✓ Jev"
            elif task.get("accepted_by") == "jev_simulated":
                flag += "  ✓ Jev (simulated)"
            lines.append(f"      - {task.get('name')}: {hours_text}{flag}")
    lines.append(
        f"  TOTAL: {estimate.get('total_engineer_days')}d "
        f"({estimate.get('total_engineer_hours')}h, confidence {estimate.get('confidence')})"
    )

    decisions = state.get("jev_decisions") or []
    lines += ["", "JEV TRIAGE / TRINITY (recover_and_handover)"]
    if not decisions:
        lines.append("  (no task in the Jev band, or Jev disabled / no JEV_API_KEY)")
    for d in decisions:
        p_accept = (d.get("probabilities") or {}).get("accept")
        p_text = f"p(accept)={p_accept:.2f}" if p_accept is not None else "p(accept)=?"
        err = f"  ERROR: {d['error'].splitlines()[0]}" if d.get("error") else ""
        tag = " [simulated]" if d.get("source") == "simulated" else ""
        if d.get("model"):
            tag += f" [{d['model']}]"
        lines.append(f"  - {d.get('task')}: {d.get('decision')}{tag} ({p_text}){err}")
    accepted = sum(
        1
        for t in (state.get("task_hours") or [])
        if t.get("accepted_by") in ("jev", "jev_simulated")
    )
    simulated = [d for d in decisions if d.get("fallback_reason")]
    who = "Jev (SIMULATED stand-in)" if simulated else "Jev"
    lines.append(f"  {who} accepted {accepted}/{len(decisions)}; the rest went to Trinity")
    if simulated:
        lines.append(
            f"  ⚠ real Jev unavailable ({simulated[0].get('fallback_reason')}): "
            f"{len(simulated)} decision(s) came from the labelled heuristic (JEV_FALLBACK=simulated)"
        )
    trinity = state.get("trinity_run") or {}
    if trinity:
        skipped = trinity.get("doubtful", 0) - trinity.get("tasks", 0)
        lines.append(
            f"  Trinity: {trinity.get('tasks', 0)} task(s) handed over"
            f" ({skipped} more left flagged by the demo cap),"
            f" recovered {trinity.get('recovered', 0)},"
            f" {trinity.get('iterations', 0)}/{trinity.get('max_iterations')} iterations"
        )

    report = state.get("analysis_report") or {}
    lines += ["", "RELIABILITY REPORT (analysis_agent)"]
    lines.append(f"  overall_confidence : {report.get('overall_confidence')}")
    lines.append(f"  grounded_task_ratio: {report.get('grounded_task_ratio')}")
    for weak in report.get("weak_points") or []:
        lines.append(f"  - [{weak.get('severity')}] {weak.get('area')}: {weak.get('issue')}")
    if report.get("summary"):
        lines.append(f"  summary: {report.get('summary')}")

    proposal = state.get("proposal")
    if proposal:
        lines += ["", "COMMERCIAL PROPOSAL (proposal_agent — bonus)", proposal]

    errors = state.get("errors") or []
    if errors:
        lines += ["", "ERRORS / ISSUES"]
        lines += [f"  - {e}" for e in errors]
    return "\n".join(lines)


async def _run_to_completion(graph, transcript: str, estimation_id: str) -> dict:
    """Start the run and auto-approve every human gate until it completes."""
    config = {"configurable": {"thread_id": estimation_id}}
    await graph.ainvoke({"transcript": transcript, "estimation_id": estimation_id}, config)

    while True:
        snapshot = await graph.aget_state(config)
        if not snapshot.next:
            return snapshot.values  # completed
        interrupts = snapshot.interrupts or ()
        if not interrupts:
            # Paused but not on an interrupt (shouldn't happen) — nudge it forward.
            await graph.ainvoke(None, config)
            continue
        gate = (interrupts[0].value or {}).get("gate", "")
        decision = _GATE_DECISIONS.get(gate, {"approved": True, "validated": True})
        print(f"  ⏸ human gate '{gate}' → auto-resume {decision}")
        await graph.ainvoke(Command(resume=decision), config)


async def _main_async(args: argparse.Namespace) -> int:
    transcript_path = Path(args.transcript)
    if not transcript_path.is_file():
        print(f"ERROR: transcript not found: {transcript_path}", file=sys.stderr)
        return 1
    transcript = transcript_path.read_text(encoding="utf-8")
    estimation_id = args.estimation_id or f"s13-{transcript_path.stem}"

    configure_logfire()  # no FastAPI app in the CLI; spans + httpx only
    if args.stub_mixed:
        _install_stub_mixed_hours()
    elif args.stub:
        _install_stub_hours()

    from app.config import get_settings
    from app.dependencies import get_jev_classifier

    hours_mode = (
        "stub mixed (offline, exercises Jev + Trinity)"
        if args.stub_mixed
        else "stub (offline)"
        if args.stub
        else "real estimate_one()"
    )
    print(f"transcript    : {transcript_path}")
    print(f"checkpointer  : {'MemorySaver' if args.memory else 'AsyncPostgresSaver (pool)'}")
    print(f"per-task hours: {hours_mode}")
    jev_state = "ON" if get_jev_classifier() else "off (disabled or no JEV_API_KEY)"
    if get_jev_classifier() and get_settings().JEV_FALLBACK == "simulated":
        jev_state += " · fallback simulated (labelled)"
    print(f"jev triage    : {jev_state}")
    settings = get_settings()
    task_cap = settings.GRAPH_RECOVERY_MAX_TASKS or "no"
    print(
        f"trinity cap   : {settings.GRAPH_RECOVERY_MAX_ITERATIONS} iterations · max {task_cap} tasks"
    )
    print(f"estimation_id : {estimation_id}\n")

    with metered() as ledger:
        if args.memory:
            from langgraph.checkpoint.memory import MemorySaver

            graph = build_graph(MemorySaver())
            state = await _run_to_completion(graph, transcript, estimation_id)
        else:
            from app.domain.graph.checkpointer import open_checkpointer

            async with open_checkpointer() as checkpointer:
                graph = build_graph(checkpointer)
                state = await _run_to_completion(graph, transcript, estimation_id)

    rendered = _render(state) + "\n\n" + _render_usage(ledger)
    print("\n" + rendered)
    if args.out:
        Path(args.out).write_text(rendered + "\n", encoding="utf-8")
        print(f"\n(run written to {args.out})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Session 13 multi-agent estimation graph.")
    parser.add_argument(
        "--transcript",
        default=str(DEFAULT_TRANSCRIPT),
        help="Path to a meeting transcript .txt (default: the complex RUTA transcript).",
    )
    parser.add_argument("--estimation-id", help="thread_id for the checkpointer (default derived).")
    parser.add_argument(
        "--memory",
        action="store_true",
        help="Use an in-process MemorySaver instead of the Postgres checkpointer.",
    )
    parser.add_argument(
        "--stub",
        action="store_true",
        help="Use canned offline per-task hours (no database for the fan-out).",
    )
    parser.add_argument(
        "--stub-mixed",
        action="store_true",
        help="Like --stub but with a mix of reliabilities, so Jev and Trinity both run.",
    )
    parser.add_argument("--out", help="Write the rendered run to this file.")
    args = parser.parse_args()
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
