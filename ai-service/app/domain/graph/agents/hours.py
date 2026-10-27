"""``hours_retrieval_agent`` — per-task hours (fan-out) + agentic recovery (join).

Two nodes cooperate:

* ``estimate_task_hours`` is the FAN-OUT BRANCH. The graph dispatches one ``Send``
  per approved task, so many copies of this node run IN PARALLEL, each doing a single
  deterministic vector search over the historical-task corpus (reuses the Session 10
  ``estimate_one``). Each branch returns a one-element ``task_hours`` list; the keyed
  ``merge_task_hours`` reducer accumulates them idempotently.

* ``recover_and_handover`` is the JOIN, run once after every branch settles. It flags
  the doubtful tasks (no match / contradictory / low reliability), lets Jev (TypeSafe
  AI's classifier, when enabled) accept outright the ones whose retrieved hours are
  good enough, and, only if any remain, runs the Session 12 agentic recovery loop ONCE over the batch (reason→act→observe:
  reformulate the query, review analogs, decide) — this is where the agent REASONS
  instead of doing a single lookup. It merges the recovered hours, builds the
  structured estimate and performs the explicit HANDOVER to ``analysis_agent`` via
  ``Command(goto=..., update=...)``.

Keeping recovery in the join (one gpt-5 loop) rather than per-branch bounds gpt-5
concurrency and mirrors ``app/domain/agent_estimation.py::agent_estimate_task_hours``.
All reusable primitives are imported at module level so tests/offline runs can swap
them.
"""

from __future__ import annotations

import logfire
import structlog
from langgraph.types import Command

# Module-level imports = the monkeypatch seam for tests and the offline runner.
from app.generation.agentic.agent_loop import run_task_hours_recovery_agent
from app.generation.rag.agent_retrieval import make_retrieval_backend
from app.generation.rag.task_hours import distance_weighted_consensus, estimate_one

from app.config import get_settings
from app.domain.graph.agents._common import (
    JEV_ACCEPTED_BY,
    build_estimate,
    flag_reason,
    jev_accepts,
    jev_candidate,
    recovery_priority,
)
from app.domain.graph.personas import persona_for
from app.foundation.llm.jev import JevTaskInput
from app.foundation.llm.usage_meter import record
from app.generation.agentic.agent_schemas import AgentTaskRef

log = structlog.get_logger()


async def estimate_task_hours(state: dict) -> dict:
    """FAN-OUT BRANCH: derive hours for ONE task (the ``Send`` arg is the state).

    ``state`` here is the ``Send`` argument ``{"module", "task", "description"}`` —
    NOT the whole graph state. A deterministic, embeddings-only search; no LLM.

    ``top_k``/``distance_threshold`` ride in that payload when ``fan_out_hours``
    resolved them from the runtime config; the settings defaults are the fallback, so
    a ``Send`` built without them (the offline runner, the tests) behaves as before.
    """
    with logfire.span("node: estimate_task_hours"):
        settings = get_settings()
        module = state["module"]
        task = state["task"]
        description = state.get("description")
        top_k = state.get("top_k") or settings.TASK_HOURS_TOP_K
        distance_threshold = state.get("distance_threshold")
        if distance_threshold is None:
            distance_threshold = settings.TASK_HOURS_DISTANCE_THRESHOLD
        est = await estimate_one(
            module,
            task,
            description,
            top_k=top_k,
            distance_threshold=distance_threshold,
        )
        log.info(
            "estimate_task_hours_branch",
            module=module,
            task=task,
            has_match=est.has_match,
            hours=est.estimated_hours,
            top_k=top_k,
            distance_threshold=distance_threshold,
        )
        # One-element list — the keyed reducer accumulates it into task_hours.
        return {"task_hours": [est.model_dump()]}


async def recover_and_handover(state: dict) -> Command:
    """JOIN: agentic recovery of doubtful tasks, build the estimate, hand over.

    Reads the full ``task_hours`` accumulator, flags the doubtful rows, runs one
    recovery loop over them (if any + a client is available), merges the recovered
    hours, assembles the estimate and hands over to ``analysis_agent``.
    """
    with logfire.span("node: recover_and_handover"):
        settings = get_settings()
        approved = state.get("approved_modules") or []
        task_hours = list(state.get("task_hours") or [])
        by_key = {(t.get("module"), t.get("task")): t for t in task_hours}
        descriptions = {
            (m.get("name"), t.get("name")): t.get("description")
            for m in approved
            for t in (m.get("tasks") or [])
        }

        flagged: list[AgentTaskRef] = []
        for row in task_hours:
            reason = flag_reason(row)
            if reason is None:
                continue
            flagged.append(
                AgentTaskRef(
                    module=row.get("module"),
                    task=row.get("task"),
                    description=descriptions.get((row.get("module"), row.get("task"))),
                    reason=reason,
                )
            )

        from app.dependencies import (
            get_async_openai_client,
            get_jev_classifier,
            get_runtime_retrieval_config,
        )

        # JEV TRIAGE: before the (expensive) gpt-5 loop, Jev decides for each doubtful
        # task that already HAS hours whether to accept them as is. Only what it does
        # not accept — plus the rows it cannot judge (no match / too weak) — reaches
        # Trinity. Any Jev failure is a verdict of "escalate", so this never loses a task.
        jev_decisions: list[dict] = []
        errors: list[str] = []
        classifier = get_jev_classifier()
        candidate_keys = {
            (row.get("module"), row.get("task"))
            for row in task_hours
            if jev_candidate(row, settings.GRAPH_JEV_MIN_RELIABILITY)
        }
        if classifier is not None and candidate_keys:
            with logfire.span("jev: triage", candidates=len(candidate_keys)):
                verdicts = await classifier.classify(
                    [
                        JevTaskInput(
                            module=f.module,
                            task=f.task,
                            description=f.description,
                            estimated_hours=by_key[(f.module, f.task)].get("estimated_hours"),
                            reliability=by_key[(f.module, f.task)].get("reliability"),
                            hours_range=by_key[(f.module, f.task)].get("hours_range"),
                            flag_reason=f.reason,
                            neighbors=by_key[(f.module, f.task)].get("neighbors") or None,
                        )
                        for f in flagged
                        if (f.module, f.task) in candidate_keys
                    ]
                )
            jev_decisions = [v.model_dump() for v in verdicts]
            # Jev reports its token usage (billed per input token). Simulated verdicts
            # (JEV_FALLBACK) and failed calls are local and free.
            for v in verdicts:
                if v.source == "jev" and v.usage:
                    record(
                        "jev",
                        v.model or settings.JEV_MODEL,
                        input_tokens=v.usage.get("input_tokens", 0),
                        output_tokens=v.usage.get("output_tokens", 0),
                    )
            accepted_keys = {
                (v["module"], v["task"])
                for v in jev_decisions
                if (v["module"], v["task"]) in candidate_keys
                and jev_accepts(v, settings.GRAPH_JEV_ACCEPT_CONFIDENCE)
            }
            jev_errors = sum(1 for v in jev_decisions if v.get("error"))
            # Stand-in verdicts that replaced a FAILED real call (JEV_FALLBACK=simulated).
            simulated = [v for v in jev_decisions if v.get("fallback_reason")]
            if simulated:
                errors.append(
                    f"jev: unavailable ({simulated[0].get('fallback_reason')}) · "
                    f"{len(simulated)} decision(s) simulated"
                )
            elif jev_errors:
                errors.append(f"jev: {jev_errors} task(s) escalated after a Jev error")
            for key in accepted_keys:
                verdict = next(v for v in jev_decisions if (v["module"], v["task"]) == key)
                by_key[key] = {
                    **by_key[key],
                    # Labelled: a simulated verdict must never pass for a real Jev decision.
                    "accepted_by": (
                        "jev_simulated" if verdict.get("source") == "simulated" else "jev"
                    ),
                    "jev_confidence": (verdict.get("probabilities") or {}).get(
                        "accept", verdict.get("confidence")
                    ),
                    # The point consensus is kept; an accepted range is resolved to it.
                    "hours_range": None,
                }
            task_hours = list(by_key.values())
            flagged = [f for f in flagged if (f.module, f.task) not in accepted_keys]

        # Demo-sized Trinity: only the worst doubtful tasks — no analog first, then the
        # lowest reliability (stable sort, so ties keep their order). The rest are left
        # exactly as an unrecovered task is: flagged, with their retrieval row.
        doubtful_count = len(flagged)
        flagged = sorted(flagged, key=lambda f: recovery_priority(by_key[(f.module, f.task)]))
        if settings.GRAPH_RECOVERY_MAX_TASKS > 0:
            flagged = flagged[: settings.GRAPH_RECOVERY_MAX_TASKS]

        client = get_async_openai_client()
        merged = task_hours
        recovered_count = 0
        # THE knob that decides how many tasks finish with no hours at all. The fan-out cut
        # only decides who is handed to this agent; this one decides who it can rescue, and
        # with a large corpus a loose value here means nothing is ever left unmatched. Read
        # from the runtime config so it can be calibrated against the structure of the day
        # during the gate-1 pause — this join runs ONCE per run, so unlike the fan-out
        # branch there is no per-task Redis traffic and no risk of a mid-flight change
        # splitting the run across two thresholds.
        agent_distance_threshold = (
            get_runtime_retrieval_config().effective_agent_search_distance_threshold()
        )
        stopped_reason = None
        iterations = 0
        trinity_usage: dict[str, int] = {}
        if flagged and client is not None:
            log.info(
                "agentic_recovery_start",
                flagged=len(flagged),
                total=len(task_hours),
                distance_threshold=agent_distance_threshold,
            )
            run = await run_task_hours_recovery_agent(
                flagged,
                client=client,
                model=settings.AGENT_MODEL,
                reasoning_effort=settings.AGENT_REASONING_EFFORT,
                max_iterations=settings.GRAPH_RECOVERY_MAX_ITERATIONS,
                retrieval_backend=make_retrieval_backend(
                    settings.AGENT_SEARCH_TOP_K, agent_distance_threshold
                ),
                consensus_fn=distance_weighted_consensus,
                persona=persona_for("recover_and_handover", enabled=settings.GRAPH_PERSONAS_ENABLED),
            )
            recovered = {
                (d.module, d.task): d
                for d in run.derivations
                if d.has_match and d.estimated_hours is not None
            }
            recovered_count = len(recovered)
            merged_map = dict(by_key)
            for key, d in recovered.items():
                base = merged_map.get(key, {"module": key[0], "task": key[1]})
                merged_map[key] = {
                    **base,
                    "estimated_hours": d.estimated_hours,
                    "reliability": d.reliability,
                    "has_match": True,
                    # The agent recovered a point estimate; drop any stale range.
                    "hours_range": None,
                }
            merged = list(merged_map.values())
            stopped_reason = run.stopped_reason
            iterations = run.iterations
            trinity_usage = run.usage or {}

        estimate = build_estimate(approved, merged)
        log.info(
            "recover_and_handover_done",
            flagged=len(flagged),
            recovered=recovered_count,
            jev_candidates=len(jev_decisions),
            jev_accepted=sum(1 for t in merged if t.get("accepted_by") in JEV_ACCEPTED_BY),
            jev_simulated=sum(1 for v in jev_decisions if v.get("source") == "simulated"),
            jev_errors=sum(1 for v in jev_decisions if v.get("error")),
            trinity_tasks=len(flagged),
            trinity_skipped=doubtful_count - len(flagged),
            max_iterations=settings.GRAPH_RECOVERY_MAX_ITERATIONS,
            trinity_input_tokens=trinity_usage.get("input_tokens", 0),
            trinity_output_tokens=trinity_usage.get("output_tokens", 0),
            trinity_reasoning_tokens=trinity_usage.get("reasoning_tokens", 0),
            # Without these three, a recovery truncated by the iteration budget is
            # indistinguishable from one that genuinely found no analog — and the whole
            # point of the unmatched rows is that they mean "no precedent", not "ran out
            # of turns". ``max_iterations`` here invalidates the coverage number.
            stopped_reason=stopped_reason,
            iterations=iterations,
            distance_threshold=agent_distance_threshold,
            total_engineer_days=estimate.get("total_engineer_days"),
        )
        # Explicit handover: pass the estimate + merged hours to analysis_agent.
        update = {
            "estimate": estimate,
            "task_hours": merged,
            "jev_decisions": jev_decisions,
            "trinity_run": {
                "doubtful": doubtful_count,
                "tasks": len(flagged),
                "recovered": recovered_count,
                "iterations": iterations,
                "max_iterations": settings.GRAPH_RECOVERY_MAX_ITERATIONS,
            },
        }
        if errors:
            update["errors"] = errors
        return Command(goto="analysis_agent", update=update)
