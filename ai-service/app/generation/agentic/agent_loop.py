"""The hand-written agent loop (Session 12).

A senior developer reads this and recognises everything: a loop that calls an LLM
which *decides*, runs *tools*, and stops when it is done. No framework.

The agent drives the TWO phases of the estimation wizard, around the human review
gate — it does NOT run one autonomous shot over the raw transcript:

* ``run_structure_agent`` — phase 1. A single reasoned ``responses.parse`` that
  decomposes the brief into modules→tasks. NO tools, NO hours (grounding the
  structure in budgets impoverished the tree — the Session 10 decision). The human
  then reviews/edits that tree.
* ``run_task_hours_recovery_agent`` — phase 2. The reason→act→observe loop, run
  ONLY over the tasks the deterministic per-task pass could not ground. For each
  such task the agent reformulates the search, gathers analogs and derives hours
  with the same deterministic consensus. This is where the loop earns its keep.

DELIBERATE EXCEPTION to the repo convention: every other LLM call in this codebase
goes through ``LLMWrapper`` (LiteLLM + Instructor). This module talks to the raw
OpenAI **Responses API** (``client.responses.create`` / ``.parse``) on purpose —
the whole point of the exercise is to drive the reason→act→observe loop by hand so
each step is visible and captured in a trace. Do not "fix" this to use LLMWrapper.

Loop mechanics (stateful chaining, phase 2):

1. ``responses.create`` with the flagged tasks + the tool schemas. gpt-5 emits
   ``reasoning`` items and ``function_call`` items and then STOPS, waiting for us.
2. We read every ``function_call`` in ``response.output``, run the matching Python
   function, and send back one ``function_call_output`` per ``call_id``.
3. We re-call with ``previous_response_id`` and ONLY the new outputs — the server
   keeps the prior reasoning/function_call items and their ordering, which sidesteps
   the gpt-5 reasoning-item ordering pitfalls.
4. Repeat until a turn returns no ``function_call`` (natural stop) or we hit
   ``max_iterations`` (safeguard). The recovered hours come from the
   ``derive_task_hours`` tool outputs — there is no terminal ``responses.parse``.
"""

from __future__ import annotations

import json
from typing import Any

import structlog

from app.domain.schemas.agent_trace import AgentStep, AgentTrace
from app.foundation.llm.usage_meter import record, usage_from_responses_api
from app.generation.agentic.agent_schemas import (
    AgentStructure,
    AgentTaskDerivation,
    AgentTaskHoursRun,
    AgentTaskRef,
)
from app.generation.agentic.agent_tools import (
    HOURS_TOOL_SCHEMAS,
    ConsensusFn,
    RetrievalBackend,
    dispatch_tool,
)

log = structlog.get_logger()

STRUCTURE_SYSTEM_PROMPT = """\
You are a senior software-delivery architect acting as an estimation agent. You \
receive a structured project brief and must DECOMPOSE it into the functional \
MODULES and the concrete engineering TASKS needed to deliver it.

This is a STRUCTURE-ONLY step: you do NOT estimate hours and you have NO \
historical sources — rely on your engineering judgement about what the project \
entails. The hours are derived in a later step by searching a historical corpus \
per task, so a good, granular decomposition here is what matters.

- Organise the work into functional blocks (e.g. Authentication & Access, \
Payments & Billing, Core Domain, Data & Integrations, Frontend/UX, Infrastructure \
& DevOps, Security & Compliance, QA & Testing, Project Management). Use the \
modules that fit THIS project; add sector-specific ones when the brief calls for \
them; omit the ones that do not apply.
- Within each module, break the work into granular tasks with a short \
`description`. Be thorough — typically 5-9 modules with several tasks each — so a \
delivery team could plan from it.
- Set `confidence` from how well-specified the brief is, and explain your \
decomposition in `reasoning`. If the brief is too vague to scope responsibly, \
return an empty `modules` list and say so in `reasoning`.\
"""

HOURS_RECOVERY_SYSTEM_PROMPT = """\
You are an estimation agent recovering hours for tasks that the standard per-task \
search could NOT ground. Each task below came back with no usable historical \
analog (or a low-confidence / contradictory one). Your job is to try harder.

Method — for EACH task in the list:
1. Call `search_budgets` with a focused, task-specific query. If the first search \
finds nothing usable, REFORMULATE — reword it, use synonyms, describe the \
underlying capability instead of the product name, or relax/drop the sector \
filter — and search again. You decide how many attempts a task is worth.
2. When you have historical analogs, call `derive_task_hours` with the task and \
those neighbours (pass each neighbour's estimated_hours AND its distance exactly \
as search_budgets returned them). This computes the hours deterministically.
3. If after genuine effort you still find no analog, leave that task unresolved — \
do NOT invent hours. Move on to the next task.

When you have processed every task, call `validate_estimate` once over the tasks \
you managed to ground, address anything it flags, then stop calling tools.

You have exactly these tools: `search_budgets`, `derive_task_hours`, \
`validate_estimate`. Never invent hours: they must come from `derive_task_hours`.\
"""


def _extract_reasoning_summary(output: list[Any]) -> str | None:
    """Concatenate the reasoning-summary text emitted in one turn, if any.

    The Responses API surfaces a summary only when the call passes
    ``reasoning={"summary": "auto"}``; even then it may be empty for cheap efforts.
    """
    parts: list[str] = []
    for item in output:
        if getattr(item, "type", None) != "reasoning":
            continue
        for summary in getattr(item, "summary", None) or []:
            text = getattr(summary, "text", None)
            if text:
                parts.append(text)
    return " ".join(parts) if parts else None


def _meter(source: str, model: str, response: Any, acc: dict[str, int] | None = None) -> None:
    """Record one Responses API round-trip in the active usage ledger (+ a local sum)."""
    usage = usage_from_responses_api(response)
    if not usage:
        return
    record(source, model, **usage)
    if acc is not None:
        for key, value in usage.items():
            acc[key] = acc.get(key, 0) + value


def _function_calls(output: list[Any]) -> list[Any]:
    return [item for item in output if getattr(item, "type", None) == "function_call"]


async def run_structure_agent(
    brief: str,
    *,
    client: Any,
    model: str,
    reasoning_effort: str = "medium",
    persona: str | None = None,
) -> tuple[AgentStructure, AgentTrace]:
    """Phase 1 — the agent proposes the module→task structure (no tools, no hours).

    ``brief`` is the composed project brief (built by the conductor from the
    reformulated query). Returns the structure plus a thin one-step trace carrying
    the reasoning summary, so the wizard can show WHY the agent decomposed it so.
    """
    instructions = STRUCTURE_SYSTEM_PROMPT
    if persona and persona.strip():
        instructions = f"{STRUCTURE_SYSTEM_PROMPT}\n\n# Additional operator instructions\n{persona.strip()}"

    log.info("agent_structure_start", model=model, effort=reasoning_effort, persona=bool(persona))
    response = await client.responses.parse(
        model=model,
        instructions=instructions,
        input=[{"role": "user", "content": brief}],
        reasoning={"effort": reasoning_effort, "summary": "auto"},
        text_format=AgentStructure,
        store=True,
    )
    _meter("structure_agent", model, response)
    structure: AgentStructure = response.output_parsed
    reasoning_summary = _extract_reasoning_summary(getattr(response, "output", []) or [])

    task_count = sum(len(m.tasks) for m in structure.modules)
    trace = AgentTrace(
        steps=[
            AgentStep(
                step=1,
                reasoning_summary=reasoning_summary,
                tool="propose_structure",
                tool_args={"modules": len(structure.modules), "tasks": task_count},
                observation=(
                    f"decomposed into {len(structure.modules)} modules / {task_count} tasks "
                    f"(confidence={structure.confidence})"
                ),
            )
        ]
    )
    log.info(
        "agent_structure_done",
        modules=len(structure.modules),
        tasks=task_count,
        confidence=structure.confidence,
    )
    return structure, trace


async def run_task_hours_recovery_agent(
    flagged_tasks: list[AgentTaskRef],
    *,
    client: Any,
    model: str,
    reasoning_effort: str = "medium",
    max_iterations: int = 10,
    retrieval_backend: RetrievalBackend,
    consensus_fn: ConsensusFn,
    persona: str | None = None,
) -> AgentTaskHoursRun:
    """Phase 2 — the reason→act→observe loop over the flagged tasks only.

    ``retrieval_backend`` and ``consensus_fn`` are injected by the conductor (the
    real rag implementations, or stubs for offline runs). The recovered hours come
    from the ``derive_task_hours`` tool outputs the loop accumulates — there is no
    terminal ``responses.parse``: the numbers are deterministic, not model-authored.
    """
    if not flagged_tasks:
        return AgentTaskHoursRun(derivations=[], trace=AgentTrace(), iterations=0)

    instructions = HOURS_RECOVERY_SYSTEM_PROMPT
    if persona and persona.strip():
        instructions = (
            f"{HOURS_RECOVERY_SYSTEM_PROMPT}\n\n# Additional operator instructions\n{persona.strip()}"
        )

    task_lines = "\n".join(
        f"- module={t.module!r} task={t.task!r}"
        + (f" description={t.description!r}" if t.description else "")
        + f" (flagged: {t.reason})"
        for t in flagged_tasks
    )
    user_message = (
        "Recover hours for these tasks. For each, search historical analogs "
        "(reformulating as needed) and derive its hours; leave it unresolved if no "
        f"analog exists.\n\n{task_lines}"
    )

    trace = AgentTrace()
    derivations: dict[tuple[str, str], AgentTaskDerivation] = {}
    step_no = 0
    stopped_reason: str = "completed"

    log.info(
        "agent_hours_recovery_start",
        model=model,
        effort=reasoning_effort,
        flagged=len(flagged_tasks),
        persona=bool(persona),
    )
    response = await client.responses.create(
        model=model,
        instructions=instructions,
        input=[{"role": "user", "content": user_message}],
        tools=HOURS_TOOL_SCHEMAS,
        reasoning={"effort": reasoning_effort, "summary": "auto"},
        store=True,
    )
    usage: dict[str, int] = {}
    _meter("trinity_recovery", model, response, usage)
    iterations = 1

    while True:
        calls = _function_calls(response.output)
        if not calls:
            break
        if iterations >= max_iterations:
            stopped_reason = "max_iterations"
            log.warning("agent_max_iterations_reached", iterations=iterations)
            break

        # gpt-5 reasons ONCE per turn even when it emits several parallel tool
        # calls, so the summary belongs to the turn. Attach it to the first step
        # and mark the siblings as parallel calls of that same turn.
        reasoning_summary = _extract_reasoning_summary(response.output)
        first_step_in_turn = step_no + 1
        tool_outputs: list[dict[str, Any]] = []
        for call in calls:
            step_no += 1
            step_reasoning = (
                reasoning_summary
                if step_no == first_step_in_turn
                else f"(parallel tool call in the same turn as STEP {first_step_in_turn})"
            )
            name = getattr(call, "name", "unknown")
            try:
                raw_args = json.loads(call.arguments)
            except (json.JSONDecodeError, TypeError) as exc:
                raw_args = {}
                result: dict[str, Any] = {"error": f"arguments were not valid JSON: {exc}"}
            else:
                try:
                    result = await dispatch_tool(
                        name, raw_args, backend=retrieval_backend, consensus_fn=consensus_fn
                    )
                except Exception as exc:  # noqa: BLE001 — return the error so the model self-corrects.
                    log.warning("agent_tool_error", tool=name, error=str(exc)[:200])
                    result = {"error": f"{type(exc).__name__}: {exc}"}

            # Capture a successful derivation so the conductor can merge it back.
            if name == "derive_task_hours" and "error" not in result:
                key = (str(result.get("module", "")), str(result.get("task", "")))
                derivations[key] = AgentTaskDerivation(
                    module=key[0],
                    task=key[1],
                    estimated_hours=result.get("estimated_hours"),
                    reliability=result.get("reliability"),
                    has_match=bool(result.get("has_match", False)),
                )

            observation = result.get("summary") or result.get("error") or json.dumps(result)[:200]
            trace.steps.append(
                AgentStep(
                    step=step_no,
                    reasoning_summary=step_reasoning,
                    tool=name,
                    tool_args=raw_args,
                    observation=observation,
                )
            )
            tool_outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": call.call_id,
                    "output": json.dumps(result),
                }
            )

        response = await client.responses.create(
            model=model,
            previous_response_id=response.id,
            input=tool_outputs,
            tools=HOURS_TOOL_SCHEMAS,
            reasoning={"effort": reasoning_effort, "summary": "auto"},
            store=True,
        )
        _meter("trinity_recovery", model, response, usage)
        iterations += 1

    log.info(
        "agent_hours_recovery_done",
        iterations=iterations,
        steps=len(trace.steps),
        derived=len(derivations),
        stopped_reason=stopped_reason,
    )
    return AgentTaskHoursRun(
        derivations=list(derivations.values()),
        trace=trace,
        iterations=iterations,
        stopped_reason=stopped_reason,  # type: ignore[arg-type]
        usage=usage or None,
    )
