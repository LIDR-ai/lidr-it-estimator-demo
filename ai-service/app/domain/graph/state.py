"""The graph's shared, typed state (Level 1 → the live multi-agent flow).

A LangGraph ``StateGraph`` threads ONE state object through every node; each node
returns a *partial* update and LangGraph merges it in. For most fields the merge is
last-writer-wins (the update replaces the value). For a field annotated with a
**reducer** the merge is delegated to that reducer instead, so a node can return
only the *new* items and LangGraph combines them with what prior nodes accumulated.

Three accumulator fields use that pattern:

* ``budget_matches`` — legacy component-level matches (``operator.add`` concat).
* ``errors`` — any node appends a soft failure without clobbering earlier ones.
* ``task_hours`` — the per-task hours FAN-OUT accumulator. This is the one the live
  session cares about: ``estimate_task_hours`` runs once per approved task in
  PARALLEL (Send API) and each branch returns a one-element list. A plain
  ``operator.add`` would append duplicates if a resume re-entered the fan-out, so it
  uses the KEYED reducer ``merge_task_hours`` (dedupe by ``(module, task)``,
  last-write-wins) — idempotent across resumes.

The state now carries the whole multi-agent flow, not just the final estimate:
``complexity`` + ``reformulated_transcript`` (classifier), ``structure`` +
``approved_modules`` (structure_agent + human gate 1), ``task_hours`` + ``estimate``
(hours agent), ``analysis_report`` (analysis_agent), ``gate*_decision`` (the human
resume payloads, kept for audit) and ``proposal`` (proposal_agent, bonus).
"""

from __future__ import annotations

import operator
from typing import Annotated, Optional

# Pydantic (used by the response model that embeds Component/BudgetMatch) requires
# typing_extensions.TypedDict on Python < 3.12; LangGraph accepts it too.
from typing_extensions import TypedDict


class Component(TypedDict):
    """One functional component the project decomposes into."""

    name: str
    category: str


class BudgetMatch(TypedDict):
    """A historical reference budget retrieved for a component.

    ``amount`` carries the matched historical item's recorded engineer-hours (the
    grounding number the estimate is built from); ``distance`` is its cosine
    distance from the query (lower = closer).
    """

    component: str
    reference_budget_id: Optional[str]
    amount: float
    distance: float


def merge_task_hours(existing: list[dict] | None, new: list[dict] | None) -> list[dict]:
    """Reducer for the per-task hours fan-out accumulator.

    Keyed by ``(module, task)``, last-write-wins. Unlike ``operator.add`` this is
    IDEMPOTENT: if a resume ever re-runs a fan-out branch (or the recovery join
    rewrites a task's hours), the task is REPLACED, not appended twice. That is the
    concrete fix for the "reducers duplicate on resume" pitfall the live session
    reproduces — the accumulator can be re-entered without growing spurious rows.
    """
    by_key: dict[tuple[str, str], dict] = {
        (t.get("module"), t.get("task")): t for t in (existing or [])
    }
    for t in new or []:
        by_key[(t.get("module"), t.get("task"))] = t
    return list(by_key.values())


class EstimationState(TypedDict, total=False):
    """The state threaded through the graph.

    ``total=False`` so a node may return a partial dict without every key present;
    the initial invoke only supplies ``transcript`` (+ ``estimation_id``).
    """

    transcript: str
    estimation_id: str

    # --- classifier_agent -------------------------------------------------- #
    complexity: Optional[str]  # "low" | "medium" | "high"
    reformulated_transcript: Optional[str]  # cleaned brief the rest of the flow reads

    # --- legacy component pipeline (kept for the pre-exercise nodes/tests) -- #
    requirements: list[str]
    components: list[Component]
    # Accumulator: grows as each component is searched (reducer = list concat).
    budget_matches: Annotated[list[BudgetMatch], operator.add]

    # --- structure_agent + human gate 1 ------------------------------------ #
    structure: Optional[dict]  # AgentStructure.model_dump() — modules → tasks, no hours
    approved_modules: Optional[list[dict]]  # TaskHoursModuleInput-shaped, after the gate
    gate1_decision: Optional[dict]  # the human resume payload (audit)

    # --- hours_retrieval_agent (fan-out accumulator + merged estimate) ------ #
    # Keyed reducer: idempotent across a resume that re-enters the fan-out.
    task_hours: Annotated[list[dict], merge_task_hours]
    estimate: Optional[dict]
    # Jev triage verdicts from the recovery join (audit; written once, last-write-wins).
    jev_decisions: Optional[list[dict]]
    # Trinity's demo-sized run: doubtful / handed over / recovered / iterations (+ cap).
    trinity_run: Optional[dict]

    # --- analysis_agent + human gate 2 ------------------------------------- #
    analysis_report: Optional[dict]  # ReliabilityReport.model_dump()
    gate2_decision: Optional[dict]
    # Money, computed by the BUSINESS BACKEND and handed over at gate 2 (the first moment
    # the final hours exist). The graph never derives it — it only carries it so the
    # proposal agent can quote it. Shape: currency / rate_eur_per_hour / contingency_pct /
    # base_eur / contingency_eur / total_eur.
    pricing: Optional[dict]

    # --- proposal_agent (bonus) -------------------------------------------- #
    proposal: Optional[str]

    status: Optional[str]  # "validated" | "needs_review"
    # Accumulator: soft failures appended by any node, never clobbered.
    errors: Annotated[list[str], operator.add]
