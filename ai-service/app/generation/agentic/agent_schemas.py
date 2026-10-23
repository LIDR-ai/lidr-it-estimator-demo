"""Schemas for the Session 12 hand-written agent.

The agent no longer runs one autonomous shot over a transcript. It now drives
the TWO phases of the existing estimation wizard, so the schemas split by phase:

* **Tool argument models** (``SearchBudgetsArgs`` / ``DeriveTaskHoursArgs`` /
  ``ValidateEstimateArgs``) — the loop validates every
  ``json.loads(function_call.arguments)`` into one of these BEFORE dispatch, so a
  malformed / hallucinated argument becomes a returned error string the model can
  self-correct from, never an exception that kills the loop.
* **Phase-1 structure** (``AgentStructure``) — the module→task tree the agent
  proposes as a free decomposition of the brief (no tools, no hours). The
  conductor maps it onto the RAG ``Estimate`` contract the wizard already renders.
* **Phase-2 hours recovery** (``AgentTaskRef`` / ``AgentTaskDerivation`` /
  ``AgentTaskHoursRun``) — the input flagged tasks and the derivations the
  recovery loop produces, one per task it managed to ground.

The trace models (``AgentStep`` / ``AgentTrace``) live in
``app.domain.schemas.agent_trace`` — a shared audit contract the RAG response
schemas also carry. They are re-exported here for backward-compatible imports.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.domain.schemas.agent_trace import AgentStep, AgentTrace  # re-export

__all__ = [
    "AgentStep",
    "AgentTrace",
    "Confidence",
    "SearchBudgetsFilters",
    "SearchBudgetsArgs",
    "DeriveTaskHoursNeighbor",
    "DeriveTaskHoursArgs",
    "ValidateComponentInput",
    "ValidateEstimateArgs",
    "AgentTaskNode",
    "AgentModuleNode",
    "AgentStructure",
    "AgentTaskRef",
    "AgentTaskDerivation",
    "AgentTaskHoursRun",
]

Confidence = Literal["low", "medium", "high"]


# --------------------------------------------------------------------------- #
# Tool argument models                                                        #
# --------------------------------------------------------------------------- #
class SearchBudgetsFilters(BaseModel):
    """Optional structural filters for a budget search.

    Mirrors the nullable ``filters`` object in the tool schema. Both fields are
    optional; ``None`` means "do not constrain on this axis".
    """

    sectors: list[str] | None = Field(
        default=None,
        description="Restrict to these client sectors (e.g. ['logistics', 'industrial']).",
    )
    component_type: str | None = Field(
        default=None,
        description="Free-text hint about the kind of component (e.g. 'mobile app').",
    )


class SearchBudgetsArgs(BaseModel):
    """Validated arguments for the ``search_budgets`` tool."""

    query: str = Field(min_length=1)
    filters: SearchBudgetsFilters | None = None


class DeriveTaskHoursNeighbor(BaseModel):
    """One historical analog the agent found via ``search_budgets``.

    The agent copies these straight from a ``search_budgets`` result item — it
    does NOT invent them. ``distance`` is what makes the consensus distance-
    weighted, so it is required.
    """

    estimated_hours: int = Field(ge=0)
    distance: float = Field(ge=0.0)
    source_id: int | None = None
    budget_id: str | None = None


class DeriveTaskHoursArgs(BaseModel):
    """Validated arguments for the ``derive_task_hours`` tool."""

    module: str = Field(min_length=1)
    task: str = Field(min_length=1)
    neighbors: list[DeriveTaskHoursNeighbor] = Field(
        description="Historical analogs (from search_budgets) whose hours anchor this task."
    )


class ValidateComponentInput(BaseModel):
    """One line of the estimate to validate."""

    name: str = Field(min_length=1)
    estimated_hours: float
    reference_amounts: list[float] = Field(default_factory=list)


class ValidateEstimateArgs(BaseModel):
    """Validated arguments for the ``validate_estimate`` tool."""

    components: list[ValidateComponentInput]
    total_hours: float


# --------------------------------------------------------------------------- #
# Phase 1 — structure proposal (no tools, no hours)                           #
# --------------------------------------------------------------------------- #
class AgentTaskNode(BaseModel):
    """One task the agent proposes inside a module (structure only, no hours)."""

    name: str = Field(min_length=1)
    description: str | None = Field(default=None, description="One-line scope of the task.")


class AgentModuleNode(BaseModel):
    """One functional module the agent proposes, decomposed into tasks."""

    name: str = Field(min_length=1)
    description: str | None = Field(default=None, description="What this module covers.")
    tasks: list[AgentTaskNode] = Field(default_factory=list)


class AgentStructure(BaseModel):
    """The agent's phase-1 output: the module→task tree, no hours, no sources.

    Deliberately rag-free (no ``SourceReference`` / ``engineer_days``): the
    conductor maps it onto the heavy RAG ``Estimate`` the wizard renders, filling
    ``engineer_days=None`` / ``grounded=False`` so the per-task hours step derives
    the numbers afterwards.
    """

    modules: list[AgentModuleNode] = Field(default_factory=list)
    confidence: Confidence
    reasoning: str = Field(description="How the decomposition was reasoned.")


# --------------------------------------------------------------------------- #
# Phase 2 — hours recovery (loop over the flagged tasks only)                  #
# --------------------------------------------------------------------------- #
class AgentTaskRef(BaseModel):
    """One approved task the deterministic pass could not ground, handed to the
    recovery agent with the reason it was flagged."""

    module: str
    task: str
    description: str | None = None
    reason: str = Field(description="Why the deterministic pass flagged this task.")


class AgentTaskDerivation(BaseModel):
    """Hours the recovery agent derived for one flagged task.

    Mirrors the fields the conductor merges back onto the deterministic
    ``TaskHoursEstimate``. ``has_match=False`` means the agent searched but still
    found nothing usable — the row stays red.
    """

    module: str
    task: str
    estimated_hours: int | None = Field(default=None, ge=0)
    reliability: float | None = Field(default=None, ge=0.0, le=1.0)
    has_match: bool = False


class AgentTaskHoursRun(BaseModel):
    """Everything the phase-2 recovery loop produced: the derivations + trace."""

    derivations: list[AgentTaskDerivation] = Field(default_factory=list)
    trace: AgentTrace
    iterations: int = Field(ge=0, description="Number of Responses API round-trips.")
    stopped_reason: Literal["completed", "max_iterations"] = "completed"
    # Summed Responses API token usage of the loop (input/output/reasoning/cached);
    # ``None`` when the client reported none (e.g. the test fakes).
    usage: dict[str, int] | None = None
