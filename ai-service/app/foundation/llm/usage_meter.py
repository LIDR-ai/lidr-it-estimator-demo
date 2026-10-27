"""Per-run token meter — who spent how many tokens, at what estimated price.

A ``ContextVar`` holds the active :class:`UsageLedger`. Every LLM call site calls
:func:`record`; with no ledger active (production, the API, the tests) it is a no-op,
so metering costs nothing unless a caller opts in with ``with metered() as ledger:``.

The ledger is a mutable object shared by reference, so it survives
``asyncio.to_thread`` (which copies the context) and ``asyncio.gather`` — the two ways
the graph nodes fan work out.

Also home of the pricing table (``MODEL_COSTS``, USD per 1M tokens) so both the
LiteLLM wrapper and the hand-written Responses API agents price calls the same way.
Prices are list prices; cached-input discounts are ignored (a slight overestimate).
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

MODEL_COSTS: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    # Session 9 — reasoning models used by the RAG estimation pipeline.
    "gpt-5": {"input": 1.25, "output": 10.00},
    "gpt-5-mini": {"input": 0.25, "output": 2.00},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    "claude-haiku-4-5-20251001": {"input": 1.00, "output": 5.00},
    "claude-sonnet-4-5": {"input": 3.00, "output": 15.00},
    # Jev (TypeSafe AI): $42 per BILLION input tokens; output tokens are free.
    "jev": {"input": 0.042, "output": 0.0},
}


def _normalise_model_name(model: str) -> str:
    """Strip provider prefixes like ``anthropic/`` that LiteLLM may emit, and fold every
    Jev alias/version (``jev-latest``, ``jev-1.14.0``…) into the one priced ``jev``."""
    name = model.split("/", 1)[1] if "/" in model else model
    return "jev" if name == "jev" or name.startswith("jev-") else name


def _estimate_cost(model: str, tokens_in: int, tokens_out: int) -> float:
    base = _normalise_model_name(model)
    costs = MODEL_COSTS.get(base) or MODEL_COSTS.get(model) or {"input": 0.0, "output": 0.0}
    return round((tokens_in * costs["input"] + tokens_out * costs["output"]) / 1_000_000, 6)


@dataclass
class UsageRecord:
    """One billable call. ``output_tokens`` already INCLUDES ``reasoning_tokens``."""

    source: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    cached_input_tokens: int = 0
    cost_usd: float = 0.0
    calls: int = 1


@dataclass
class UsageLedger:
    records: list[UsageRecord] = field(default_factory=list)

    def totals(self) -> UsageRecord:
        return _sum("total", "*", self.records)

    def by_source(self) -> dict[str, UsageRecord]:
        return self._group(lambda r: r.source)

    def by_model(self) -> dict[str, UsageRecord]:
        return self._group(lambda r: r.model)

    def _group(self, key) -> dict[str, UsageRecord]:
        groups: dict[str, list[UsageRecord]] = {}
        for r in self.records:
            groups.setdefault(key(r), []).append(r)
        return {
            k: _sum(rs[0].source if len({r.source for r in rs}) == 1 else "*", rs[0].model, rs)
            for k, rs in groups.items()
        }


def _sum(source: str, model: str, records: list[UsageRecord]) -> UsageRecord:
    return UsageRecord(
        source=source,
        model=model,
        input_tokens=sum(r.input_tokens for r in records),
        output_tokens=sum(r.output_tokens for r in records),
        reasoning_tokens=sum(r.reasoning_tokens for r in records),
        cached_input_tokens=sum(r.cached_input_tokens for r in records),
        cost_usd=round(sum(r.cost_usd for r in records), 6),
        calls=sum(r.calls for r in records),
    )


_current: ContextVar[UsageLedger | None] = ContextVar("usage_ledger", default=None)


@contextmanager
def metered() -> Iterator[UsageLedger]:
    """Activate a fresh ledger for the enclosed block (nesting restores the outer one)."""
    ledger = UsageLedger()
    token = _current.set(ledger)
    try:
        yield ledger
    finally:
        _current.reset(token)


def record(
    source: str,
    model: str,
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    reasoning_tokens: int = 0,
    cached_input_tokens: int = 0,
    cost_usd: float | None = None,
) -> UsageRecord | None:
    """Append one call to the active ledger; no-op (``None``) when none is active."""
    ledger = _current.get()
    if ledger is None:
        return None
    rec = UsageRecord(
        source=source,
        model=_normalise_model_name(model),
        input_tokens=input_tokens or 0,
        output_tokens=output_tokens or 0,
        reasoning_tokens=reasoning_tokens or 0,
        cached_input_tokens=cached_input_tokens or 0,
        cost_usd=(
            _estimate_cost(model, input_tokens or 0, output_tokens or 0)
            if cost_usd is None
            else cost_usd
        ),
    )
    ledger.records.append(rec)
    return rec


def usage_from_responses_api(response) -> dict[str, int]:
    """Token counts from an OpenAI Responses API object (defensive: fakes lack usage)."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    out_details = getattr(usage, "output_tokens_details", None)
    in_details = getattr(usage, "input_tokens_details", None)
    return {
        "input_tokens": getattr(usage, "input_tokens", 0) or 0,
        "output_tokens": getattr(usage, "output_tokens", 0) or 0,
        "reasoning_tokens": getattr(out_details, "reasoning_tokens", 0) or 0,
        "cached_input_tokens": getattr(in_details, "cached_tokens", 0) or 0,
    }


def usage_from_chat_completion(completion) -> dict[str, int]:
    """Token counts from a Chat Completions object (LiteLLM / Instructor raw response)."""
    usage = getattr(completion, "usage", None)
    if usage is None:
        return {}
    out_details = getattr(usage, "completion_tokens_details", None)
    in_details = getattr(usage, "prompt_tokens_details", None)
    return {
        "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
        "reasoning_tokens": getattr(out_details, "reasoning_tokens", 0) or 0,
        "cached_input_tokens": getattr(in_details, "cached_tokens", 0) or 0,
    }
