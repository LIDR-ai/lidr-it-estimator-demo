#!/usr/bin/env python3
"""Token / cost comparison of the multi-agent graph WITHOUT vs WITH the Jev triage.

Jev only changes what happens inside ``recover_and_handover``: every doubtful task it
accepts is one less task for Trinity's gpt-5 recovery loop. A fair comparison must
therefore hold everything else fixed, so this script FORKS the graph after gate 1:

1. COMMON LEG (paid once): START → classifier (gpt-4o-mini) → structure (gpt-5) → gate 1.
2. BRANCH "off" and BRANCH "on": each a fresh thread that REPLAYS the common leg's
   output (no LLM calls), approves gate 1 with the SAME structure and runs
   fan-out → recover_and_handover → analysis_agent up to gate 2 — once with Jev
   disabled, once with it enabled. The per-task retrieval is deterministic, so both
   branches see the same hours; the script checks that and warns if they differ.

The proposal leg is skipped on purpose: it does not depend on Jev and only adds noise.

Usage::

    docker compose cp ai-service/exercises/session-12/sample_transcript_complex.txt \\
        ai-service:/tmp/transcript.txt

    # real Jev (needs JEV_API_KEY and Jev up)
    docker compose exec ai-service python scripts/compare_jev_tokens.py \\
        --transcript /tmp/transcript.txt --out /tmp/cmp.md

    # Jev SIMULATED (deterministic, accepts 60% of its candidates) — a what-if
    # projection with a fixed acceptance rate; labelled SIMULATED in the report
    docker compose exec ai-service python scripts/compare_jev_tokens.py \\
        --transcript /tmp/transcript.txt --stub-mixed --simulate-jev-accept-rate 0.6

Prices come from ``MODEL_COSTS`` (list prices, cached-input discount ignored). Jev
reports its own token usage (TypeSafe bills input tokens only); the simulated Jev
makes no calls and costs nothing.
Retrieval embeddings are identical in both branches and are not metered.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
for path in (REPO_ROOT, Path(__file__).resolve().parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from langgraph.checkpoint.memory import MemorySaver  # noqa: E402
from langgraph.types import Command  # noqa: E402

import run_graph_s13  # noqa: E402
from app.domain.graph import build as build_mod  # noqa: E402
from app.domain.graph.agents import hours as hours_mod  # noqa: E402
from app.domain.graph.agents._common import flag_reason  # noqa: E402
from app.foundation.llm.jev import JevVerdict  # noqa: E402
from app.foundation.llm.usage_meter import UsageLedger, UsageRecord, metered  # noqa: E402


class SimulatedJev:
    """Deterministic stand-in: accepts ``rate`` of its candidates (by task hash)."""

    def __init__(self, rate: float) -> None:
        self.rate = rate

    async def classify(self, tasks):
        out = []
        for t in tasks:
            digest = hashlib.sha256(f"{t.module}/{t.task}".encode()).digest()
            accept = digest[0] / 256 < self.rate
            p = 0.9 if accept else 0.2
            out.append(
                JevVerdict(
                    module=t.module,
                    task=t.task,
                    decision="accept" if accept else "escalate",
                    confidence=max(p, 1 - p),
                    probabilities={"accept": p, "escalate": round(1 - p, 3)},
                )
            )
        return out


# --- the two legs ------------------------------------------------------------ #


async def run_common_leg(transcript: str) -> tuple[dict, UsageLedger]:
    graph = build_mod.build_graph(MemorySaver())
    config = {"configurable": {"thread_id": "cmp-base"}}
    with metered() as ledger:
        await graph.ainvoke({"transcript": transcript, "estimation_id": "cmp-base"}, config)
    values = (await graph.aget_state(config)).values
    return values, ledger


def _replay_common(common: dict) -> None:
    """Swap the two upstream agents for replays of the common leg (no LLM calls)."""

    async def classifier_replay(state):
        return Command(
            goto="structure_agent",
            update={
                "complexity": common.get("complexity"),
                "reformulated_transcript": common.get("reformulated_transcript"),
            },
        )

    async def structure_replay(state):
        return {"structure": common.get("structure")}

    build_mod.classifier_agent = classifier_replay
    build_mod.structure_agent = structure_replay


async def run_branch(name: str, transcript: str, jev) -> dict:
    import app.dependencies as deps

    deps.get_jev_classifier = lambda: jev  # the join imports it from app.dependencies

    retrieved: dict[tuple[str, str], dict] = {}
    inner = hours_mod.estimate_one

    async def recording_estimate_one(module, name_, description, **kwargs):
        est = await inner(module, name_, description, **kwargs)
        retrieved[(module, name_)] = est.model_dump()
        return est

    hours_mod.estimate_one = recording_estimate_one
    try:
        graph = build_mod.build_graph(MemorySaver())
        thread = f"cmp-{name}"
        config = {"configurable": {"thread_id": thread}}
        await graph.ainvoke({"transcript": transcript, "estimation_id": thread}, config)
        with metered() as ledger:
            # gate 1 approved as proposed → fan-out → Jev → Trinity → analysis → gate 2
            await graph.ainvoke(Command(resume={"approved": True}), config)
        values = (await graph.aget_state(config)).values
    finally:
        hours_mod.estimate_one = inner

    doubtful = [k for k, row in retrieved.items() if flag_reason(row) is not None]
    decisions = values.get("jev_decisions") or []
    accepted = sum(
        1
        for t in values.get("task_hours") or []
        if t.get("accepted_by") in ("jev", "jev_simulated")
    )
    return {
        "name": name,
        "ledger": ledger,
        "retrieved": retrieved,
        "tasks": len(retrieved),
        "doubtful": len(doubtful),
        "jev_candidates": len(decisions),
        "jev_accepted": accepted,
        "jev_errors": sum(1 for d in decisions if d.get("error")),
        "trinity_tasks": len(doubtful) - accepted,
        "days": (values.get("estimate") or {}).get("total_engineer_days"),
    }


# --- report ------------------------------------------------------------------ #

_LABELS = {
    "ComplexityClassification": "classifier (Morpheus)",
    "structure_agent": "structure (Neo, gpt-5)",
    "trinity_recovery": "recovery (Trinity, gpt-5)",
    "ReliabilityReport": "analysis (Oracle)",
    "jev": "Jev triage (TypeSafe)",
}


def _row(rec: UsageRecord | None) -> tuple[int, int, int, int, float]:
    if rec is None:
        return 0, 0, 0, 0, 0.0
    return rec.calls, rec.input_tokens, rec.output_tokens, rec.reasoning_tokens, rec.cost_usd


def _fmt(n: int) -> str:
    return f"{n:,}".replace(",", ".")


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def render(
    common: UsageLedger,
    off: dict | list[dict],
    on: dict | list[dict],
    *,
    simulated: bool,
    max_iterations: int | None = None,
    trinity_max_tasks: int | None = None,
) -> str:
    """Markdown report. ``off``/``on`` may be one run each or a list of repeats; the
    detailed per-agent table shows the first pair, the savings use the MEANS."""
    offs = off if isinstance(off, list) else [off]
    ons = on if isinstance(on, list) else [on]
    off, on = offs[0], ons[0]
    lo, ln = off["ledger"], on["ledger"]
    so, sn = lo.by_source(), ln.by_source()
    sources = [s for s in _LABELS if s in so or s in sn] + sorted(
        (set(so) | set(sn)) - set(_LABELS)
    )
    to, tn = lo.totals(), ln.totals()
    tc = common.totals()

    jev_label = "SIMULATED Jev" if simulated else "real Jev"
    lines = [
        "# Token comparison · graph without Jev vs with Jev",
        "",
        f"Jev mode: **{jev_label}**"
        + (" (no Jev calls: no Jev tokens)" if simulated else " · priced from its reported usage"),
        "",
        "## Common leg (paid once, identical for both)",
        "",
        f"classifier + structure: {_fmt(tc.input_tokens)} in / {_fmt(tc.output_tokens)} out "
        f"({_fmt(tc.reasoning_tokens)} reasoning) · **${tc.cost_usd:.4f}**",
        "",
        "## Variable leg (fan-out → recover_and_handover → analysis)",
        "",
        "| Agent | Without Jev: calls · in / out (reasoning) · USD | With Jev: calls · in / out (reasoning) · USD |",
        "|---|---|---|",
    ]
    for s in sources:
        a, b = _row(so.get(s)), _row(sn.get(s))
        cell = lambda r: (  # noqa: E731
            f"{r[0]} · {_fmt(r[1])} / {_fmt(r[2])} ({_fmt(r[3])}) · ${r[4]:.4f}" if r[0] else "—"
        )
        lines.append(f"| {_LABELS.get(s, s)} | {cell(a)} | {cell(b)} |")
    lines.append(
        f"| **Total** | **{_fmt(to.input_tokens + to.output_tokens)} tokens · ${to.cost_usd:.4f}** "
        f"| **{_fmt(tn.input_tokens + tn.output_tokens)} tokens · ${tn.cost_usd:.4f}** |"
    )

    def tokens(run):
        t = run["ledger"].totals()
        return t.input_tokens + t.output_tokens

    def cost(run):
        return run["ledger"].totals().cost_usd

    def trinity_cost(run):
        return _row(run["ledger"].by_source().get("trinity_recovery"))[4]

    def trinity_calls(run):
        return _row(run["ledger"].by_source().get("trinity_recovery"))[0]

    cost_off, cost_on = _mean([cost(r) for r in offs]), _mean([cost(r) for r in ons])
    saved_tokens = round(_mean([tokens(r) for r in offs]) - _mean([tokens(r) for r in ons]))
    saved_usd = round(cost_off - cost_on, 6)
    pct_branch = 100 * saved_usd / cost_off if cost_off else 0.0
    full_off = tc.cost_usd + cost_off
    pct_full = 100 * saved_usd / full_off if full_off else 0.0
    accepted = on["jev_accepted"]
    trin_saved = _mean([trinity_cost(r) for r in offs]) - _mean([trinity_cost(r) for r in ons])
    per_task = trin_saved / accepted if accepted else None

    lines += [
        "",
        "## Triage",
        "",
        "| | Without Jev | With Jev |",
        "|---|---|---|",
        f"| tasks / doubtful after retrieval | {off['tasks']} / {off['doubtful']} | {on['tasks']} / {on['doubtful']} |",
        f"| Jev candidates / accepted / errors | — | {on['jev_candidates']} / {on['jev_accepted']} / {on['jev_errors']} |",
        f"| tasks handed to Trinity | {off['trinity_tasks']} | {on['trinity_tasks']} |",
        f"| estimate (engineer-days) | {off['days']} | {on['days']} |",
    ]
    if max_iterations:

        def calls_cell(runs):
            calls = [trinity_calls(r) for r in runs]
            flag = " ⚠ hit the cap" if any(c >= max_iterations for c in calls) else ""
            return f"{'/'.join(str(c) for c in calls)} (cap {max_iterations}){flag}"

        lines.append(f"| Trinity round-trips | {calls_cell(offs)} | {calls_cell(ons)} |")

    if len(offs) > 1:

        def spread(runs):
            xs = [cost(r) for r in runs]
            return f"${_mean(xs):.4f} (min ${min(xs):.4f} · max ${max(xs):.4f})"

        lines += [
            "",
            f"## Repeats ({len(offs)} runs per branch, same structure and retrieval)",
            "",
            "| | Without Jev | With Jev |",
            "|---|---|---|",
            f"| variable-leg cost: mean (min · max) | {spread(offs)} | {spread(ons)} |",
        ]
        overlap = max(cost(r) for r in ons) >= min(cost(r) for r in offs)
        if overlap:
            lines.append(
                "\n⚠ The two ranges OVERLAP: the saving is within gpt-5's run-to-run noise "
                "for this many repeats — present it as a trend, not a measured figure."
            )

    lines += [
        "",
        "## Savings" + (f" (means over {len(offs)} runs)" if len(offs) > 1 else ""),
        "",
    ]
    if not accepted:
        lines.append(
            "- ⚠ **Jev accepted no task in this run**, so both branches handed Trinity the "
            "SAME tasks: any difference below is pure gpt-5 run-to-run noise, not a Jev effect."
        )
    lines += [
        f"- Tokens saved in the variable leg: **{_fmt(saved_tokens)}**",
        f"- Cost saved: **${saved_usd:.4f}** = **{pct_branch:.1f}%** of the variable leg, "
        f"**{pct_full:.1f}%** of the whole run (common leg ${tc.cost_usd:.4f} + variable ${cost_off:.4f})",
    ]
    if per_task is not None:
        lines.append(
            f"- Trinity cost avoided per task Jev accepted ≈ **${per_task:.4f}** "
            f"→ ≈ ${per_task * 100:.2f} per 100 accepted tasks"
        )
    lines += [
        "",
        "Caveats: gpt-5 token counts vary between runs (re-run for a range); Trinity works "
        "in BATCH and is capped at GRAPH_RECOVERY_MAX_ITERATIONS round-trips — when it hits the cap "
        "its cost tracks the cap, not the task count, and the saving is UNDERSTATED; the "
        "saving per accepted task is an average, not a constant; prices "
        "are list prices from MODEL_COSTS; embeddings are not metered (identical in both).",
    ]
    if trinity_max_tasks is not None:
        lines.append(
            "Trinity task cap (GRAPH_RECOVERY_MAX_TASKS): "
            + (
                f"**{trinity_max_tasks}** — with a cap below the doubtful count both branches "
                "hand Trinity the same number of tasks and Jev's saving shrinks or vanishes."
                if trinity_max_tasks > 0
                else "**off** for this comparison, so every task Jev does not accept reaches "
                "Trinity and the saving is measurable (the demo graph caps it)."
            )
        )
    if simulated:
        lines.append(
            "**SIMULATED**: Jev's decisions come from a deterministic stand-in, not the real "
            "API — the Trinity/analysis tokens are real, the acceptance rate is assumed."
        )
    return "\n".join(lines)


# --- main -------------------------------------------------------------------- #


async def _main_async(args: argparse.Namespace) -> int:
    transcript_path = Path(args.transcript)
    if not transcript_path.is_file():
        print(f"ERROR: transcript not found: {transcript_path}", file=sys.stderr)
        return 1
    transcript = transcript_path.read_text(encoding="utf-8")

    from app.config import get_settings
    from app.dependencies import get_jev_classifier

    settings = get_settings()
    # The graph hands Trinity at most GRAPH_RECOVERY_MAX_TASKS doubtful tasks. Under that
    # cap both branches would send it the same number and hide Jev's effect, so the
    # comparison runs uncapped unless asked otherwise (Settings is a cached singleton).
    settings.GRAPH_RECOVERY_MAX_TASKS = args.trinity_max_tasks
    if args.stub_mixed:
        run_graph_s13._install_stub_mixed_hours()
    elif args.stub:
        run_graph_s13._install_stub_hours()

    simulated = args.simulate_jev_accept_rate is not None
    jev = SimulatedJev(args.simulate_jev_accept_rate) if simulated else get_jev_classifier()
    if jev is None:
        print(
            "ERROR: Jev is not configured (GRAPH_JEV_ENABLED/JEV_API_KEY). "
            "Set the key or pass --simulate-jev-accept-rate.",
            file=sys.stderr,
        )
        return 1

    print("1/3 common leg (classifier + structure)…", flush=True)
    common_values, common_ledger = await run_common_leg(transcript)
    _replay_common(common_values)
    offs, ons = [], []
    for i in range(args.repeats):
        tag = f" (repeat {i + 1}/{args.repeats})" if args.repeats > 1 else ""
        print(f"2/3 branch WITHOUT Jev{tag}…", flush=True)
        offs.append(await run_branch(f"off-{i}", transcript, None))
        print(f"3/3 branch WITH Jev{tag}…", flush=True)
        ons.append(await run_branch(f"on-{i}", transcript, jev))

    if any(r["retrieved"] != offs[0]["retrieved"] for r in offs + ons):
        print("WARNING: per-task retrieval differed between branches — comparison not exact.")

    report = render(
        common_ledger,
        offs,
        ons,
        simulated=simulated,
        max_iterations=settings.GRAPH_RECOVERY_MAX_ITERATIONS,
        trinity_max_tasks=settings.GRAPH_RECOVERY_MAX_TASKS,
    )
    print("\n" + report)
    if args.out:
        Path(args.out).write_text(report + "\n", encoding="utf-8")
        print(f"\n(report written to {args.out})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare graph token usage without/with Jev.")
    parser.add_argument("--transcript", default=str(run_graph_s13.DEFAULT_TRANSCRIPT))
    parser.add_argument("--stub", action="store_true", help="Canned offline per-task hours.")
    parser.add_argument(
        "--stub-mixed",
        action="store_true",
        help="Canned hours with a mix of reliabilities (guarantees Jev candidates).",
    )
    parser.add_argument(
        "--simulate-jev-accept-rate",
        type=float,
        metavar="RATE",
        help="Use a deterministic simulated Jev accepting this fraction (0..1) of candidates.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="Run each branch N times and report mean/min/max (gpt-5 is noisy; 3 is a good demo value).",
    )
    parser.add_argument(
        "--trinity-max-tasks",
        type=int,
        default=0,
        metavar="N",
        help="Cap the doubtful tasks handed to Trinity (default 0 = no cap, so Jev's saving shows).",
    )
    parser.add_argument("--out", help="Write the markdown report to this file.")
    args = parser.parse_args()
    return asyncio.run(_main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
