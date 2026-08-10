"""Run the eval suite and print a metrics table.

    python -m evals.run_evals                                   # local default model
    python -m evals.run_evals --model qwen2.5:7b
    python -m evals.run_evals --provider cloud
    python -m evals.run_evals --retriever keyword                # retrieval ablation
    python -m evals.run_evals --limit 3                          # smoke run

Results are written to evals/results/<label>.json so runs can be compared
later without re-spending tokens.

Each case runs against a **fresh copy of the database**, so proposals from one
case can't leak into the next and a run can never mutate your working DB.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from agent import loop_manual
from data import seed
from evals.metrics import CaseResult, aggregate, load_cases, score
from server import db

RESULTS = Path(__file__).parent / "results"


def latest_proposal(dispute_id: str, db_path: Path) -> dict[str, Any] | None:
    rows = db.query(
        "SELECT * FROM proposals WHERE dispute_id = ? ORDER BY id DESC LIMIT 1",
        (dispute_id,),
        db_path,
    )
    return rows[0] if rows else None


def describe_exception(exc: BaseException, depth: int = 0) -> str:
    """Unwrap ExceptionGroups down to the cause that actually matters.

    anyio task groups raise `ExceptionGroup: unhandled errors in a TaskGroup
    (1 sub-exception)`, which says nothing. A whole 12-case cloud run reported
    as twelve clean misses because of this — the real cause was a TLS failure
    on the very first request, and the harness rendered it as a model result.

    An eval harness that turns infrastructure failure into a plausible-looking
    score is worse than one that crashes.
    """
    if depth > 5:
        return f"{type(exc).__name__}: {exc}"
    inner = getattr(exc, "exceptions", None)
    if inner:
        return " | ".join(describe_exception(e, depth + 1) for e in inner)
    cause = exc.__cause__ or exc.__context__
    if cause is not None and type(exc).__name__ in {"APIConnectionError", "ExceptionGroup"}:
        return f"{type(exc).__name__}: {exc} <- {describe_exception(cause, depth + 1)}"
    return f"{type(exc).__name__}: {exc}"


async def run_case(case: dict[str, Any], args: argparse.Namespace, db_path: Path) -> CaseResult:
    try:
        out = await loop_manual.run(
            case["id"], args.provider, verbose=False, model_override=args.model
        )
    except Exception as exc:  # a crashed run is a data point, not a stop
        out = {"error": describe_exception(exc), "trace": []}
    return score(case, out, latest_proposal(case["id"], db_path))


def fmt_pct(x: float | None) -> str:
    return "  n/a" if x is None else f"{100 * x:5.1f}%"


def print_table(results: list[CaseResult]) -> None:
    print()
    hdr = f"  {'case':<8} {'expected':<19} {'got':<19} {'ok':<3} {'cite':<6} {'ret':<4} {'amt':<4}"
    print(f"{hdr} {'s':>6}")
    print(f"  {'-' * 8} {'-' * 19} {'-' * 19} {'-' * 3} {'-' * 6} {'-' * 4} {'-' * 4} {'-' * 6}")
    for r in results:
        got = r.disposition or "(none)"
        ok = "✓" if r.correct else "✗"
        if r.cited_forbidden:
            cite = "WRONG"
        elif r.hallucinated_citation:
            cite = "halluc"
        elif r.cited_gold:
            cite = "gold"
        elif r.cited:
            cite = "off"
        else:
            cite = "-"
        ret = "hit" if r.retrieval_hit else ("miss" if r.searched else "none")
        # "-" means the case labels no expected amount, not that the amount passed.
        amt = "-" if r.amount_correct is None else ("✓" if r.amount_correct else "✗")
        mark = " ⚠" if r.adversarial else ""
        row = f"  {r.case_id:<8} {r.expected:<19} {got:<19} {ok:<3} {cite:<6} {ret:<4} {amt:<4}"
        print(f"{row} {r.seconds:6.1f}{mark}")


async def _run_all_cases(
    cases: list[dict[str, Any]], args: argparse.Namespace, db_path: Path
) -> list[CaseResult]:
    """Run every case inside one event loop.

    The original design called asyncio.run() per case, which created and
    destroyed an event loop 30 times.  On Windows the ProactorEventLoop
    teardown races with httpx's AsyncClient.__del__, producing a wall of
    'Event loop is closed' tracebacks.  A single loop avoids the issue;
    per-case isolation still comes from reseeding the DB each iteration.
    """
    results: list[CaseResult] = []
    for i, case in enumerate(cases, 1):
        seed.build(db_path)
        print(f"  [{i}/{len(cases)}] {case['id']} ... ", end="", flush=True)
        res = await run_case(case, args, db_path)
        results.append(res)
        print(f"{'ok' if res.correct else 'MISS'} ({res.seconds:.1f}s)")

        if res.tool_calls == 0 and "no proposal recorded" not in res.error:
            print(
                f"\n  ABORTING — this looks like infrastructure, not the model:\n  {res.error}"
            )
            break
    return results


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate the dispute triage agent.")
    ap.add_argument("--provider", default="local", choices=loop_manual.KINDS)
    ap.add_argument("--model", default=None, help="override model, e.g. qwen2.5:7b, hermes3:8b")
    ap.add_argument("--retriever", default=None, choices=["keyword", "embedding"])
    ap.add_argument("--limit", type=int, default=None, help="run only the first N cases")
    ap.add_argument("--label", default=None, help="name for the results file")
    args = ap.parse_args()

    if args.retriever:
        os.environ["RETRIEVER"] = args.retriever

    cases = load_cases()
    if args.limit:
        cases = cases[: args.limit]

    label = args.label or f"{args.provider}-{(args.model or 'default').replace(':', '_')}"
    if args.retriever:
        label += f"-{args.retriever}"

    shown_model = args.model or "from .env"
    print(f"\nrunning {len(cases)} cases · provider={args.provider} · model={shown_model}")
    if args.provider == "cloud":
        print("note: this spends API credits — one call per agent step, per case.")

    # Isolate the run: a temp DB means eval writes never touch the working
    # database, and it is reseeded before every case (see the loop below) so no
    # case can score against a proposal another case wrote.
    tmpdir = Path(tempfile.mkdtemp(prefix="dispute-eval-"))
    db_path = tmpdir / "bank.db"
    original = db.DB_PATH
    db.DB_PATH = db_path
    # Critical: the MCP server is a subprocess and cannot see the line above.
    # It reads DISPUTE_DB from the environment instead.
    os.environ["DISPUTE_DB"] = str(db_path)

    started = time.time()
    try:
        results = asyncio.run(_run_all_cases(cases, args, db_path))
    finally:
        db.DB_PATH = original
        os.environ.pop("DISPUTE_DB", None)
        shutil.rmtree(tmpdir, ignore_errors=True)

    print_table(results)
    agg = aggregate(results)

    print(f"\n  {'metric':<26} value")
    print(f"  {'-' * 26} {'-' * 10}")
    print(f"  {'disposition accuracy':<26} {fmt_pct(agg['disposition_accuracy'])}")
    grounded = fmt_pct(agg["grounded_accuracy"])
    print(f"  {'grounded accuracy':<26} {grounded}   (correct AND cited gold)")
    amount = fmt_pct(agg["amount_accuracy"])
    print(f"  {'amount accuracy':<26} {amount}   (over {agg['amount_cases']} labelled cases)")
    print(f"  {'retrieval recall':<26} {fmt_pct(agg['retrieval_recall'])}")
    print(f"  {'searched at all':<26} {fmt_pct(agg['searched_rate'])}")
    print(f"  {'injection resisted':<26} {fmt_pct(agg['injection_resisted'])}")
    print(f"  {'hallucinated citations':<26} {agg['hallucinated_citations']:>6}")
    print(f"  {'cited a forbidden policy':<26} {agg['forbidden_citations']:>6}")
    print(f"  {'argument repairs':<26} {agg['arg_repairs']:>6}   (wrong param names we fixed)")
    print(f"  {'produced no proposal':<26} {agg['no_proposal']:>6}")
    print(f"  {'AUTO-COMMITS (must be 0)':<26} {agg['auto_commits']:>6}")
    print(f"  {'avg tool calls':<26} {agg['avg_tool_calls']:>6.1f}")
    print(f"  {'avg seconds/case':<26} {agg['avg_seconds']:>6.1f}")
    print(
        f"  {'tokens (in/out)':<26} {agg['total_prompt_tokens']}/{agg['total_completion_tokens']}"
    )
    print(f"\n  total wall clock: {time.time() - started:.0f}s")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / f"{label}.json"
    out_path.write_text(
        json.dumps(
            {
                "label": label,
                # Bumped whenever a metric changes meaning, so an old file is
                # never silently compared against a new run. `retrieval_recall`
                # used to over-count (it regex-scanned the whole payload, so a
                # policy merely *mentioned* in another's text counted as
                # retrieved); v1 is the first version that reads policy_id.
                "metrics_version": 1,
                "provider": args.provider,
                "model": args.model,
                "retriever": args.retriever,
                "aggregate": agg,
                "cases": [r.as_dict() for r in results],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    # Display only — `relative_to` raises when the results directory is not under
    # the cwd (a different drive on Windows is enough). Letting that propagate
    # would skip the safety gate immediately below, which is the one thing in
    # this file that must always run.
    try:
        shown_path: Path | str = out_path.relative_to(Path.cwd())
    except ValueError:
        shown_path = out_path
    print(f"  written to {shown_path}")

    # The safety invariant is not a metric to observe — it is a gate.
    if agg["auto_commits"]:
        print("\n  FAIL: an agent run produced a non-pending proposal.")
        sys.exit(1)


if __name__ == "__main__":
    main()
