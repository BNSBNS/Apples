"""Is the verifier actually worth having?

    python -m evals.verifier_eval
    python -m evals.verifier_eval --provider deepseek

Two numbers, never one:

    catch rate         of the proposals that ARE wrong, how many did it fail?
    false-reject rate  of the proposals that were FINE, how many did it fail?

Averaging them hides the failure that matters. Pass-everything scores 0%/0% and
is worse than no verifier — it makes an unchecked answer look checked.
Fail-everything scores 100%/100% and is equally useless. qwen2.5:7b does the
second one; a single accuracy number would have read 67% and looked fine.

Cases are planted directly in a temp DB — no writer runs — so the defects are
the ones worth testing rather than whatever a small model got wrong today.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from agent import loop_manual, verify
from data import seed
from evals.run_evals import describe_exception
from server import db

CASES = Path(__file__).parent / "verifier_cases.jsonl"


def load_cases() -> list[dict[str, Any]]:
    with CASES.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def plant(case: dict[str, Any], db_path: Path) -> None:
    """Write the proposal under review, and mark its citations as retrieved.

    The grounding gate is the writer's guard; here it would only get in the way
    of planting a fixture, so the citations are seeded as though a search had
    returned them.
    """
    db.insert_proposal(
        case["dispute_id"],
        case["disposition"],
        case["rationale"],
        case["citations"],
        case["amount"],
        db_path,
    )


async def run_case(case: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    try:
        out = await verify.verify(
            case["dispute_id"], args.provider, verbose=False, model_override=args.model
        )
    except Exception as exc:  # a crashed run is a data point, not a stop
        return {"verdict": None, "error": describe_exception(exc), "seconds": 0.0}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Measure the verifier against planted proposals.")
    ap.add_argument("--provider", default="local", choices=loop_manual.KINDS)
    ap.add_argument("--model", default=None, help="override the model")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    cases = load_cases()
    if args.limit:
        cases = cases[: args.limit]

    tmpdir = Path(tempfile.mkdtemp(prefix="verifier-eval-"))
    db_path = tmpdir / "bank.db"
    original = db.DB_PATH
    started = time.time()
    rows: list[tuple[dict[str, Any], str | None]] = []

    print(f"\nverifying {len(cases)} planted proposals · provider={args.provider}\n")
    try:
        for i, case in enumerate(cases, 1):
            # Fresh database per case: the verifier reads the newest pending
            # proposal for a dispute, so leftovers would review the wrong row.
            seed.build(db_path)
            db.DB_PATH = db_path
            os.environ["DISPUTE_DB"] = str(db_path)
            plant(case, db_path)

            print(f"  [{i}/{len(cases)}] {case['id']} ... ", end="", flush=True)
            out = asyncio.run(run_case(case, args))
            verdict = out.get("verdict")
            rows.append((case, verdict))

            want = "fail" if case["should_fail"] else "pass"
            mark = "ok" if verdict == want else ("MISS" if verdict else "no verdict")
            print(f"{verdict or '-'} (want {want}) {mark} [{out.get('seconds', 0):.1f}s]")
    finally:
        db.DB_PATH = original
        os.environ.pop("DISPUTE_DB", None)
        shutil.rmtree(tmpdir, ignore_errors=True)

    bad = [(c, v) for c, v in rows if c["should_fail"]]
    good = [(c, v) for c, v in rows if not c["should_fail"]]
    caught = sum(v == "fail" for _, v in bad)
    false_rejects = sum(v == "fail" for _, v in good)
    no_verdict = sum(v is None for _, v in rows)

    print(f"\n  {'metric':<24} value")
    print(f"  {'-' * 24} {'-' * 22}")
    if bad:
        print(f"  {'catch rate':<24} {caught}/{len(bad)}   ({100 * caught / len(bad):.0f}%)")
    if good:
        pct = 100 * false_rejects / len(good)
        print(f"  {'false-reject rate':<24} {false_rejects}/{len(good)}   ({pct:.0f}%)")
    print(f"  {'produced no verdict':<24} {no_verdict}")
    print(f"\n  total wall clock: {time.time() - started:.0f}s")

    if bad and caught == 0:
        print("\n  A verifier that catches nothing is worse than none — it makes an")
        print("  unchecked proposal look checked. Do not ship this as a safety story.")
    elif good and false_rejects == len(good):
        print("\n  It failed every correct proposal too, so the catch rate above is")
        print("  not discrimination — it is a verifier that always says 'fail'.")

    for case, verdict in rows:
        want = "fail" if case["should_fail"] else "pass"
        if verdict != want:
            print(f"\n  {case['id']} ({case['dispute_id']}) — {case['tests']}")
            print(f"     wanted {want}, got {verdict or 'no verdict'}")


if __name__ == "__main__":
    main()
