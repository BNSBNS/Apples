"""The human gate.

This file is the other half of the safety design, and it is deliberately not
importable by the agent. The agent can only ever write a `pending` proposal;
moving it to `approved` or `denied` happens here, driven by a person.

That separation is the whole argument: there is no code path from the model to
a committed money movement. Not a prompt instruction the model might ignore —
a structural property of the system.

    python -m agent.approval --list
    python -m agent.approval --approve 3 --by "your.name"
    python -m agent.approval --deny 4 --by "your.name"
"""

from __future__ import annotations

import argparse
import sys

from server import db

TERMINAL = {"approved", "denied"}


def list_pending() -> list[dict]:
    return db.query(
        """
        SELECT p.*, d.category, d.customer_id
        FROM proposals p JOIN disputes d ON d.id = p.dispute_id
        WHERE p.status = 'pending'
        ORDER BY p.id
        """
    )


def decide(proposal_id: int, decision: str, by: str) -> dict:
    """Commit a human decision. Refuses to re-decide an already-decided proposal."""
    if decision not in TERMINAL:
        raise ValueError(f"decision must be one of {sorted(TERMINAL)}")

    row = db.one("SELECT * FROM proposals WHERE id = ?", (proposal_id,))
    if row is None:
        return {"ok": False, "error": f"no proposal {proposal_id}"}
    if row["status"] != "pending":
        return {
            "ok": False,
            "error": f"proposal {proposal_id} is already '{row['status']}' "
            f"(decided by {row['decided_by']}) — refusing to overwrite",
        }

    conn = db.connect()
    try:
        conn.execute(
            "UPDATE proposals SET status=?, decided_at=datetime('now'), decided_by=? "
            "WHERE id=? AND status='pending'",
            (decision, by, proposal_id),
        )
        conn.commit()
    finally:
        conn.close()
    return {"ok": True, "proposal_id": proposal_id, "status": decision, "decided_by": by}


def main() -> None:
    ap = argparse.ArgumentParser(description="Review agent-proposed dispute resolutions.")
    ap.add_argument("--list", action="store_true", help="show pending proposals")
    ap.add_argument("--approve", type=int, metavar="ID")
    ap.add_argument("--deny", type=int, metavar="ID")
    ap.add_argument("--by", default="", help="who is deciding (required to decide)")
    args = ap.parse_args()

    if args.approve is not None or args.deny is not None:
        # Never silently pick one. This is the money gate: `--approve 3 --deny 4`
        # used to approve 3 and drop the deny without a word, which is the worst
        # possible reading of an ambiguous command here.
        if args.approve is not None and args.deny is not None:
            sys.exit("--approve and --deny are mutually exclusive: decide one proposal at a time.")
        if not args.by:
            sys.exit("--by is required: a decision must be attributable to a person.")
        pid = args.approve if args.approve is not None else args.deny
        decision = "approved" if args.approve is not None else "denied"
        out = decide(pid, decision, args.by)
        print(out.get("error") if not out["ok"] else f"proposal {pid} -> {decision} by {args.by}")
        return

    rows = list_pending()
    if not rows:
        print("no pending proposals.")
        return
    print(f"{len(rows)} pending proposal(s):\n")
    for r in rows:
        amt = f"  amount={r['amount']}" if r["amount"] is not None else ""
        print(f"  [{r['id']}] {r['dispute_id']} ({r['category']})  -> {r['disposition']}{amt}")
        print(f"       cites: {r['citations']}")
        print(f"       {r['rationale'][:150]}")
        # The independent reviewer's opinion, shown next to the proposal rather
        # than used to filter it. A reviewer who never sees a disagreement is
        # being managed, not informed.
        if r["verdict"] is None:
            print("       review: not reviewed")
        else:
            print(f"       review: {r['verdict'].upper()} — {(r['verdict_reasons'] or '')[:120]}")
        print()
    print("approve with:  python -m agent.approval --approve <id> --by <name>")


if __name__ == "__main__":
    main()
