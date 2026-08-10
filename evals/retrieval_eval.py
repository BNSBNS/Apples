"""Measure retrieval on its own, with no LLM in the loop.

This exists to answer one question before any reranker gets built: **is the
right clause being retrieved but ranked badly?**

A cross-encoder reranker can only fix ordering. If the gold policy already
ranks first, reranking has no headroom and the ablation would measure nothing —
a null result dressed up as engineering. If gold is retrieved but sits at rank
3-5, there is real headroom and the reranker is justified.

Runs in seconds and costs nothing, which is the other point: retrieval is the
part of a RAG system you can iterate on fast. Put the LLM in the loop and every
experiment costs minutes.

    python -m evals.retrieval_eval
    python -m evals.retrieval_eval --retriever keyword
"""

from __future__ import annotations

import argparse
import os

from evals.metrics import load_cases
from server import db


def build_query(dispute: dict, rich: bool = False) -> str:
    """Approximate the query a competent agent would write.

    Deliberately never the gold clause text — that would measure nothing.

    Two modes, because the difference between them turned out to be the whole
    story:

    `narrative` — dispute category, account type, and the customer's own words.

    `rich` — the same, plus the case facts the agent has *already fetched* by
    the time it searches: prior claim count, account standing, and the disputed
    amount. Several gold policies (escalation triggers, out-of-scope rules) are
    reachable only from these facts. They appear nowhere in the customer's
    narrative, so a narrative-only query cannot retrieve them at any k, and no
    reranker can recover what was never retrieved.
    """
    base = (
        f"{dispute['account_type']} account, {dispute['category'].replace('_', ' ')} dispute. "
        f"{dispute['customer_statement']}"
    )
    if not rich:
        return base

    facts = [
        f"account status {dispute['account_status']}",
        f"{dispute['prior_claims_12m']} prior claims in 12 months",
    ]
    if dispute.get("amount") is not None:
        facts.append(f"disputed amount {dispute['amount']:.2f}")
    return f"{base} Case facts: {', '.join(facts)}."


def rank_of(gold: list[str], hits: list[tuple]) -> int | None:
    """1-based rank of the first gold policy in the hit list, or None."""
    for i, (chunk, _score) in enumerate(hits, start=1):
        if chunk.policy_id in gold:
            return i
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Measure retrieval quality without an LLM.")
    ap.add_argument("--retriever", default=None, choices=["keyword", "embedding"])
    ap.add_argument("-k", type=int, default=8, help="how deep to retrieve")
    ap.add_argument(
        "--rich",
        action="store_true",
        help="include case facts the agent has already fetched (amount, prior claims, standing)",
    )
    ap.add_argument("--rerank", action="store_true", help="apply the cross-encoder reranker")
    args = ap.parse_args()

    if args.retriever:
        os.environ["RETRIEVER"] = args.retriever

    # Imported after RETRIEVER is set — the factory reads it at construction.
    from server.retrieval import get_retriever

    retriever = get_retriever(rerank=args.rerank)
    kind = type(retriever).__name__
    if args.rerank:
        kind = f"{type(getattr(retriever, 'base', retriever)).__name__} + CrossEncoder"

    cases = load_cases()
    mode = "rich (with fetched case facts)" if args.rich else "narrative only"
    print(f"\n{kind} · k={args.k} · {len(cases)} cases · query: {mode}\n")
    print(f"  {'case':<8} {'gold':<9} {'rank':>5}  top hit")
    print(f"  {'-' * 8} {'-' * 9} {'-' * 5}  {'-' * 46}")

    ranks: list[int | None] = []
    for case in cases:
        dispute = db.one(
            "SELECT d.*, c.account_type, c.status AS account_status, c.prior_claims_12m, "
            "t.amount FROM disputes d "
            "JOIN customers c ON c.id = d.customer_id "
            "LEFT JOIN transactions t ON t.id = d.transaction_id "
            "WHERE d.id = ?",
            (case["id"],),
        )
        hits = retriever.search(build_query(dispute, rich=args.rich), k=args.k)
        r = rank_of(case["gold_policies"], hits)
        ranks.append(r)

        top = hits[0][0]
        shown = f"{top.policy_id} {top.heading[:38]}"
        print(f"  {case['id']:<8} {case['gold_policies'][0]:<9} {str(r or '-'):>5}  {shown}")

    n = len(ranks)
    found = [r for r in ranks if r is not None]
    at1 = sum(r == 1 for r in found) / n
    mrr = sum(1 / r for r in found) / n

    # Only report cut-offs at or below k. Nothing was ranked deeper than k, so
    # recall@5 from a k=3 run is really recall@3 wearing a bigger label — it
    # understates the retriever and is not comparable with a k=8 run.
    print()
    print(f"  recall@1   {at1:6.1%}")
    for cut in (3, 5):
        if cut < args.k:
            print(f"  recall@{cut}   {sum(r <= cut for r in found) / n:6.1%}")
    if args.k > 1:
        print(f"  recall@{args.k}   {len(found) / n:6.1%}")
    print(f"  MRR        {mrr:6.3f}")

    # The reranker decision, stated in terms of what was measured.
    headroom = len(found) / n - at1
    print()
    if not found:
        print("  VERDICT: retrieval is failing outright. A reranker cannot fix recall —")
        print("           fix chunking or the embedding model first.")
    elif headroom < 0.15:
        print(f"  VERDICT: gold already ranks first {at1:.0%} of the time; only {headroom:.0%} of")
        print("           cases have gold retrieved-but-misranked. A reranker has almost no")
        print("           headroom here — it would measure noise. Skip it.")
    else:
        print(f"  VERDICT: gold is retrieved {len(found) / n:.0%} of the time but ranks first only")
        print(f"           {at1:.0%}. That {headroom:.0%} gap is exactly what a cross-encoder")
        print("           reranks. Worth building.")


if __name__ == "__main__":
    main()
