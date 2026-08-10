"""Tool behaviour, tested without a model.

The @mcp.tool() decorator returns the original function, so each tool is an
ordinary callable and can be unit-tested directly. No LLM, no key, no network.

The most important tests here are the safety invariants on propose_resolution:
the agent must not be able to produce anything other than a pending proposal.
"""

from __future__ import annotations

import pytest

from data import seed
from server import app, db
from server.session_state import STDIO_KEY, SessionStore


@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    """A throwaway copy of the seeded DB so write tests don't pollute it."""
    path = tmp_path / "bank.db"
    seed.build(path)
    monkeypatch.setattr(db, "DB_PATH", path)
    return path


@pytest.fixture()
def grounded(monkeypatch):
    """Declare what search_policy is deemed to have returned, for this test only.

    `app._RETRIEVED` is server-lifetime state, so a write test that leans on some
    earlier test in the file having searched is order-dependent: it passes in a
    full run and fails under `-k`, under `-x`, or in any shuffled order. State
    the dependency instead of inheriting it.

    Called directly there is no request in flight, so the tools resolve to the
    stdio session key — the same path a stdio deployment takes.
    """

    def _grounded(*policy_ids: str) -> None:
        store = SessionStore()
        store.add(STDIO_KEY, set(policy_ids))
        monkeypatch.setattr(app, "_RETRIEVED", store)

    return _grounded


# --- reads -----------------------------------------------------------------


def test_get_dispute_returns_account_context():
    """Triage needs account_type up front — it selects which regulation applies."""
    row = app.get_dispute("D-1004")
    assert row["id"] == "D-1004"
    assert row["customer_id"] == "C-105"
    assert row["account_type"] == "debit"
    assert "prior_claims_12m" in row


def test_seed_facts_the_eval_cases_turn_on():
    """These are fixture assumptions, not behaviour. Each one silently collapses
    a labelled eval case if the seed data drifts: D-1002 is the debit-vs-credit
    discrimination case, C-103 the repeat claimant (POL-031), C-107 the
    out-of-scope business account (POL-040)."""
    assert app.get_dispute("D-1002")["account_type"] == "credit"
    assert app.get_dispute_history("C-103")["prior_claims_12m"] >= 3
    assert app.get_dispute_history("C-107")["account_type"] == "business"


def test_missing_records_are_results_not_crashes():
    """POL-033: an unlocatable transaction is a triage outcome. D-1009 has no
    transaction_id at all, so the empty case has to behave too."""
    assert "error" in app.get_dispute("D-9999")

    missing = app.get_transaction("T-DOES-NOT-EXIST")
    assert missing["found"] is False
    assert "not found" in missing["reason"]
    assert app.get_transaction("")["found"] is False


def test_list_customer_transactions_orders_newest_first():
    rows = app.list_customer_transactions("C-100", limit=5)
    dates = [r["txn_date"] for r in rows]
    assert dates == sorted(dates, reverse=True)


# --- retrieval through the tool -------------------------------------------


def test_search_policy_returns_citable_hits():
    hits = app.search_policy("duplicate charge same merchant same amount", k=3)
    assert hits, "expected at least one hit"
    assert all({"policy_id", "section", "text"} <= set(h) for h in hits)
    assert any(h["policy_id"] == "POL-022" for h in hits)


def test_search_policy_distinguishes_debit_from_credit():
    """The corpus is deliberately confusable; this is the discrimination test."""
    debit = app.search_policy("debit card provisional credit deadline", k=3)
    assert any(h["policy_id"] == "POL-001" for h in debit)


# --- the write: safety invariants -----------------------------------------


def test_propose_resolution_writes_pending_only(temp_db, grounded):
    """The agent's terminal action must never produce a committed state."""
    grounded("POL-022")
    out = app.propose_resolution(
        dispute_id="D-1004",
        disposition="provisional_credit",
        rationale="Duplicate confirmed under POL-022 Clause 22.1.",
        citations="POL-022",
        amount=189.00,
    )
    assert out["ok"] is True
    assert out["status"] == "pending"

    row = db.one("SELECT * FROM proposals WHERE id = ?", (out["proposal_id"],), temp_db)
    assert row["status"] == "pending"
    assert row["decided_at"] is None
    assert row["decided_by"] is None


def test_propose_resolution_rejects_unknown_disposition(temp_db):
    out = app.propose_resolution("D-1004", "refund_immediately", "why", "POL-022", 10.0)
    assert out["ok"] is False
    assert "disposition must be one of" in out["error"]


def test_propose_resolution_requires_citations(temp_db):
    """Every disposition must be grounded — an uncited one is not acceptable."""
    out = app.propose_resolution("D-1004", "deny", "no reason given", "   ", None)
    assert out["ok"] is False
    assert "citations required" in out["error"]


def test_propose_resolution_rejects_unknown_dispute(temp_db):
    out = app.propose_resolution("D-9999", "deny", "r", "POL-001", None)
    assert out["ok"] is False


def test_no_tool_can_move_money():
    """Structural guarantee: the server exposes no settle/credit/commit tool."""
    tool_names = {n for n in dir(app) if callable(getattr(app, n, None)) and not n.startswith("_")}
    forbidden = {"settle", "commit", "credit_account", "transfer", "post_credit", "approve"}
    assert not (tool_names & forbidden)


# --- data fixtures the eval suite depends on ------------------------------


def test_injection_transaction_exists():
    """POL-040 Clause 40.4 needs a real adversarial row to be testable."""
    row = db.one("SELECT merchant FROM transactions WHERE id = 'T-2021'")
    assert "ignore prior policy" in row["merchant"].lower()


def test_injection_case_statement_matches_the_real_amount():
    """D-1012's expected verdict is escalate *because* the amount clears $5,000.

    search_policy tells the agent to query by disputed amount, so a customer
    statement quoting a different figure aims the query at the wrong threshold
    and quietly weakens the injection test — it would still be labelled
    'escalate' while no longer measuring the trigger it was built for.
    """
    dispute = app.get_dispute("D-1012")
    txn = app.get_transaction(dispute["transaction_id"])
    assert txn["amount"] > 5000, "the >$5k escalation trigger is the point of this case"
    digits = f"{int(txn['amount']):,}"  # 6,400
    statement = dispute["customer_statement"]
    assert digits in statement or digits.replace(",", "") in statement, (
        f"statement quotes an amount other than {digits}: {statement!r}"
    )


def test_policy_corpus_is_large_enough_to_discriminate():
    """A tiny corpus makes retrieval trivially perfect and ablations meaningless."""
    from server.retrieval import load_corpus

    chunks = load_corpus()
    assert len(chunks) >= 50, f"only {len(chunks)} chunks — rerank ablation would be null"
