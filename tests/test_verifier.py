"""The second agent, tested without a model.

Two claims are worth guarding here, and neither needs an LLM: the verifier
cannot reach the tool that writes money decisions, and its review annotates the
proposal rather than altering it. Whether the verifier is any *good* is a
measurement, not an assertion — see evals/verifier_eval.py.
"""

from __future__ import annotations

import json

import pytest

from agent import verify
from data import seed
from server import app, db
from server.session_state import STDIO_KEY, SessionStore


@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    path = tmp_path / "bank.db"
    seed.build(path)
    monkeypatch.setattr(db, "DB_PATH", path)
    store = SessionStore()
    store.add(STDIO_KEY, {"POL-022"})
    monkeypatch.setattr(app, "_RETRIEVED", store)
    return path


@pytest.fixture()
def proposed(temp_db):
    """A pending proposal for the verifier to review."""
    out = app.propose_resolution("D-1004", "provisional_credit", "duplicate", "POL-022", 189.0)
    assert out["ok"] is True
    return out["proposal_id"]


def test_verifier_cannot_reach_the_write_tool():
    """The multi-agent boundary, stated structurally.

    The client-side allowlist stops the model asking for it; under HTTP the
    verifier's token also lacks `dispute:propose`, so the server refuses even if
    it does. This asserts the first half — the second is in test_http_auth.py.
    """
    assert "propose_resolution" not in verify.VERIFIER_TOOLS
    assert "record_verdict" in verify.VERIFIER_TOOLS
    assert (
        verify.verifier_spec(
            {
                "dispute_id": "D-1004",
                "disposition": "deny",
                "amount": None,
                "citations": "POL-022",
                "rationale": "r",
            }
        ).terminal_tool
        == "record_verdict"
    )


def test_verdict_annotates_without_altering_the_proposal(proposed):
    """A verifier must not be able to change what the analyst decided — only to
    disagree with it in a field the human can see."""
    before = db.one("SELECT * FROM proposals WHERE id = ?", (proposed,))

    out = app.record_verdict("D-1004", "fail", "POL-030 governs a partial dispense, not POL-022.")
    assert out["ok"] is True

    after = db.one("SELECT * FROM proposals WHERE id = ?", (proposed,))
    assert after["verdict"] == "fail"
    assert "POL-030" in after["verdict_reasons"]
    # The decision itself is untouched.
    for field in ("disposition", "amount", "status", "citations", "rationale"):
        assert after[field] == before[field]


@pytest.mark.parametrize(
    ("dispute_id", "verdict", "reasons", "expected"),
    [
        ("D-1004", "maybe", "unsure", "verdict must be one of"),
        ("D-1004", "fail", "   ", "reasons required"),
        ("D-1011", "pass", "fine", "no pending proposal"),
    ],
    ids=["unknown-verdict", "empty-reasons", "nothing-to-review"],
)
def test_record_verdict_rejects(proposed, dispute_id, verdict, reasons, expected):
    out = app.record_verdict(dispute_id, verdict, reasons)
    assert out["ok"] is False
    assert expected in out["error"]


def test_second_verdict_is_rejected(proposed):
    """Same rule as one-proposal-per-dispute: a second call is the model losing
    track, not a considered revision."""
    assert app.record_verdict("D-1004", "pass", "looks right")["ok"] is True

    second = app.record_verdict("D-1004", "fail", "changed my mind")
    assert second["ok"] is False
    assert "already carries" in second["error"]
    assert db.one("SELECT verdict FROM proposals WHERE id = ?", (proposed,))["verdict"] == "pass"


def test_verdict_is_read_from_the_result_not_the_attempt():
    """A rejected `record_verdict` wrote nothing, so it is not a verdict — the
    same distinction `recorded_a_decision` makes in the loop."""
    ok = {"tool": "record_verdict", "result": json.dumps({"ok": True, "verdict": "fail"})}
    refused = {"tool": "record_verdict", "result": json.dumps({"ok": False, "error": "no"})}

    assert verify.verdict_from_trace([ok]) == "fail"
    assert verify.verdict_from_trace([refused]) is None
    assert verify.verdict_from_trace([refused, ok]) == "fail"
    assert verify.verdict_from_trace([]) is None
