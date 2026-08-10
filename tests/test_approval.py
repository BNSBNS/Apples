"""The human gate's CLI.

This is the only thing in the system that can commit a decision, so an ambiguous
command must stop, never guess.
"""

from __future__ import annotations

import pytest

from agent import approval
from data import seed
from server import db


@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    path = tmp_path / "bank.db"
    seed.build(path)
    monkeypatch.setattr(db, "DB_PATH", path)
    return path


def _pending(dispute_id: str = "D-1004") -> int:
    return db.insert_proposal(dispute_id, "provisional_credit", "r", "POL-022", 189.0)


def test_approve_and_deny_together_is_refused(temp_db, monkeypatch):
    """`--approve 1 --deny 2` used to approve 1 and drop the deny silently."""
    pid = _pending()
    monkeypatch.setattr(
        "sys.argv", ["approval", "--approve", str(pid), "--deny", "2", "--by", "tester"]
    )

    with pytest.raises(SystemExit) as exc:
        approval.main()
    assert "mutually exclusive" in str(exc.value)

    row = db.one("SELECT status FROM proposals WHERE id = ?", (pid,))
    assert row["status"] == "pending", "an ambiguous command must decide nothing"


def test_deciding_requires_a_person(temp_db, monkeypatch):
    pid = _pending()
    monkeypatch.setattr("sys.argv", ["approval", "--approve", str(pid)])

    with pytest.raises(SystemExit) as exc:
        approval.main()
    assert "--by is required" in str(exc.value)
    assert db.one("SELECT status FROM proposals WHERE id = ?", (pid,))["status"] == "pending"


def test_approve_commits_with_attribution(temp_db):
    pid = _pending()
    out = approval.decide(pid, "approved", "tester")
    assert out["ok"] is True

    row = db.one("SELECT * FROM proposals WHERE id = ?", (pid,))
    assert row["status"] == "approved"
    assert row["decided_by"] == "tester"
    assert row["decided_at"] is not None


def test_already_decided_proposal_is_not_overwritten(temp_db):
    pid = _pending()
    approval.decide(pid, "approved", "first")
    out = approval.decide(pid, "denied", "second")

    assert out["ok"] is False
    assert "refusing to overwrite" in out["error"]
    assert db.one("SELECT * FROM proposals WHERE id = ?", (pid,))["decided_by"] == "first"
