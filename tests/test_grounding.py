"""Grounding and isolation guarantees.

Every test here exists because a live model actually did the thing being
guarded against. They are regression tests, not hypotheticals.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest

from data import seed
from server import app, db
from server.session_state import STDIO_KEY


def test_citations_are_extracted_whatever_the_model_wrapped_them_in(temp_db, fresh_retrieval):
    """`citations` is typed as a plain string; a model does not reliably respect
    that. llama3.2 passed a JSON array rendered as a string, and `.split(",")`
    sliced through the quotes producing '["POL-030"' — so a real citation was
    rejected as unknown. Extracting by pattern sidesteps the wrapper entirely.
    """
    assert app.parse_citations(" POL-001 ; pol-031 ") == {"POL-001", "POL-031"}
    assert app.parse_citations('["POL-030", "POL-031"]') == {"POL-030", "POL-031"}
    assert app.parse_citations("no policy applies here") == set()

    # And through the real tool, not the parser alone.
    app.search_policy("duplicate charge same merchant same amount", k=4)
    assert app.propose_resolution("D-1004", "provisional_credit", "r", '["POL-022"]', 189.0)["ok"]


@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    path = tmp_path / "bank.db"
    seed.build(path)
    monkeypatch.setattr(db, "DB_PATH", path)
    return path


def test_known_policy_ids_matches_corpus():
    ids = app.known_policy_ids()
    assert "POL-001" in ids and "POL-040" in ids
    assert len(ids) == 15


@pytest.fixture()
def fresh_retrieval():
    """Reset the server's record of what it has retrieved.

    `_RETRIEVED` is now partitioned by MCP session. Called directly like this
    there is no request in flight, so everything lands under the stdio key —
    which is exactly the path a stdio deployment takes.
    """
    app._RETRIEVED.clear()
    yield app._RETRIEVED
    app._RETRIEVED.clear()


def _ground(*policy_ids: str) -> None:
    """Mark policy ids as retrieved by the current (stdio) session."""
    app._RETRIEVED.add(STDIO_KEY, set(policy_ids))


def test_invented_citation_is_rejected(temp_db, fresh_retrieval):
    """qwen2.5:7b invented POL-055 and POL-123 after a failed search and cited
    them confidently. The server must refuse ids that do not exist."""
    out = app.propose_resolution(
        "D-1004", "provisional_credit", "Looks like a duplicate.", "POL-055,POL-123", 189.0
    )
    assert out["ok"] is False
    assert "unknown policy id" in out["error"]
    assert "POL-055" in out["error"]


def test_error_does_not_leak_the_valid_id_list(temp_db, fresh_retrieval):
    """An earlier version listed every valid id in the rejection message, which
    handed the model a menu to pick a passing citation from."""
    out = app.propose_resolution("D-1004", "escalate", "r", "POL-999", None)
    assert out["ok"] is False
    # At most the offending id should appear — not the corpus.
    assert "POL-022" not in out["error"]
    assert "POL-001" not in out["error"]


def test_partially_invented_citation_is_rejected(temp_db, fresh_retrieval):
    """One real id does not launder a fake one."""
    out = app.propose_resolution("D-1004", "escalate", "r", "POL-022,POL-999", None)
    assert out["ok"] is False
    assert "POL-999" in out["error"]


def test_real_but_unretrieved_citation_is_rejected(temp_db, fresh_retrieval):
    """The actual observed failure: right answer, real citation, never read it.

    POL-032 exists, so an existence check passes it. But the agent never
    retrieved it, so the citation is not grounded and must be refused.
    """
    out = app.propose_resolution("D-1001", "provisional_credit", "r", "POL-032", 842.5)
    assert out["ok"] is False
    assert "ungrounded citation" in out["error"]
    assert "POL-032" in out["error"]
    assert "not called search_policy yet" in out["error"]


def test_citation_is_accepted_once_retrieved(temp_db, fresh_retrieval):
    """Search first, then cite what came back — the intended path."""
    hits = app.search_policy("duplicate charge same merchant same amount", k=4)
    retrieved = {h["policy_id"] for h in hits}
    assert "POL-022" in retrieved, "fixture assumption: this query should surface POL-022"

    out = app.propose_resolution("D-1004", "provisional_credit", "r", "POL-022", 189.0)
    assert out["ok"] is True


def test_citation_matching_is_case_and_space_tolerant(temp_db, fresh_retrieval):
    """Don't reject a grounded citation over whitespace or casing."""
    app.search_policy("duplicate charge same merchant same amount", k=4)
    _ground("POL-031")
    out = app.propose_resolution("D-1004", "escalate", "r", " pol-022 , POL-031 ", None)
    assert out["ok"] is True


def test_second_proposal_for_same_dispute_is_rejected(temp_db, fresh_retrieval):
    """The agent called propose_resolution twice on one dispute. Once only."""
    app.search_policy("duplicate charge same merchant same amount", k=4)
    first = app.propose_resolution("D-1004", "provisional_credit", "r", "POL-022", 189.0)
    assert first["ok"] is True

    # Cite a *grounded* policy so this reaches the duplicate check rather than
    # tripping the grounding gate first — the checks are ordered.
    second = app.propose_resolution("D-1004", "escalate", "changed my mind", "POL-022", None)
    assert second["ok"] is False
    assert "already exists" in second["error"]

    rows = db.query("SELECT * FROM proposals WHERE dispute_id = 'D-1004'", (), temp_db)
    assert len(rows) == 1, "a second proposal must not be written"


def test_server_subprocess_honours_dispute_db_env(tmp_path):
    """The isolation bug that broke the eval harness.

    The MCP server runs in its own process, so patching db.DB_PATH in the parent
    does nothing. It must pick the database up from DISPUTE_DB instead —
    otherwise eval runs write to the real database while the harness reads an
    empty temp copy and reports 'no proposal recorded'.
    """
    alt = tmp_path / "alt.db"
    seed.build(alt)

    env = dict(os.environ, DISPUTE_DB=str(alt))
    result = subprocess.run(
        [sys.executable, "-c", "from server import db; print(db.DB_PATH)"],
        capture_output=True,
        text=True,
        env=env,
        cwd=os.getcwd(),
        check=True,
    )
    assert str(alt) in result.stdout.strip()


def test_default_db_path_when_env_unset(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "DISPUTE_DB"}
    result = subprocess.run(
        [sys.executable, "-c", "from server import db; print(db.DB_PATH)"],
        capture_output=True,
        text=True,
        env=env,
        cwd=os.getcwd(),
        check=True,
    )
    assert result.stdout.strip().endswith("bank.db")
