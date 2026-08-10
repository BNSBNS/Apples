"""Guard: running the test suite must never write to the working database.

This existed as a real leak. `tests/test_loop.py` drives the MCP server over
stdio; the server is a separate process, so a `db.DB_PATH` monkeypatch in the
parent does nothing and any scripted `propose_resolution` call wrote a row
straight into `data/bank.db`.

Cheap to assert, and it fails loudly if someone adds a test that forgets to
isolate the subprocess.
"""

from __future__ import annotations

from server import db


def test_working_database_has_no_test_residue():
    """The seeded database ships with zero proposals; tests must leave it so."""
    rows = db.query("SELECT dispute_id, disposition, rationale FROM proposals")
    assert rows == [], (
        f"{len(rows)} proposal(s) leaked into the working database: {rows}. "
        "A test wrote through the MCP subprocess without setting DISPUTE_DB."
    )
