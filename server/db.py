"""SQLite access for the dispute desk. Stdlib only.

**Why this is still SQLite.** Serving over HTTP raises the obvious question, and
the answer is that a Postgres migration is not the config change it looks like.
The DSN is the only endpoint-shaped part of it:

    connect       sqlite3.connect(path) — a FILE PATH   ->  psycopg.connect(dsn)
    rows          conn.row_factory = sqlite3.Row        ->  psycopg.rows.dict_row
    placeholders  ?                                     ->  %s, in EVERY statement
    autoincrement INTEGER PRIMARY KEY AUTOINCREMENT     ->  GENERATED ... AS IDENTITY
    now           datetime('now')                       ->  now()
    new row id    cur.lastrowid                         ->  RETURNING id
    schema load   conn.executescript()                  ->  no equivalent; split it
    foreign keys  PRAGMA foreign_keys = ON              ->  on by default
    lifecycle     connect-per-query, fine on a file     ->  needs a pool; per-query
                                                            TCP connects are fatal

Twelve raw statements live outside this module (seven in server/app.py, three in
agent/approval.py, two in the eval scripts), so "swap the driver" would mean
moving those behind named functions first. That is a real refactor for a
migration that is not happening — and Postgres would need Docker, which breaks
the promise that this repo runs free and offline.

WAL is enabled below because concurrent HTTP sessions make reader/writer
contention real, and it is one line rather than an architecture.
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Any

# The MCP server runs as a SEPARATE PROCESS, so monkeypatching this module in
# the parent does not affect it. Anything that needs the server to use a
# different database (the eval harness, for one) must pass DISPUTE_DB through
# the subprocess environment. Learned the hard way: without this, eval runs
# silently wrote to the real database while the harness read an empty temp copy
# and reported "no proposal recorded".
DB_PATH = Path(os.getenv("DISPUTE_DB") or (Path(__file__).parent.parent / "data" / "bank.db"))


def connect(db_path: Path | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path or DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    # Readers no longer block on the writer. Matters once concurrent HTTP
    # sessions can propose while others are searching. Persists in the file, so
    # this is set-once-per-database rather than per-connection work.
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def query(
    sql: str, params: tuple[Any, ...] = (), db_path: Path | None = None
) -> list[dict[str, Any]]:
    conn = connect(db_path)
    try:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    finally:
        conn.close()


def one(
    sql: str, params: tuple[Any, ...] = (), db_path: Path | None = None
) -> dict[str, Any] | None:
    rows = query(sql, params, db_path)
    return rows[0] if rows else None


def insert_proposal(
    dispute_id: str,
    disposition: str,
    rationale: str,
    citations: str,
    amount: float | None,
    db_path: Path | None = None,
) -> int:
    """Write a PENDING proposal. This is the only write the agent can make.

    Deliberately cannot set status: a proposal is always born 'pending' and only
    a human (approve.py) can move it. There is no code path from the agent to a
    committed money movement.
    """
    conn = connect(db_path)
    try:
        cur = conn.execute(
            "INSERT INTO proposals (dispute_id, disposition, rationale, citations, amount) "
            "VALUES (?,?,?,?,?)",
            (dispute_id, disposition, rationale, citations, amount),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def record_verdict(
    proposal_id: int,
    verdict: str,
    reasons: str,
    verified_by: str,
    db_path: Path | None = None,
) -> None:
    """Attach a verifier's review to a proposal.

    Constrained the same way `insert_proposal` is: it touches only the three
    verdict columns, so a verifier cannot change the disposition, the amount or
    the status. The review is advisory — it informs the human gate rather than
    replacing it. `AND verdict IS NULL` makes a double-write a no-op rather than
    an overwrite.
    """
    conn = connect(db_path)
    try:
        conn.execute(
            "UPDATE proposals SET verdict=?, verdict_reasons=?, verified_by=? "
            "WHERE id=? AND verdict IS NULL",
            (verdict, reasons, verified_by, proposal_id),
        )
        conn.commit()
    finally:
        conn.close()
