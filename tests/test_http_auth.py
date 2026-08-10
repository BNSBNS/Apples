"""The security surface that only exists over HTTP.

Under stdio the server is a subprocess owned by one caller, so none of this
mattered: there was no credential to check and no second tenant to leak to.
Serving one process to many callers turns both into real questions, and this
file is where they get answered.

The isolation test is the one that earns its place. A cross-session grounding
leak passes every manual check — one agent run looks perfect — and only shows up
when two callers overlap, which is exactly when it matters.
"""

from __future__ import annotations

import asyncio

import httpx2
import pytest

from data import seed
from server import app as server_app
from server import db
from server.http_app import StaticTokenVerifier, build_app
from server.session_state import STDIO_KEY, SessionStore, require_scope, session_key


class FakeContext:
    """Stands in for the MCP-injected Context, which only exists mid-request."""

    def __init__(self, session_id: str | None) -> None:
        self.headers = {"mcp-session-id": session_id} if session_id else {}


@pytest.fixture()
def fresh_store(monkeypatch):
    store = SessionStore()
    monkeypatch.setattr(server_app, "_RETRIEVED", store)
    return store


@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    """Any test that reaches propose_resolution writes a row; without this it
    writes into the working database, which test_no_db_pollution.py rightly
    fails the build over."""
    path = tmp_path / "bank.db"
    seed.build(path)
    monkeypatch.setattr(db, "DB_PATH", path)
    return path


# --- the transport actually rejects anonymous callers ----------------------


@pytest.mark.parametrize("token", [None, "not-a-real-token"], ids=["absent", "invalid"])
def test_request_without_a_valid_token_is_rejected(token):
    """401 before any MCP method runs — the credential check is transport-level."""

    async def go() -> int:
        headers = {"accept": "application/json, text/event-stream"}
        if token:
            headers["authorization"] = f"Bearer {token}"
        transport = httpx2.ASGITransport(app=build_app())
        async with httpx2.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, headers=headers
            )
            return resp.status_code

    assert asyncio.run(go()) == 401


def test_token_table_grants_the_two_agent_roles():
    """The writer may propose; the verifier may only read.

    This is what makes Stage 2's tool allowlist a real boundary rather than a
    client-side courtesy: the verifier's credential cannot propose even if the
    model asks for the tool by name.
    """
    verifier = StaticTokenVerifier()
    writer = asyncio.run(verifier.verify_token("writer-token"))
    reader = asyncio.run(verifier.verify_token("verifier-token"))

    assert server_app.SCOPE_PROPOSE in writer.scopes
    assert server_app.SCOPE_VERIFY not in writer.scopes

    # The verifier may review but not propose — this is what makes the tool
    # allowlist in agent/verify.py a boundary rather than a courtesy.
    assert server_app.SCOPE_PROPOSE not in reader.scopes
    assert {server_app.SCOPE_SEARCH, server_app.SCOPE_VERIFY} <= set(reader.scopes)

    # subject is what makes a proposal attributable to a person.
    assert writer.subject and "@" in writer.subject


# --- scope enforcement runs the same code in both modes --------------------


def test_stdio_scopes_default_open_and_can_be_restricted(monkeypatch):
    """Unrestricted by default — a stdio server is a subprocess owned by its
    caller. The point of keeping one code path is the second half: a test drives
    denial through the same branch that guards production, not a stdio-only stub.
    """
    monkeypatch.delenv("MCP_STDIO_SCOPES", raising=False)
    assert require_scope(server_app.SCOPE_PROPOSE) is None

    monkeypatch.setenv("MCP_STDIO_SCOPES", server_app.SCOPE_SEARCH)
    assert require_scope(server_app.SCOPE_SEARCH) is None
    denied = require_scope(server_app.SCOPE_PROPOSE)
    assert denied is not None and server_app.SCOPE_PROPOSE in denied


def test_tools_refuse_when_the_scope_is_missing(monkeypatch, fresh_store, temp_db):
    monkeypatch.setenv("MCP_STDIO_SCOPES", server_app.SCOPE_SEARCH)

    out = server_app.propose_resolution("D-1004", "escalate", "r", "POL-022", None)
    assert out["ok"] is False
    assert "permission denied" in out["error"]

    # ...and the read tool, which the same credential does allow, still works.
    assert server_app.search_policy("duplicate charge", k=2)[0].get("error") is None


# --- the leak this stage exists to prevent ---------------------------------


def test_two_sessions_do_not_share_retrieved_policies(fresh_store, temp_db):
    """Session A's search must not ground session B's citation.

    With one module-level set this passes silently in single-agent testing and
    is a cross-tenant grounding failure in production: B cites a policy it never
    read, and the gate that exists to catch exactly that waves it through.
    """
    server_app.search_policy("duplicate charge same merchant", k=3, ctx=FakeContext("session-a"))

    assert fresh_store.get("session-a"), "A retrieved something"
    assert fresh_store.get("session-b") == set(), "B retrieved nothing and must inherit nothing"

    grounded = server_app.propose_resolution(
        "D-1004", "provisional_credit", "r", "POL-022", 189.0, ctx=FakeContext("session-a")
    )
    assert grounded["ok"] is True

    leaked = server_app.propose_resolution(
        "D-1011", "provisional_credit", "r", "POL-022", 58.0, ctx=FakeContext("session-b")
    )
    assert leaked["ok"] is False
    assert "ungrounded citation" in leaked["error"]


def test_missing_session_header_falls_back_to_the_stdio_key():
    assert session_key(None) == STDIO_KEY
    assert session_key(FakeContext(None)) == STDIO_KEY
    assert session_key(FakeContext("abc123")) == "abc123"


# --- the map must not grow forever -----------------------------------------


def test_expired_sessions_are_evicted():
    store = SessionStore(ttl=0.0)
    store.add("old", {"POL-001"})
    assert store.get("new") == set()
    assert len(store) == 1, "the expired session should be gone, not merely stale"


def test_session_count_is_bounded():
    """TTL alone is not enough — a flood of short-lived sessions outruns it."""
    store = SessionStore(ttl=3600, max_sessions=5)
    for i in range(50):
        store.add(f"session-{i}", {"POL-001"})
    assert len(store) <= 5
