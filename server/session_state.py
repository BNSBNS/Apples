"""Per-session grounding state, and the scope check.

Both exist because the server stopped being a per-run subprocess. Under stdio a
module-level set was correct; under HTTP one process serves many callers, and
that same set leaks — B could cite what A retrieved.

The key is the `Mcp-Session-Id` header, which is what the SDK's own
`StreamableHTTPSessionManager` routes on. Not `ctx.session`: the SDK builds one
of those **per request**, so keying on it yields an always-empty set and the
grounding gate then rejects every citation.

Limitation worth knowing: in-process session state means this server is stateful
and won't scale horizontally without sticky sessions or shared storage.
"""

from __future__ import annotations

import os
import time

MCP_SESSION_HEADER = "mcp-session-id"

# Stdio has no session header and is single-tenant by construction: the process
# belongs to one agent run. One fixed key keeps that path on the same code as
# the HTTP path rather than branching around it.
STDIO_KEY = "__stdio__"

# The SDK evicts its own transports on `session_idle_timeout`; it does not know
# about ours, so this map needs its own bound or a long-lived server leaks one
# entry per session forever.
TTL_SECONDS = float(os.getenv("MCP_SESSION_TTL", "3600"))
MAX_SESSIONS = int(os.getenv("MCP_MAX_SESSIONS", "1000"))


class SessionStore:
    """Session id -> the set of policy ids search_policy has handed that caller.

    Deliberately not a `WeakKeyDictionary`: there is no long-lived object per
    session to hang it off (see the module docstring), so expiry is explicit.
    """

    def __init__(self, ttl: float = TTL_SECONDS, max_sessions: int = MAX_SESSIONS) -> None:
        self.ttl = ttl
        self.max_sessions = max_sessions
        self._data: dict[str, set[str]] = {}
        self._touched: dict[str, float] = {}

    def _drop(self, key: str) -> None:
        self._data.pop(key, None)
        self._touched.pop(key, None)

    def _expire(self, now: float) -> None:
        # `>=` rather than `>`: time.monotonic() has finite resolution, so two
        # calls in the same tick read equal. With `>` a zero TTL never expires
        # anything, which is both surprising and untestable.
        for key in [k for k, t in self._touched.items() if now - t >= self.ttl]:
            self._drop(key)

    def _enforce_bound(self, current: str) -> None:
        """Bound by count as well as by age — a flood of short-lived sessions
        outruns any TTL. Oldest first, and never the session being served."""
        while len(self._data) > self.max_sessions:
            candidates = [k for k in self._touched if k != current]
            if not candidates:
                return
            self._drop(min(candidates, key=lambda k: self._touched[k]))

    def get(self, key: str) -> set[str]:
        now = time.monotonic()
        self._expire(now)
        self._touched[key] = now
        bucket = self._data.setdefault(key, set())
        # After insertion, not before: evicting first leaves the map at
        # max_sessions + 1 once the new key lands.
        self._enforce_bound(key)
        return bucket

    def add(self, key: str, policy_ids: set[str]) -> None:
        self.get(key).update(policy_ids)

    def clear(self) -> None:
        self._data.clear()
        self._touched.clear()

    def __len__(self) -> int:
        return len(self._data)


def session_key(ctx: object | None) -> str:
    """The partition key for this request: the MCP session id, or the stdio key.

    `ctx` is an `mcp.server.mcpserver.Context` when the SDK injected one. It is
    typed loosely and read defensively because the tools are also called
    directly as plain functions by the test suite, where there is no request.
    """
    headers = getattr(ctx, "headers", None) if ctx is not None else None
    if not headers:
        return STDIO_KEY
    for name, value in headers.items():
        if name.lower() == MCP_SESSION_HEADER and value:
            return value
    return STDIO_KEY


def caller_scopes() -> set[str] | None:
    """Scopes on the caller's bearer token, or None when unauthenticated (stdio).

    Imported lazily: the auth module pulls in the HTTP stack, and the stdio path
    must not require it.
    """
    try:
        from mcp.server.auth.middleware.auth_context import get_access_token
    except Exception:  # pragma: no cover - auth extras absent
        return None
    token = get_access_token()
    return set(token.scopes) if token else None


def require_scope(scope: str) -> str | None:
    """Return an error message if the caller may not use this tool, else None.

    One code path, deliberately. The obvious alternative — `if token: check()` —
    means the branch that actually guards production is the branch tests never
    execute. Here the stdio path resolves to a *configured* scope set
    (`MCP_STDIO_SCOPES`, defaulting to everything, because a stdio server is a
    single-tenant subprocess owned by its caller), so a test can drive a denial
    through exactly the code that runs under HTTP.
    """
    scopes = caller_scopes()
    if scopes is None:
        configured = os.getenv("MCP_STDIO_SCOPES")
        if configured is None:
            return None  # stdio, unrestricted by default
        scopes = {s.strip() for s in configured.split(",") if s.strip()}
    if scope in scopes:
        return None
    return (
        f"permission denied: this tool requires the {scope!r} scope. "
        f"The credential presented grants: {sorted(scopes) or 'nothing'}."
    )
