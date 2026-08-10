"""The deployment entry point: the same MCP server, over authenticated HTTP.

    uvicorn server.http_app:app --port 8000

Nothing in `server/app.py` changed to make this work — same tools, same resource,
same prompt. Only the transport and the credential check are new.

`streamable_http_app()` returns a Starlette app rather than binding its own
server, so this deploys behind a proxy like anything else.

Static tokens are the demo shortcut: `TokenVerifier` is a one-method Protocol, so
a real deployment swaps `StaticTokenVerifier` for JWT or RFC 7662 introspection
and nothing else moves. What is *not* a shortcut is what the token carries —
`subject` is the human a proposal is attributable to, `scopes` are what they may
do.

`AuthSettings.issuer_url` is required (the SDK publishes OAuth
protected-resource metadata), so with static tokens it advertises an authorization
server that does not exist. That is the one place this file pretends.
"""

from __future__ import annotations

import os

from mcp.server.auth.provider import AccessToken, TokenVerifier
from mcp.server.auth.settings import AuthSettings

from server.app import SCOPE_PROPOSE, SCOPE_SEARCH, SCOPE_VERIFY, mcp

ISSUER_URL = os.getenv("MCP_ISSUER_URL", "http://localhost:8000")
RESOURCE_URL = os.getenv("MCP_RESOURCE_URL", "http://localhost:8000")


def _load_tokens() -> dict[str, tuple[str, list[str]]]:
    """token -> (subject, scopes), read from MCP_TOKENS.

    Format: `token:subject:scope|scope,token:subject:scope`. Deliberately awkward
    to write, because it should not survive contact with a real deployment.

    The default pair exists so the HTTP path is runnable offline with no setup,
    and mirrors the two agents. Note what the verifier's token does NOT carry:
    without `dispute:propose` it cannot record a resolution even if the model
    asks for the tool by name. That is what turns the client-side tool allowlist
    in agent/verify.py from a courtesy into a boundary.
    """
    raw = os.getenv("MCP_TOKENS")
    if not raw:
        return {
            "writer-token": ("analyst@northwind.example", [SCOPE_SEARCH, SCOPE_PROPOSE]),
            "verifier-token": ("verifier@northwind.example", [SCOPE_SEARCH, SCOPE_VERIFY]),
        }
    tokens: dict[str, tuple[str, list[str]]] = {}
    for entry in raw.split(","):
        token, subject, scopes = entry.strip().split(":", 2)
        tokens[token] = (subject, [s for s in scopes.split("|") if s])
    return tokens


class StaticTokenVerifier(TokenVerifier):
    """Validates bearer tokens against a configured table.

    Swap this one class for JWT/introspection and the rest of the server is
    unaffected — that is the whole point of `TokenVerifier` being a Protocol.
    """

    def __init__(self, tokens: dict[str, tuple[str, list[str]]] | None = None) -> None:
        self.tokens = tokens if tokens is not None else _load_tokens()

    async def verify_token(self, token: str) -> AccessToken | None:
        found = self.tokens.get(token)
        if found is None:
            return None
        subject, scopes = found
        return AccessToken(
            token=token,
            client_id=subject,
            scopes=scopes,
            subject=subject,
            resource=RESOURCE_URL,
        )


def build_app(verifier: TokenVerifier | None = None):
    """Build the authenticated ASGI app.

    Rebuilding `MCPServer` here rather than mutating the module-level `mcp` keeps
    the stdio entry point (`python -m server.app`) completely unauthenticated and
    untouched, which is what the existing test suite and `mcp dev` rely on.
    """
    mcp.settings.auth = AuthSettings(
        issuer_url=ISSUER_URL,
        resource_server_url=RESOURCE_URL,
        required_scopes=[SCOPE_SEARCH],
    )
    mcp._token_verifier = verifier or StaticTokenVerifier()
    return mcp.streamable_http_app()


app = build_app()
