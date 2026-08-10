"""The MCP -> OpenAI schema bridge.

This exists because of the single most expensive bug in the project.

The MCP `Tool` attribute is `input_schema`. `inputSchema` is only the wire
alias and reading it yields nothing. The original code did
`getattr(t, "inputSchema", None) or {"type": "object", "properties": {}}`,
which meant **every tool was advertised to the model with no parameters at
all**. Models then guessed argument names, guessed wrong, and the failures
looked precisely like model incompetence — which is what was initially
concluded from them.

The lesson is about the `or {}` more than the attribute name: a fallback that
silently substitutes an empty value turns a wiring bug into a plausible story
about something else. These tests assert the schema actually arrives.
"""

from __future__ import annotations

import asyncio
import os
import sys

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from agent.loop_manual import mcp_tools_to_openai, tool_schema


class _FakeTool:
    def __init__(self, name: str, schema: dict | None, attr: str = "input_schema") -> None:
        self.name = name
        self.description = "d"
        if schema is not None:
            setattr(self, attr, schema)


SCHEMA = {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}


def test_reads_snake_case_attribute():
    assert tool_schema(_FakeTool("t", SCHEMA)) == SCHEMA


def test_falls_back_to_camel_case_alias():
    """Tolerate the alias if a future SDK exposes only that."""
    assert tool_schema(_FakeTool("t", SCHEMA, attr="inputSchema")) == SCHEMA


def test_missing_schema_raises_instead_of_returning_empty():
    """The whole point: fail loudly rather than advertise a parameterless tool."""
    with pytest.raises(RuntimeError, match="no input schema"):
        tool_schema(_FakeTool("t", None))


def test_conversion_preserves_parameters():
    converted = mcp_tools_to_openai([_FakeTool("t", SCHEMA)])
    assert converted[0]["function"]["parameters"] == SCHEMA


# --- against the real server ----------------------------------------------


def _live_tools():
    async def go():
        params = StdioServerParameters(
            command=sys.executable, args=["-m", "server.app"], cwd=os.getcwd(), env=dict(os.environ)
        )
        async with stdio_client(params) as (r, w), ClientSession(r, w) as s:
            await s.initialize()
            return (await s.list_tools()).tools

    return asyncio.run(go())


def test_real_server_tools_all_expose_parameters():
    """Every real tool must reach the model with its actual parameters.

    Named rather than counted: a bare `len(...) == 7` fails informatively only by
    accident, and the point is *which* tools exist — the writer's terminal action
    and the verifier's are different tools on purpose.
    """
    converted = mcp_tools_to_openai(_live_tools())
    assert {t["function"]["name"] for t in converted} == {
        "get_dispute",
        "get_transaction",
        "list_customer_transactions",
        "get_dispute_history",
        "search_policy",
        "propose_resolution",
        "record_verdict",
    }

    for tool in converted:
        params = tool["function"]["parameters"]
        assert params.get("properties"), f"{tool['function']['name']} advertised no parameters"


def test_known_parameter_names_survive_the_bridge():
    """Spot-check the exact names models were getting wrong."""
    by_name = {
        t["function"]["name"]: t["function"]["parameters"]
        for t in mcp_tools_to_openai(_live_tools())
    }

    assert "transaction_id" in by_name["get_transaction"]["properties"]
    assert "dispute_id" in by_name["get_dispute"]["properties"]
    assert "query" in by_name["search_policy"]["properties"]

    required = by_name["propose_resolution"].get("required", [])
    assert {"dispute_id", "disposition", "rationale", "citations"} <= set(required)
