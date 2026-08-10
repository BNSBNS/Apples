"""Prove the agent loop is correct independently of model quality.

A real LLM is non-deterministic and, on a 3B local model, frequently wrong.
That makes it useless for testing *the loop*. So we stub the model with a
scripted sequence of replies and assert on the mechanics that actually break
hand-written loops:

  - JSON-string arguments get parsed
  - every tool_call_id receives exactly one matching `tool` message
  - malformed JSON is fed back instead of crashing
  - a tool call written as prose is detected, not silently accepted
  - the iteration cap holds

These run offline with no model and no API key.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import pytest

from agent import loop_manual
from data import seed

# --- minimal stand-ins for the OpenAI response objects --------------------


@dataclass
class _Fn:
    name: str
    arguments: str


@dataclass
class _Call:
    id: str
    function: _Fn
    type: str = "function"


@dataclass
class _Msg:
    content: str | None = None
    tool_calls: list[_Call] | None = None


@dataclass
class _Choice:
    message: _Msg


@dataclass
class _Reply:
    choices: list[_Choice]
    usage: Any = None


class ScriptedCompletions:
    """Replays a fixed list of replies and records what it was sent.

    `create` is async because the loop now drives `AsyncOpenAI` — a synchronous
    client would block the event loop it shares with the MCP session, which is
    harmless with one request in flight and not harmless under concurrent HTTP
    sessions. The stub mirrors the real SDK rather than the other way round.
    """

    def __init__(self, script: list[_Msg]) -> None:
        self.script = script
        self.seen: list[list[dict[str, Any]]] = []

    async def create(self, **kwargs: Any) -> _Reply:
        self.seen.append(list(kwargs["messages"]))
        msg = self.script[min(len(self.seen) - 1, len(self.script) - 1)]
        return _Reply(choices=[_Choice(message=msg)])


class ScriptedClient:
    """Quacks like `AsyncOpenAI` for the one call the loop makes."""

    def __init__(self, script: list[_Msg]) -> None:
        self.completions = ScriptedCompletions(script)
        self.chat = SimpleNamespace(completions=self.completions)


@dataclass
class _StubProvider:
    client: Any
    name: str = "stub"
    model: str = "stub-model"


@pytest.fixture(autouse=True)
def isolate_server_db(tmp_path, monkeypatch):
    """Point the MCP *subprocess* at a throwaway database.

    These tests drive the real server over stdio, so any test whose script
    calls propose_resolution writes a row. Without this fixture those rows land
    in the working database — which is exactly what happened before it existed.
    Patching db.DB_PATH is not enough; the server is a separate process and
    only reads DISPUTE_DB from the environment.
    """
    path = tmp_path / "test.db"
    seed.build(path)
    monkeypatch.setenv("DISPUTE_DB", str(path))
    return path


def _install(monkeypatch: pytest.MonkeyPatch, script: list[_Msg]) -> ScriptedClient:
    client = ScriptedClient(script)
    monkeypatch.setattr(
        loop_manual, "get_provider", lambda kind, model_override=None: _StubProvider(client)
    )
    return client


# --- tests -----------------------------------------------------------------


def test_happy_path_pairs_every_tool_call_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two parallel calls in one turn must produce two `tool` messages."""
    script = [
        _Msg(
            tool_calls=[
                _Call("call_a", _Fn("get_dispute", json.dumps({"dispute_id": "D-1004"}))),
                _Call("call_b", _Fn("get_transaction", json.dumps({"transaction_id": "T-2012"}))),
            ]
        ),
        _Msg(content="Done — proposal recorded."),
    ]
    client = _install(monkeypatch, script)

    out = asyncio.run(loop_manual.run("D-1004", "stub", verbose=False))
    assert out["final_text"] == "Done — proposal recorded."

    # Second request must carry the assistant turn plus one tool msg per id.
    sent = client.completions.seen[1]
    tool_msgs = [m for m in sent if m.get("role") == "tool"]
    assert {m["tool_call_id"] for m in tool_msgs} == {"call_a", "call_b"}

    # And the real data made it back from the MCP server, not a stub.
    joined = " ".join(m["content"] for m in tool_msgs)
    assert "D-1004" in joined and "Halcyon Apparel" in joined


def test_malformed_json_arguments_are_fed_back(monkeypatch: pytest.MonkeyPatch) -> None:
    """A bad-JSON argument string must not crash the loop."""
    script = [
        _Msg(tool_calls=[_Call("bad", _Fn("get_dispute", "{not valid json"))]),
        _Msg(content="recovered"),
    ]
    client = _install(monkeypatch, script)

    out = asyncio.run(loop_manual.run("D-1004", "stub", verbose=False))
    assert out["final_text"] == "recovered"

    tool_msg = next(m for m in client.completions.seen[1] if m.get("role") == "tool")
    assert tool_msg["tool_call_id"] == "bad"
    assert "not valid JSON" in tool_msg["content"]


def test_unknown_tool_returns_error_and_schema_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Calling a tool that doesn't exist is answered, not raised."""
    script = [
        _Msg(tool_calls=[_Call("x", _Fn("no_such_tool", "{}"))]),
        _Msg(content="ok"),
    ]
    client = _install(monkeypatch, script)

    asyncio.run(loop_manual.run("D-1004", "stub", verbose=False))
    tool_msg = next(m for m in client.completions.seen[1] if m.get("role") == "tool")
    # The model must learn both that the tool is unknown and what does exist.
    assert "no tool called" in tool_msg["content"]
    assert "get_dispute" in tool_msg["content"]


def test_wrong_argument_name_is_repaired(monkeypatch: pytest.MonkeyPatch) -> None:
    """The exact llama3.2/qwen failure: `id` instead of `dispute_id`.

    Now that tools reach the model with real schemas, the loop repairs this
    unambiguous rename in-flight and the call succeeds, rather than bouncing
    an error back and hoping the model reads it.
    """
    script = [
        _Msg(tool_calls=[_Call("w", _Fn("get_dispute", json.dumps({"id": "D-1004"})))]),
        _Msg(content="ok"),
    ]
    client = _install(monkeypatch, script)

    out = asyncio.run(loop_manual.run("D-1004", "stub", verbose=False))
    assert out["repairs"] == ["get_dispute: 'id' -> 'dispute_id'"]

    tool_msg = next(m for m in client.completions.seen[1] if m.get("role") == "tool")
    # The repaired call returned real data, not a validation error.
    assert "C-105" in tool_msg["content"]
    assert "HINT" not in tool_msg["content"]


def test_ambiguous_bad_arguments_still_get_the_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Repair is deliberately narrow. Two unknown keys is not a typo, so the
    call must fail and the model must be handed the real schema."""
    script = [
        _Msg(
            tool_calls=[_Call("w", _Fn("get_dispute", json.dumps({"foo": "D-1004", "bar": "x"})))]
        ),
        _Msg(content="ok"),
    ]
    client = _install(monkeypatch, script)

    out = asyncio.run(loop_manual.run("D-1004", "stub", verbose=False))
    assert out["repairs"] == []

    tool_msg = next(m for m in client.completions.seen[1] if m.get("role") == "tool")
    assert "HINT" in tool_msg["content"]
    assert "dispute_id" in tool_msg["content"]


def test_textual_tool_call_is_detected() -> None:
    """Prose that merely describes a call must not be accepted as an answer."""
    assert loop_manual.looks_like_textual_tool_call(
        '{"name": "get_dispute", "parameters": {"id": "D-1004"}}'
    )
    assert loop_manual.looks_like_textual_tool_call(
        'I will call {"name": "search_policy", "arguments": {"query": "x"}}'
    )
    assert not loop_manual.looks_like_textual_tool_call("Proposal recorded for D-1004.")
    assert not loop_manual.looks_like_textual_tool_call("")


def test_textual_tool_call_triggers_correction(monkeypatch: pytest.MonkeyPatch) -> None:
    """A described call gets pushed back on, then the model can finish."""
    script = [
        _Msg(content='{"name": "get_dispute", "parameters": {"id": "D-1004"}}'),
        _Msg(content="Actually finished now."),
    ]
    client = _install(monkeypatch, script)

    out = asyncio.run(loop_manual.run("D-1004", "stub", verbose=False))
    assert out["final_text"] == "Actually finished now."

    corrections = [
        m
        for m in client.completions.seen[1]
        if m.get("role") == "user" and "did not execute" in m.get("content", "")
    ]
    assert len(corrections) == 1


def test_nudge_fires_when_agent_wont_decide(monkeypatch: pytest.MonkeyPatch) -> None:
    """A model that keeps searching must be told to decide.

    Reproduces the observed qwen2.5:7b failure: evidence gathered, terminal
    action never taken. Without the nudge the loop burns to MAX_STEPS and
    returns nothing.
    """
    forever_searching = _Msg(
        tool_calls=[_Call("s", _Fn("search_policy", json.dumps({"query": "duplicate"})))]
    )
    client = _install(monkeypatch, [forever_searching])

    asyncio.run(loop_manual.run("D-1004", "stub", verbose=False))

    nudges = [
        m
        for req in client.completions.seen
        for m in req
        if m.get("role") == "user" and "gathered enough evidence" in m.get("content", "")
    ]
    assert nudges, "expected a nudge once the step budget was reached"
    # Exactly one nudge, not one per step.
    last_request = client.completions.seen[-1]
    assert sum("gathered enough evidence" in m.get("content", "") for m in last_request) == 1


def _propose(call_id: str, citations: str, **over: Any) -> _Msg:
    """A scripted propose_resolution call."""
    args = {
        "dispute_id": "D-1004",
        "disposition": "escalate",
        "rationale": "r",
        "citations": citations,
        **over,
    }
    return _Msg(tool_calls=[_Call(call_id, _Fn("propose_resolution", json.dumps(args)))])


_SEARCH = _Msg(
    tool_calls=[
        _Call("s", _Fn("search_policy", json.dumps({"query": "duplicate charge same merchant"})))
    ]
)


def test_nudge_does_not_fire_when_agent_decides(monkeypatch: pytest.MonkeyPatch) -> None:
    """An agent that proposes promptly should never see the nudge.

    It has to search first: citing POL-022 without retrieving it is refused by
    the grounding gate, and a refused call records nothing — so a script that
    skips the search is not testing a successful proposal at all.
    """
    client = _install(monkeypatch, [_SEARCH, _propose("p", "POL-022"), _Msg(content="done")])

    asyncio.run(loop_manual.run("D-1004", "stub", verbose=False))
    all_msgs = [m for req in client.completions.seen for m in req]
    assert not any("gathered enough evidence" in m.get("content", "") for m in all_msgs)
    assert not any("have not recorded a decision" in m.get("content", "") for m in all_msgs)


def test_rejected_proposal_does_not_count_as_a_decision(monkeypatch: pytest.MonkeyPatch) -> None:
    """A refused propose_resolution must not satisfy the completion check.

    The bug: `proposed` was computed from the trace, which records *attempted*
    calls. One rejection — an ungrounded citation is the common one — latched it
    true for the rest of the run, disabling both the nudge and the chase. The
    loop then returned a clean "finished" result with nothing in the database,
    at precisely the moment the agent most needed pushing.
    """
    # POL-032 exists but was never retrieved, so the server refuses this.
    client = _install(monkeypatch, [_propose("u", "POL-032"), _Msg(content="all done")])

    out = asyncio.run(loop_manual.run("D-1004", "stub", verbose=False))

    attempted = [e for e in out["trace"] if e["tool"] == "propose_resolution"]
    assert attempted and "ungrounded citation" in attempted[0]["result"]

    chases = [
        m
        for req in client.completions.seen
        for m in req
        if m.get("role") == "user" and "have not recorded a decision" in m.get("content", "")
    ]
    assert chases, "a rejected proposal must still be chased"


def test_recorded_a_decision_reads_the_result_not_the_attempt() -> None:
    """The predicate itself: only an `ok: true` payload counts."""
    ok = {"tool": "propose_resolution", "result": json.dumps({"ok": True, "proposal_id": 1})}
    refused = {"tool": "propose_resolution", "result": json.dumps({"ok": False, "error": "no"})}
    garbage = {"tool": "propose_resolution", "result": "connection reset"}
    other = {"tool": "search_policy", "result": json.dumps({"ok": True})}

    assert loop_manual.recorded_a_decision([ok])
    assert loop_manual.recorded_a_decision([refused, ok])
    assert not loop_manual.recorded_a_decision([refused])
    # Unparseable is treated as "not recorded": chasing again is cheap, a false
    # positive loses the decision outright.
    assert not loop_manual.recorded_a_decision([garbage])
    assert not loop_manual.recorded_a_decision([other])
    assert not loop_manual.recorded_a_decision([])


def test_agent_recovers_from_ungrounded_citation(monkeypatch: pytest.MonkeyPatch) -> None:
    """The grounding gate must be recoverable, not just punitive.

    Reproduces the observed qwen2.5:7b behaviour deterministically: propose with
    a real-but-unretrieved citation, get refused, search, then propose again
    citing what came back. Driving this with the live model costs ~10 minutes a
    case; here it costs two seconds, and it tests the thing that actually
    matters — that the refusal message is actionable.
    """
    propose_ungrounded = _Msg(
        tool_calls=[
            _Call(
                "u",
                _Fn(
                    "propose_resolution",
                    json.dumps(
                        {
                            "dispute_id": "D-1004",
                            "disposition": "provisional_credit",
                            "rationale": "looks duplicated",
                            "citations": "POL-032",
                            "amount": 189.0,
                        }
                    ),
                ),
            )
        ]
    )
    search = _Msg(
        tool_calls=[
            _Call(
                "s",
                _Fn("search_policy", json.dumps({"query": "duplicate charge same merchant"})),
            )
        ]
    )
    propose_grounded = _Msg(
        tool_calls=[
            _Call(
                "g",
                _Fn(
                    "propose_resolution",
                    json.dumps(
                        {
                            "dispute_id": "D-1004",
                            "disposition": "provisional_credit",
                            "rationale": "POL-022 Clause 22.1 satisfied",
                            "citations": "POL-022",
                            "amount": 189.0,
                        }
                    ),
                ),
            )
        ]
    )
    client = _install(
        monkeypatch, [propose_ungrounded, search, propose_grounded, _Msg(content="done")]
    )

    out = asyncio.run(loop_manual.run("D-1004", "stub", verbose=False))
    assert out["final_text"] == "done"

    tool_msgs = [m for req in client.completions.seen for m in req if m.get("role") == "tool"]
    joined = "\n".join(m["content"] for m in tool_msgs)

    # The first attempt was refused, and told the model why in actionable terms.
    assert "ungrounded citation" in joined
    assert "POL-032" in joined
    # The second attempt, made after searching, was accepted.
    assert '"ok": true' in joined.lower() or "'ok': true" in joined.lower()


def test_iteration_cap_holds(monkeypatch: pytest.MonkeyPatch) -> None:
    """A model that never stops calling tools must terminate the loop."""
    script = [
        _Msg(tool_calls=[_Call("loop", _Fn("get_dispute", json.dumps({"dispute_id": "D-1004"})))])
    ]
    _install(monkeypatch, script)

    out = asyncio.run(loop_manual.run("D-1004", "stub", verbose=False))
    assert out["steps"] == loop_manual.MAX_STEPS
    assert "MAX_STEPS" in out["error"]
