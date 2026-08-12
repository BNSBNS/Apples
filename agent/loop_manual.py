"""THE TEACHING FILE — the agentic loop, written out by hand.

Read this one top to bottom. Everything an "agent framework" does for you is
here in about 80 lines of actual logic, and none of it is magic.

The shape:

    tools = ask the MCP server what it can do        (once)
    loop:
        reply = ask the model, giving it those tools
        if reply.finish_reason != "tool_calls": done
        for each requested call:
            run it on the MCP server
            append the result as a `tool` message
        repeat

Three things that break hand-written loops, all of them handled below:

1. `function.arguments` arrives as a **JSON string**, not a dict. Parse it, and
   be ready for a small model to emit invalid JSON.
2. Every `tool_call_id` the model produced **must** get exactly one matching
   `tool` message back. Miss one and the next request 400s.
3. The loop needs a hard iteration cap. A model that keeps calling tools
   forever is a real failure mode, not a hypothetical one.

Run it:
    python -m agent.loop_manual --dispute D-1004                 # local, free
    python -m agent.loop_manual --dispute D-1004 --provider deepseek
    python -m agent.loop_manual --dispute D-1004 --server http://localhost:8000/mcp
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import sys
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamable_http_client

from agent.prompts import SYSTEM
from agent.providers import KINDS, get_provider

MAX_STEPS = 12

# Mid-sized local models reliably *gather* evidence and then never commit to the
# terminal action — they keep searching until the step cap and produce nothing.
# Observed directly: qwen2.5:7b spent 8-10 tool calls on D-1001/D-1002 and never
# called propose_resolution. A budget nudge fixes it: once the agent has done
# enough work, tell it explicitly that it now has to decide. Real harnesses do
# this; it is not a hack, it is closing the loop.
NUDGE_AT = 5
TERMINAL_TOOL = "propose_resolution"


@dataclass(frozen=True)
class AgentSpec:
    """What makes one agent different from another. There is nothing else.

    The loop below is identical for the triage writer and the verifier — same
    tool-call plumbing, same nudge, same step cap. Only these five values differ,
    so this is a Parameter Object rather than a class hierarchy: subclasses would
    hold nothing but methods returning constants. `Provider` in agent/providers.py
    is the same shape for the same reason.

    `terminal_tool` has to be here rather than a module constant. It drives
    `recorded_a_decision()`, the nudge and the chase, so an agent that cannot call
    `propose_resolution` would otherwise be told to "call propose_resolution now"
    — a tool it does not have.
    """

    system: str
    task: str
    allowed_tools: frozenset[str]
    terminal_tool: str
    nudge: str


# ANSI colours, disabled when piped.
_C = sys.stdout.isatty()
DIM, BOLD, CYAN, GREEN, YELLOW, RESET = (
    ("\033[2m", "\033[1m", "\033[36m", "\033[32m", "\033[33m", "\033[0m") if _C else ("",) * 6
)


class _ToolProxy:
    """Presents an MCP tool under a different name (for namespace prefixing).

    `mcp_tools_to_openai` and `tool_schema` read `.name`, `.description`, and
    `.input_schema` — this proxy overrides `.name` and delegates the rest.
    """

    def __init__(self, prefixed_name: str, original: Any) -> None:
        self._original = original
        self.name = prefixed_name

    @property
    def description(self) -> str | None:
        return self._original.description

    @property
    def input_schema(self) -> Any:
        return self._original.input_schema


def tool_schema(tool: Any) -> dict[str, Any]:
    """Get a tool's JSON Schema off an MCP Tool object.

    The attribute is `input_schema`. `inputSchema` is only the wire alias, and
    reading it returns nothing — the same snake_case/camelCase trap as
    `is_error` on CallToolResult.

    This one was expensive. Reading the wrong name meant every tool went to the
    model with an EMPTY parameter schema, so no model was ever told what
    arguments any tool accepts. The resulting wrong-argument-name failures look
    exactly like model incompetence, which is what I initially concluded. Hence
    the assertion below: fail loudly rather than degrade into a silent, plausible
    lie about model quality.
    """
    schema = getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", None)
    if not schema:
        raise RuntimeError(
            f"tool {getattr(tool, 'name', '?')!r} exposed no input schema — "
            "the MCP SDK attribute name has changed again. Do not fall back to an "
            "empty schema: the model would be left guessing argument names."
        )
    return schema


def mcp_tools_to_openai(mcp_tools: list[Any]) -> list[dict[str, Any]]:
    """Translate MCP tool definitions into OpenAI function-calling schema.

    This is the entire "integration" between MCP and the model API: a shape
    change. MCP gives `name` / `description` / `input_schema`; OpenAI wants them
    nested under `function` with `parameters`. Any provider that speaks
    function calling needs a translator this size and no bigger.
    """
    out = []
    for t in mcp_tools:
        schema = tool_schema(t)
        out.append(
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": (t.description or "").strip(),
                    "parameters": schema,
                },
            }
        )
    return out


def looks_like_textual_tool_call(text: str) -> bool:
    """Did the model *describe* a tool call instead of emitting one?

    A real failure mode on small models: the reply contains something like
    `{"name": "get_dispute", "parameters": {...}}` as prose. The turn ends
    cleanly with no tool_calls, so an unguarded loop returns a confident
    non-answer. Cheap to detect, and worth detecting.
    """
    if not text or "{" not in text:
        return False
    lowered = text.lower()
    return ('"name"' in lowered or "'name'" in lowered) and (
        "parameters" in lowered or "arguments" in lowered
    )


def schema_hint(name: str, schemas: dict[str, dict[str, Any]]) -> str:
    """Re-state a tool's exact parameters after a failed call.

    Small models routinely invent plausible-but-wrong argument names (`id`
    instead of `dispute_id`). Echoing the raw validation error back is usually
    not enough to recover; handing them the literal schema is. This costs a few
    tokens and measurably improves recovery on weak local models.
    """
    if name not in schemas:
        return f"HINT: there is no tool called `{name}`. Available tools: {sorted(schemas)}."
    schema = schemas[name] or {}
    props = list((schema.get("properties") or {}).keys())
    required = schema.get("required") or []
    if not props:
        return f"HINT: `{name}` takes no arguments."
    return (
        f"HINT: `{name}` accepts exactly these parameters: {props}. "
        f"Required: {required}. Use these names verbatim — do not rename or abbreviate them."
    )


def coerce_arguments(
    name: str, args: dict[str, Any], schemas: dict[str, dict[str, Any]]
) -> tuple[dict[str, Any], str | None]:
    """Repair a single unambiguously-misnamed argument before calling the tool.

    Observed with qwen2.5:7b: it calls `get_dispute(dispute_id=...)` correctly,
    then immediately calls `get_transaction(id=...)` — abbreviating the name —
    and repeats the identical wrong call three times in a row even after being
    handed the literal schema. Telling the model was not sufficient.

    The rule is deliberately narrow, so this repairs typos rather than guessing
    intent: fire only when exactly one required parameter is missing AND the
    model supplied exactly one key the schema does not define. Anything more
    ambiguous is left to fail, with the schema hint.

    Repairs are returned, not hidden — the caller records them, and the eval
    harness reports how often they fire. A model needing frequent repair is a
    finding about the model, and burying it would corrupt the measurement.
    """
    schema = schemas.get(name) or {}
    props = schema.get("properties") or {}
    if not props:
        return args, None

    required = [p for p in (schema.get("required") or []) if p not in args]
    extra = [k for k in args if k not in props]

    if len(required) == 1 and len(extra) == 1:
        fixed = dict(args)
        fixed[required[0]] = fixed.pop(extra[0])
        return fixed, f"{name}: {extra[0]!r} -> {required[0]!r}"

    return args, None


def recorded_a_decision(trace: list[dict[str, Any]], terminal_tool: str = TERMINAL_TOOL) -> bool:
    """Did a call to the agent's terminal tool actually *succeed*?

    Not "was one attempted". The server rejects proposals for real reasons — an
    ungrounded citation, an invented policy id, an unknown disposition — and a
    rejected call writes nothing. Treating the attempt as the decision means one
    early rejection permanently disables both the budget nudge and the
    completion chase, and the run then returns a clean-looking result with no
    proposal in the database at all. The rejection is exactly the moment the
    agent most needs pushing.

    A malformed payload counts as "not recorded": chasing once more is cheap,
    while a false positive loses the decision entirely.
    """
    for entry in trace:
        if entry.get("tool") != terminal_tool:
            continue
        try:
            payload = json.loads(entry.get("result", ""))
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(payload, dict) and payload.get("ok") is True:
            return True
    return False


@contextlib.asynccontextmanager
async def open_transport(server: str | None) -> AsyncIterator[tuple[Any, Any]]:
    """Open stdio or streamable-http and yield the same `(read, write)` pair.

    The whole transport difference lives here, in one context manager, and the
    loop below never learns which one it got. Both SDK clients yield the identical
    2-tuple, so this stays a swap rather than a branch threaded through the loop.

    stdio  — the server is a subprocess we spawn. DISPUTE_DB (and RETRIEVER) are
             forwarded so the child reads the database the caller intends; without
             that it silently falls back to defaults. See server/db.py.
    http   — the server is already running somewhere. The bearer token rides on a
             pre-configured httpx client; `streamable_http_client` has no
             `headers=` argument of its own.
    """
    if server is None:
        params = StdioServerParameters(
            command=sys.executable, args=["-m", "server.app"], cwd=os.getcwd(), env=dict(os.environ)
        )
        async with stdio_client(params) as streams:
            yield streams
        return

    import httpx2

    token = os.getenv("MCP_TOKEN", "")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    async with (
        httpx2.AsyncClient(headers=headers, timeout=120) as http_client,
        streamable_http_client(server, http_client=http_client) as streams,
    ):
        yield streams


@contextlib.asynccontextmanager
async def open_fx_transport() -> AsyncIterator[tuple[Any, Any]]:
    """Open a stdio transport to the FX rates MCP server."""
    params = StdioServerParameters(
        command=sys.executable, args=["-m", "fx_server.app"], cwd=os.getcwd(), env=dict(os.environ)
    )
    async with stdio_client(params) as streams:
        yield streams


def result_to_text(result: Any) -> str:
    """Flatten an MCP CallToolResult into a string for the model."""
    parts: list[str] = []
    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
    return "\n".join(parts) if parts else "(no content)"


def triage_spec(dispute_id: str) -> AgentSpec:
    """The original agent, expressed as a spec. This is the whole diff between
    "the loop" and "an agent": the loop is generic, the spec is the role."""
    return AgentSpec(
        system=SYSTEM,
        task=f"Triage dispute {dispute_id}.",
        allowed_tools=frozenset(),  # empty = every tool the server offers
        terminal_tool=TERMINAL_TOOL,
        nudge=(
            "You have gathered enough evidence. Do not search again. "
            f"Call `{TERMINAL_TOOL}` now with your disposition, a short rationale, "
            "and the policy ids you retrieved. If the evidence is unclear or a "
            "mandatory trigger applies, use 'escalate'."
        ),
    )


async def run(
    dispute_id: str,
    provider_kind: str,
    verbose: bool = True,
    model_override: str | None = None,
    server: str | None = None,
    observer: Any = None,
    use_fx: bool = False,
) -> dict[str, Any]:
    """Triage one dispute — the original entry point, unchanged for callers.

    `server=None` spawns the MCP server over stdio; a URL talks to an
    already-running one over streamable-http.

    `use_fx=True` also spawns the FX rates server and merges its tools.
    """
    if not use_fx:
        return await run_agent(
            triage_spec(dispute_id),
            provider_kind,
            verbose=verbose,
            model_override=model_override,
            server=server,
            label=dispute_id,
            observer=observer,
        )

    async with open_fx_transport() as (fx_read, fx_write):
        async with ClientSession(fx_read, fx_write) as fx_session:
            await fx_session.initialize()
            fx_session._server_name = "fx"
            return await run_agent(
                triage_spec(dispute_id),
                provider_kind,
                verbose=verbose,
                model_override=model_override,
                server=server,
                label=dispute_id,
                observer=observer,
                extra_servers=[fx_session],
            )


async def run_agent(
    spec: AgentSpec,
    provider_kind: str,
    verbose: bool = True,
    model_override: str | None = None,
    server: str | None = None,
    label: str = "",
    observer: Any = None,
    extra_servers: list[Any] | None = None,
) -> dict[str, Any]:
    """Run one agent to completion. The loop is the same for every role.

    `observer`, when provided, is called as `observer(kind, data_dict)` at
    every interesting moment — tool calls, results, nudges, the final
    answer.  The dashboard wires this to an SSE stream; the eval harness
    could wire it to a log.  `verbose` print output is independent.

    `extra_servers` is a list of additional (already-initialized)
    ClientSessions whose tools are merged into the agent's tool list. Each
    tool is routed to the session that owns it via a name→session map.
    """
    notify = observer or (lambda _kind, _data: None)
    provider = get_provider(provider_kind, model_override)

    started = time.time()
    prompt_tokens = completion_tokens = 0
    # Ordered record of what the agent actually did. The eval harness scores
    # against this: which tools were reached for, in what order, and whether
    # the retrieval step surfaced the governing policy.
    trace: list[dict[str, Any]] = []
    # Argument names the harness had to fix for the model. Surfaced, not hidden:
    # a high repair count is a fact about the model, not something to bury.
    repairs: list[str] = []

    async with open_transport(server) as (read, write), ClientSession(read, write) as session:
        await session.initialize()

        # --- 1. Discover what the server can do -------------------------
        listed = await session.list_tools()
        # An agent sees only the tools its role allows. This is a *behavioural*
        # boundary — it stops the model reaching for what it should not use. The
        # real boundary is the token scope the server checks (server/app.py); a
        # verifier's credential cannot propose even if it asks for the tool.
        offered = [
            t for t in listed.tools if not spec.allowed_tools or t.name in spec.allowed_tools
        ]

        # tool name → (session, original_name). The LLM sees namespaced
        # names when multiple servers are connected; `call_tool` gets the
        # original name the server registered. Without this, two servers
        # exposing the same tool name would silently overwrite each other.
        tool_router: dict[str, tuple[ClientSession, str]] = {
            t.name: (session, t.name) for t in offered
        }

        # Merge tools from any extra MCP servers, namespaced.
        for extra_session in extra_servers or []:
            extra_listed = await extra_session.list_tools()
            server_name = getattr(extra_session, "_server_name", "extra")
            for t in extra_listed.tools:
                if not spec.allowed_tools or t.name in spec.allowed_tools:
                    if t.name in tool_router:
                        prefixed = f"{server_name}__{t.name}"
                    else:
                        prefixed = t.name
                    offered.append(_ToolProxy(prefixed, t))
                    tool_router[prefixed] = (extra_session, t.name)

        tools = mcp_tools_to_openai(offered)
        schemas = {t.name: tool_schema(t) for t in offered}
        if verbose:
            print(f"{DIM}connected — {len(tools)} tools from {1 + len(extra_servers or [])} server(s){RESET}")
            print(f"{DIM}provider: {provider}{RESET}\n")
        notify("connect", {"tools": [t.name for t in offered], "provider": str(provider)})

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": spec.system},
            {"role": "user", "content": spec.task},
        ]

        # --- 2. The loop ------------------------------------------------
        nudged = chased = False
        for step in range(1, MAX_STEPS + 1):
            notify("step", {"step": step})
            # Budget nudge: enough evidence gathered, still no decision.
            proposed = recorded_a_decision(trace, spec.terminal_tool)
            if step >= NUDGE_AT and not proposed and not nudged:
                nudged = True
                if verbose:
                    print(f"  {YELLOW}!{RESET} {step - 1} steps, no decision yet — nudging")
                notify("nudge", {"step": step})
                messages.append({"role": "user", "content": spec.nudge})

            reply = await provider.client.chat.completions.create(
                model=provider.model,
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )
            if reply.usage:
                prompt_tokens += reply.usage.prompt_tokens or 0
                completion_tokens += reply.usage.completion_tokens or 0

            msg = reply.choices[0].message
            calls = msg.tool_calls or []

            # The model is done: no tool calls requested.
            if not calls:
                text = (msg.content or "").strip()

                # Observed failure on small local models: instead of
                # emitting a real tool_call, they *describe* one in prose.
                # The turn looks finished, so a naive loop silently returns
                # a non-answer. Detect it and push back once per step
                # rather than accepting the dead turn.
                if looks_like_textual_tool_call(text) and step < MAX_STEPS:
                    if verbose:
                        print(f"  {YELLOW}!{RESET} model wrote a tool call as text; correcting")
                    notify("correction", {"reason": "textual_tool_call", "step": step})
                    messages.append({"role": "assistant", "content": text})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "That was written as plain text, so it did not execute. "
                                "Issue it as a real tool call using the function-calling "
                                "interface, with the exact parameter names from the schema."
                            ),
                        }
                    )
                    continue

                # Ending the turn without recording a decision is a failed
                # triage, not a finished one. Models routinely narrate a
                # conclusion in prose ("I propose provisional credit...")
                # without ever calling the tool that records it. Push back once.
                if not proposed and not chased and step < MAX_STEPS:
                    chased = True
                    if verbose:
                        print(f"  {YELLOW}!{RESET} finished without recording a decision — chasing")
                    notify("chase", {"step": step})
                    messages.append({"role": "assistant", "content": text})
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "You have not recorded a decision. Describing one in prose does "
                                f"not count. Call the `{spec.terminal_tool}` tool now."
                            ),
                        }
                    )
                    continue

                if verbose and text:
                    print(f"{BOLD}final:{RESET} {text}\n")
                notify("final", {"text": text, "step": step})
                return {
                    "dispute_id": label,
                    "provider": provider.name,
                    "model": provider.model,
                    "steps": step,
                    "final_text": text,
                    "trace": trace,
                    "repairs": repairs,
                    "messages": messages,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "seconds": round(time.time() - started, 2),
                }

            # The model sometimes reasons in text before issuing tool calls.
            # Surface this so the dashboard can show what the model is thinking.
            if msg.content and msg.content.strip():
                if verbose:
                    print(f"  {DIM}thinking: {msg.content.strip()[:120]}{RESET}")
                notify("reasoning", {"text": msg.content.strip(), "step": step})

            # Echo the assistant turn back, tool_calls intact. Building this
            # explicitly (rather than dumping the SDK object) keeps the wire
            # format visible and avoids sending fields Ollama may reject.
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {
                                "name": c.function.name,
                                "arguments": c.function.arguments,
                            },
                        }
                        for c in calls
                    ],
                }
            )

            # --- 3. Execute every call, answer every id -----------------
            for call in calls:
                name = call.function.name

                # Pitfall #1: arguments is a JSON *string*, and a small model
                # will sometimes emit malformed JSON. Feed the error back
                # rather than crashing — the model can usually recover.
                try:
                    args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError as exc:
                    payload = f"ERROR: arguments were not valid JSON ({exc}). Retry the call."
                    if verbose:
                        print(f"  {YELLOW}!{RESET} {name} — bad JSON arguments")
                    messages.append({"role": "tool", "tool_call_id": call.id, "content": payload})
                    continue

                args, repair = coerce_arguments(name, args, schemas)
                if repair:
                    repairs.append(repair)

                if verbose:
                    shown = json.dumps(args)
                    print(f"  {CYAN}→{RESET} {BOLD}{name}{RESET}{DIM}({shown[:100]}){RESET}")
                    if repair:
                        print(f"    {YELLOW}repaired argument:{RESET} {DIM}{repair}{RESET}")
                notify("tool_call", {"name": name, "args": args, "step": step})
                if repair:
                    notify("repair", {"detail": repair, "step": step})

                try:
                    target_session, original_name = tool_router.get(name, (session, name))
                    result = await target_session.call_tool(original_name, arguments=args)
                    payload = result_to_text(result)
                    # A tool can also fail *inside* a successful call: the
                    # RPC succeeds and the result carries an error flag.
                    # Note the attribute is `is_error` — `isError` is only
                    # the wire alias, and reading that silently yields
                    # False, which makes this whole branch dead code.
                    if getattr(result, "is_error", False):
                        payload = f"{payload}\n{schema_hint(name, schemas)}"
                except Exception as exc:  # tool crash must not kill the loop
                    payload = f"ERROR calling {name}: {exc}\n{schema_hint(name, schemas)}"
                    if verbose:
                        print(f"  {YELLOW}!{RESET} {payload.splitlines()[0]}")

                if verbose:
                    first = payload.replace("\n", " ")[:110]
                    print(f"    {DIM}{first}{RESET}")

                trace.append({"tool": name, "args": args, "result": payload})
                notify("tool_result", {"name": name, "result": payload, "step": step})

                # Pitfall #2: one tool message per tool_call_id, always.
                messages.append({"role": "tool", "tool_call_id": call.id, "content": payload})

            if verbose:
                print()

        # Pitfall #3: ran out of steps without the model finishing.
        notify("step_limit", {"max_steps": MAX_STEPS})
        return {
            "dispute_id": label,
            "provider": provider.name,
            "model": provider.model,
            "steps": MAX_STEPS,
            "final_text": "",
            "trace": trace,
            "repairs": repairs,
            "messages": messages,
            "error": f"hit MAX_STEPS ({MAX_STEPS}) without a final answer",
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "seconds": round(time.time() - started, 2),
        }


async def run_followup(
    messages: list[dict[str, Any]],
    user_message: str,
    provider_kind: str,
    model_override: str | None = None,
    server: str | None = None,
    observer: Any = None,
    use_fx: bool = False,
) -> dict[str, Any]:
    """Run a follow-up turn with full tool access.

    Takes the message history from a previous triage run, appends the new
    user message, opens MCP server(s), and runs a short agent loop so the
    model can call tools (e.g. FX rate lookups) to answer the question.
    """
    notify = observer or (lambda _kind, _data: None)
    provider = get_provider(provider_kind, model_override)

    started = time.time()
    prompt_tokens = completion_tokens = 0
    trace: list[dict[str, Any]] = []
    repairs: list[str] = []

    max_turns = 5

    stack = contextlib.AsyncExitStack()
    try:
        read, write = await stack.enter_async_context(open_transport(server))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()

        listed = await session.list_tools()
        offered = list(listed.tools)
        tool_router: dict[str, tuple[ClientSession, str]] = {
            t.name: (session, t.name) for t in offered
        }

        if use_fx:
            fx_read, fx_write = await stack.enter_async_context(open_fx_transport())
            fx_session = await stack.enter_async_context(ClientSession(fx_read, fx_write))
            await fx_session.initialize()
            fx_listed = await fx_session.list_tools()
            for t in fx_listed.tools:
                if t.name in tool_router:
                    prefixed = f"fx__{t.name}"
                else:
                    prefixed = t.name
                offered.append(_ToolProxy(prefixed, t))
                tool_router[prefixed] = (fx_session, t.name)

        tools = mcp_tools_to_openai(offered)
        schemas = {t.name: tool_schema(t) for t in offered}

        msgs = list(messages)
        msgs.append({"role": "user", "content": user_message})

        for step in range(1, max_turns + 1):
            notify("step", {"step": step})

            reply = await provider.client.chat.completions.create(
                model=provider.model,
                messages=msgs,
                tools=tools,
                tool_choice="auto",
            )
            if reply.usage:
                prompt_tokens += reply.usage.prompt_tokens or 0
                completion_tokens += reply.usage.completion_tokens or 0

            msg = reply.choices[0].message
            calls = msg.tool_calls or []

            if not calls:
                text = (msg.content or "").strip()
                notify("chat_reply", {"text": text})
                msgs.append({"role": "assistant", "content": text})
                return {
                    "text": text,
                    "messages": msgs,
                    "steps": step,
                    "trace": trace,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "seconds": round(time.time() - started, 2),
                }

            if msg.content and msg.content.strip():
                notify("reasoning", {"text": msg.content.strip(), "step": step})

            msgs.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": c.id,
                        "type": "function",
                        "function": {
                            "name": c.function.name,
                            "arguments": c.function.arguments,
                        },
                    }
                    for c in calls
                ],
            })

            for call in calls:
                name = call.function.name
                try:
                    args = json.loads(call.function.arguments or "{}")
                except json.JSONDecodeError as exc:
                    payload = f"ERROR: arguments were not valid JSON ({exc}). Retry the call."
                    msgs.append({"role": "tool", "tool_call_id": call.id, "content": payload})
                    continue

                args, repair = coerce_arguments(name, args, schemas)
                if repair:
                    repairs.append(repair)

                notify("tool_call", {"name": name, "args": args, "step": step})
                if repair:
                    notify("repair", {"detail": repair, "step": step})

                try:
                    target_session, original_name = tool_router.get(name, (session, name))
                    result = await target_session.call_tool(original_name, arguments=args)
                    payload = result_to_text(result)
                    if getattr(result, "is_error", False):
                        payload = f"{payload}\n{schema_hint(name, schemas)}"
                except Exception as exc:
                    payload = f"ERROR calling {name}: {exc}\n{schema_hint(name, schemas)}"

                trace.append({"tool": name, "args": args, "result": payload})
                notify("tool_result", {"name": name, "result": payload, "step": step})
                msgs.append({"role": "tool", "tool_call_id": call.id, "content": payload})

        text = "(reached follow-up step limit without a final answer)"
        notify("chat_reply", {"text": text})
        return {
            "text": text,
            "messages": msgs,
            "steps": max_turns,
            "trace": trace,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "seconds": round(time.time() - started, 2),
        }
    finally:
        await stack.aclose()


def main() -> None:
    ap = argparse.ArgumentParser(description="Triage a dispute with a hand-written agent loop.")
    ap.add_argument("--dispute", default="D-1004", help="dispute id, e.g. D-1004")
    ap.add_argument("--provider", default="local", choices=KINDS)
    ap.add_argument("--model", default=None, help="override the model, e.g. qwen2.5:7b")
    ap.add_argument(
        "--server",
        default=None,
        metavar="URL",
        help="talk to a running MCP server over HTTP (e.g. http://localhost:8000/mcp); "
        "omit to spawn one over stdio. Token comes from MCP_TOKEN.",
    )
    ap.add_argument("--fx", action="store_true", help="also connect to the FX rates MCP server")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    out = asyncio.run(
        run(
            args.dispute,
            args.provider,
            verbose=not args.quiet,
            model_override=args.model,
            server=args.server,
            use_fx=args.fx,
        )
    )

    print(f"{GREEN}{'─' * 62}{RESET}")
    if out.get("error"):
        print(f"{YELLOW}{out['error']}{RESET}")
    print(
        f"{DIM}{out['provider']}/{out['model']} · {out['steps']} steps · "
        f"{out['seconds']}s · {out['prompt_tokens']}+{out['completion_tokens']} tokens{RESET}"
    )


if __name__ == "__main__":
    main()
