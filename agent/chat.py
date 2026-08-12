"""Interactive chat interface for the LangGraph dispute triage agent.

    python -m agent.chat --dispute D-1004 --provider openai
    python -m agent.chat --provider openai                    # free-form chat
    python -m agent.chat --provider local --model qwen2.5:7b  # Ollama

Connects to the MCP server over stdio (same subprocess as loop_manual.py),
builds the LangGraph agent, and runs an interactive loop. Each tool call
prints inline so you see the full MCP flow.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from agent.graph import (
    MAX_STEPS,
    MAX_TOKENS,
    AgentState,
    build_graph,
    mcp_tools_to_langchain,
)
from agent.loop_manual import open_fx_transport
from agent.prompts import SYSTEM
from agent.providers import KINDS, PROVIDERS, ALIASES

load_dotenv()


def _langfuse_callback():
    """Return a Langfuse CallbackHandler if configured, else None."""
    if not os.getenv("LANGFUSE_SECRET_KEY"):
        return None
    try:
        import tls_trust
        tls_trust.enable()
        from langfuse.langchain import CallbackHandler
        return CallbackHandler()
    except ImportError:
        return None


_C = sys.stdout.isatty()
DIM, BOLD, CYAN, GREEN, YELLOW, RESET = (
    ("\033[2m", "\033[1m", "\033[36m", "\033[32m", "\033[33m", "\033[0m")
    if _C else ("",) * 6
)

NUDGE = (
    "You have gathered enough evidence. Do not search again. "
    "Call `propose_resolution` now with your disposition, a short rationale, "
    "and the policy ids you retrieved. If the evidence is unclear or a "
    "mandatory trigger applies, use 'escalate'."
)


def _resolve_provider(kind: str, model_override: str | None = None):
    """Resolve provider config to (model, base_url, api_key)."""
    resolved = ALIASES.get(kind, kind)
    config = PROVIDERS[resolved]
    model = model_override or os.getenv(config.model_env) or config.default_model
    api_key = os.getenv(config.key_env) if config.key_env else "not-needed"
    return model, config.base_url, api_key


def _print_tool_calls(msg: AIMessage) -> None:
    for tc in msg.tool_calls:
        args = tc["args"]
        shown = ", ".join(f"{k}={v!r}" for k, v in args.items())
        if len(shown) > 100:
            shown = shown[:100] + "..."
        print(f"  {CYAN}>{RESET} {BOLD}{tc['name']}{RESET}({DIM}{shown}{RESET})")


def _print_tool_result(msg: ToolMessage) -> None:
    first_line = msg.content.replace("\n", " ")[:120]
    print(f"    {DIM}{first_line}{RESET}")


async def run_interactive(args: argparse.Namespace) -> None:
    model_name, base_url, api_key = _resolve_provider(args.provider, args.model)

    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "server.app"],
        cwd=os.getcwd(),
        env=dict(os.environ),
    )

    print(f"{DIM}connecting to MCP server(s)...{RESET}")

    # Context manager stack: primary server always, FX server when --fx.
    import contextlib
    stack = contextlib.AsyncExitStack()

    try:
        read, write = await stack.enter_async_context(stdio_client(params))
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()
        listed = await session.list_tools()
        tools = mcp_tools_to_langchain(listed.tools, session)

        if args.fx:
            fx_read, fx_write = await stack.enter_async_context(open_fx_transport())
            fx_session = await stack.enter_async_context(ClientSession(fx_read, fx_write))
            await fx_session.initialize()
            fx_listed = await fx_session.list_tools()
            tools.extend(mcp_tools_to_langchain(fx_listed.tools, fx_session))

        server_count = 2 if args.fx else 1
        print(
            f"{DIM}connected — {len(tools)} tools from {server_count} server(s), "
            f"provider={args.provider}, model={model_name}{RESET}"
        )
        print(
            f"{DIM}budget: {MAX_STEPS} steps, "
            f"{MAX_TOKENS:,} tokens max{RESET}\n"
        )

        app = build_graph(
            tools=tools,
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            nudge_text=NUDGE,
            system_prompt=SYSTEM,
        )

        lf = _langfuse_callback()
        invoke_config = {"callbacks": [lf]} if lf else {}

        initial_messages: list = []

        if args.dispute:
            initial_messages.append(
                HumanMessage(content=f"Triage dispute {args.dispute}.")
            )
            print(f"{GREEN}> Triage dispute {args.dispute}{RESET}\n")

            state = AgentState(messages=initial_messages)
            result = await app.ainvoke(state, config=invoke_config)

            for msg in result["messages"]:
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    _print_tool_calls(msg)
                elif isinstance(msg, ToolMessage):
                    _print_tool_result(msg)
                elif isinstance(msg, AIMessage) and msg.content:
                    print(f"\n{BOLD}assistant:{RESET} {msg.content}\n")

            initial_messages = result["messages"]
            print(f"\n{DIM}{'─' * 60}{RESET}")
            tokens = result.get("total_tokens", 0)
            steps = result.get("steps", 0)
            print(f"{DIM}{steps} steps, {tokens:,} tokens used{RESET}\n")

        print(f"{DIM}Type a message (or 'quit' to exit){RESET}")
        while True:
            try:
                user_input = input(f"\n{GREEN}you:{RESET} ").strip()
            except (EOFError, KeyboardInterrupt):
                print(f"\n{DIM}goodbye{RESET}")
                break

            if not user_input or user_input.lower() in ("quit", "exit", "q"):
                break

            state = AgentState(
                messages=initial_messages + [HumanMessage(content=user_input)]
            )
            result = await app.ainvoke(state, config=invoke_config)

            for msg in result["messages"]:
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    _print_tool_calls(msg)
                elif isinstance(msg, ToolMessage):
                    _print_tool_result(msg)
                elif isinstance(msg, AIMessage) and msg.content:
                    print(f"\n{BOLD}assistant:{RESET} {msg.content}")

            initial_messages = result["messages"]
            tokens = result.get("total_tokens", 0)
            steps = result.get("steps", 0)
            print(f"\n{DIM}{steps} steps, {tokens:,} tokens used{RESET}")

        if lf:
            from langfuse import get_client
            get_client().flush()

    finally:
        await stack.aclose()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Interactive LangGraph agent for dispute triage."
    )
    ap.add_argument("--dispute", default=None, help="dispute id to triage on startup")
    ap.add_argument("--provider", default="openai", choices=KINDS)
    ap.add_argument("--model", default=None, help="override the model name")
    ap.add_argument("--fx", action="store_true", help="also connect to the FX rates MCP server")
    args = ap.parse_args()
    asyncio.run(run_interactive(args))


if __name__ == "__main__":
    main()
