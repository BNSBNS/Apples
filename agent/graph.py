"""LangGraph agent wired to the Dispute Desk MCP server.

Same server, same tools, same protocol — different client. Where
`loop_manual.py` is a hand-written while loop, this is an explicit graph:
two nodes, one conditional edge, and typed state flowing between them.

    python -m agent.chat --dispute D-1004 --provider openai

Budget controls (max steps, token ceiling) are first-class: the graph
checks them before every LLM call and hard-stops if either is exceeded.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Annotated, Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from mcp import ClientSession
from pydantic import BaseModel

from agent.loop_manual import result_to_text, tool_schema

MAX_STEPS = 12
NUDGE_AT = 5
MAX_TOKENS = 50_000


# -- State -------------------------------------------------------------------

def _add_messages(
    existing: list[BaseMessage], new: list[BaseMessage]
) -> list[BaseMessage]:
    return existing + new


@dataclass
class AgentState:
    messages: Annotated[list[BaseMessage], _add_messages] = field(
        default_factory=list
    )
    steps: int = 0
    total_tokens: int = 0
    nudged: bool = False
    chased: bool = False
    done: bool = False


# -- MCP tool wrapper --------------------------------------------------------

class MCPTool(BaseTool):
    """Wraps one MCP tool as a LangChain tool.

    Each call goes through session.call_tool() — the same RPC
    loop_manual.py uses. The schema comes from the MCP server's
    input_schema attribute, not the wire alias.
    """

    session: Any = None
    args_schema: type[BaseModel] | None = None

    class Config:
        arbitrary_types_allowed = True

    def _run(self, **kwargs: Any) -> str:
        raise NotImplementedError("Use async")

    async def _arun(self, **kwargs: Any) -> str:
        result = await self.session.call_tool(self.name, arguments=kwargs)
        text = result_to_text(result)
        if getattr(result, "is_error", False):
            return f"ERROR: {text}"
        return text


def mcp_tools_to_langchain(
    mcp_tools: list[Any], session: ClientSession
) -> list[MCPTool]:
    """Convert MCP tool definitions into LangChain tools."""
    out = []
    for t in mcp_tools:
        schema = tool_schema(t)
        props = schema.get("properties", {})
        required = schema.get("required", [])

        # Build a Pydantic model from the MCP schema so LangChain
        # knows the argument names and types for function calling.
        fields: dict[str, Any] = {}
        for name, prop in props.items():
            py_type = str  # safe default
            if prop.get("type") == "integer":
                py_type = int
            elif prop.get("type") == "number":
                py_type = float
            default = ... if name in required else None
            fields[name] = (py_type, default)

        args_model = type(f"{t.name}_args", (BaseModel,), {
            "__annotations__": {k: v[0] for k, v in fields.items()},
            **{k: v[1] for k, v in fields.items()},
        }) if fields else None

        out.append(MCPTool(
            name=t.name,
            description=(t.description or "").strip(),
            session=session,
            args_schema=args_model,
        ))
    return out


# -- Graph nodes -------------------------------------------------------------

def build_graph(
    tools: list[MCPTool],
    model_name: str = "gpt-4o-mini",
    base_url: str | None = None,
    api_key: str | None = None,
    terminal_tool: str = "propose_resolution",
    nudge_text: str = "",
    system_prompt: str = "",
) -> StateGraph:
    """Build and compile the agent graph."""

    model = ChatOpenAI(
        model=model_name,
        temperature=0.2,
        base_url=base_url,
        api_key=api_key or os.getenv("OPENAI_API_KEY", "not-set"),
    ).bind_tools(tools)

    async def call_model(state: AgentState) -> dict:
        """LLM node — ask the model, check budget."""
        step = state.steps + 1

        # Hard budget limits.
        if step > MAX_STEPS:
            return {
                "done": True,
                "messages": [AIMessage(content=f"[stopped: hit {MAX_STEPS} step limit]")],
            }
        if state.total_tokens > MAX_TOKENS:
            return {
                "done": True,
                "messages": [AIMessage(content=f"[stopped: hit {MAX_TOKENS} token limit]")],
            }

        msgs = list(state.messages)

        # Ensure system prompt is always the first message.
        if system_prompt and (
            not msgs or not isinstance(msgs[0], SystemMessage)
        ):
            msgs.insert(0, SystemMessage(content=system_prompt))

        # Budget nudge at NUDGE_AT if no decision yet.
        if step >= NUDGE_AT and not state.nudged and nudge_text:
            nudge_needed = not _has_successful_terminal(msgs, terminal_tool)
            if nudge_needed:
                msgs.append(HumanMessage(content=nudge_text))

        response = await model.ainvoke(msgs)

        tokens = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            tokens = response.usage_metadata.get("total_tokens", 0)

        updates: dict[str, Any] = {
            "messages": [response],
            "steps": step,
            "total_tokens": state.total_tokens + tokens,
        }

        if state.nudged or (step >= NUDGE_AT and nudge_text):
            updates["nudged"] = True

        return updates

    async def execute_tools(state: AgentState) -> dict:
        """Tool node — execute each tool call on the MCP server."""
        last = state.messages[-1]
        if not isinstance(last, AIMessage) or not last.tool_calls:
            return {"messages": []}

        tool_map = {t.name: t for t in tools}
        results: list[ToolMessage] = []

        for tc in last.tool_calls:
            tool = tool_map.get(tc["name"])
            if not tool:
                results.append(ToolMessage(
                    content=f"Unknown tool: {tc['name']}",
                    tool_call_id=tc["id"],
                ))
                continue

            try:
                output = await tool._arun(**tc["args"])
            except Exception as exc:
                output = f"ERROR: {exc}"

            results.append(ToolMessage(
                content=output,
                tool_call_id=tc["id"],
            ))

        return {"messages": results}

    def should_continue(state: AgentState) -> str:
        """Route: tool calls → execute. Otherwise → done."""
        if state.done:
            return END

        last = state.messages[-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"

        # Completion chase: model finished without calling terminal tool.
        # chased is set by chase_node's return dict, not here — edge
        # functions must be pure (read state, return a string).
        if not state.chased and not _has_successful_terminal(
            state.messages, terminal_tool
        ):
            return "chase"

        return END

    async def chase_node(state: AgentState) -> dict:
        """Push the model to call the terminal tool if it narrated instead."""
        return {
            "messages": [HumanMessage(
                content=(
                    "You have not recorded a decision. Describing one in prose "
                    f"does not count. Call `{terminal_tool}` now."
                )
            )],
            "chased": True,
        }

    # -- Wire the graph ------------------------------------------------------
    graph = StateGraph(AgentState)

    graph.add_node("agent", call_model)
    graph.add_node("tools", execute_tools)
    graph.add_node("chase", chase_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {
        "tools": "tools",
        "chase": "chase",
        END: END,
    })
    graph.add_edge("tools", "agent")
    graph.add_edge("chase", "agent")

    return graph.compile()


# -- Helpers -----------------------------------------------------------------

def _has_successful_terminal(
    messages: list[BaseMessage], terminal_tool: str
) -> bool:
    """Check if the terminal tool was called and succeeded.

    Single pass: index tool-call ids from AIMessages, then check each
    ToolMessage result against the index.
    """
    terminal_ids: set[str] = set()
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                if tc["name"] == terminal_tool:
                    terminal_ids.add(tc["id"])
        elif isinstance(msg, ToolMessage) and msg.tool_call_id in terminal_ids:
            try:
                payload = json.loads(msg.content)
            except (json.JSONDecodeError, TypeError):
                continue
            if isinstance(payload, dict) and payload.get("ok") is True:
                return True
    return False
