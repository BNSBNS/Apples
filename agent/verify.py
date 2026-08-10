"""The second agent: an independent check on what the first one proposed.

    python -m agent.verify --dispute D-1004
    python -m agent.verify --dispute D-1004 --provider deepseek

Two agents, one loop. The only difference from triage is an `AgentSpec`; the loop
in `loop_manual.py` grew no branches to support a second role.

It gets its **own tools** and re-fetches the evidence: handed only the writer's
text, a model is a second opinion on the same evidence rather than an
independent check. It **annotates rather than gates**, because a second gate
would let one model's mistake suppress another's correct answer — and because
the writer then needed no changes at all.

It cannot call `propose_resolution`: not offered the tool, and under HTTP its
token lacks `dispute:propose`. One behavioural boundary, one enforced.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from typing import Any

from agent.loop_manual import (
    DIM,
    GREEN,
    RESET,
    YELLOW,
    AgentSpec,
    run_agent,
)
from agent.prompts import VERIFIER
from agent.providers import KINDS
from server import db

TERMINAL_TOOL = "record_verdict"

# Read-only, plus the one write it is allowed. `propose_resolution` is absent by
# design — see the module docstring.
VERIFIER_TOOLS = frozenset(
    {"get_dispute", "get_transaction", "get_dispute_history", "search_policy", TERMINAL_TOOL}
)


def pending_proposal(dispute_id: str) -> dict[str, Any] | None:
    return db.one(
        "SELECT * FROM proposals WHERE dispute_id = ? AND status = 'pending' ORDER BY id DESC",
        (dispute_id,),
    )


def verifier_spec(proposal: dict[str, Any]) -> AgentSpec:
    """Build the review task from the draft.

    The draft has to be in the prompt — the verifier is reviewing it. What must
    NOT come from the draft is the evidence, which the agent fetches itself.
    """
    amount = "none" if proposal["amount"] is None else f"{proposal['amount']:.2f}"
    return AgentSpec(
        system=VERIFIER,
        task=(
            f"Review the proposed resolution for dispute {proposal['dispute_id']}.\n\n"
            f"  disposition: {proposal['disposition']}\n"
            f"  amount:      {amount}\n"
            f"  citations:   {proposal['citations']}\n"
            f"  rationale:   {proposal['rationale']}\n\n"
            "Gather the evidence yourself, then record your verdict."
        ),
        allowed_tools=VERIFIER_TOOLS,
        terminal_tool=TERMINAL_TOOL,
        nudge=(
            "You have gathered enough evidence. Do not search again. "
            f"Call `{TERMINAL_TOOL}` now with verdict='pass' or verdict='fail' "
            "and concrete reasons."
        ),
    )


async def verify(
    dispute_id: str,
    provider_kind: str = "local",
    verbose: bool = True,
    model_override: str | None = None,
    server: str | None = None,
) -> dict[str, Any]:
    """Review the pending proposal for one dispute. Returns the run result."""
    proposal = pending_proposal(dispute_id)
    if proposal is None:
        return {"error": f"no pending proposal for {dispute_id} — run the triage agent first"}

    out = await run_agent(
        verifier_spec(proposal),
        provider_kind,
        verbose=verbose,
        model_override=model_override,
        server=server,
        label=dispute_id,
    )
    out["verdict"] = verdict_from_trace(out.get("trace") or [])
    return out


def verdict_from_trace(trace: list[dict[str, Any]]) -> str | None:
    """The verdict actually recorded, read from the tool result rather than the
    attempt — the same distinction `recorded_a_decision` makes, for the same
    reason: a rejected call wrote nothing."""
    for entry in trace:
        if entry.get("tool") != TERMINAL_TOOL:
            continue
        try:
            payload = json.loads(entry.get("result", ""))
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(payload, dict) and payload.get("ok") is True:
            return payload.get("verdict")
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description="Independently review a proposed resolution.")
    ap.add_argument("--dispute", default="D-1004", help="dispute id, e.g. D-1004")
    ap.add_argument("--provider", default="local", choices=KINDS)
    ap.add_argument("--model", default=None, help="override the model")
    ap.add_argument("--server", default=None, metavar="URL", help="MCP server URL (else stdio)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    out = asyncio.run(
        verify(
            args.dispute,
            args.provider,
            verbose=not args.quiet,
            model_override=args.model,
            server=args.server,
        )
    )

    if out.get("error"):
        print(f"{YELLOW}{out['error']}{RESET}")
        return

    print(f"{GREEN}{'─' * 62}{RESET}")
    verdict = out.get("verdict")
    print(f"verdict: {verdict or f'{YELLOW}none recorded{RESET}'}")
    print(
        f"{DIM}{out['provider']}/{out['model']} · {out['steps']} steps · "
        f"{out['seconds']}s · {out['prompt_tokens']}+{out['completion_tokens']} tokens{RESET}"
    )
    print(f"{DIM}see it alongside the proposal: python -m agent.approval --list{RESET}")


if __name__ == "__main__":
    main()
