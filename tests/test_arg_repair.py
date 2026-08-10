"""Argument-name repair.

qwen2.5:7b called `get_dispute(dispute_id=...)` correctly and then
`get_transaction(id=...)` — abbreviating the parameter — and repeated the
identical wrong call three times after being shown the schema. Telling the
model was not enough, so the harness repairs the unambiguous case.

The rule must stay narrow: repair typos, never guess intent.
"""

from __future__ import annotations

import pytest

from agent.loop_manual import coerce_arguments

SCHEMAS = {
    "get_transaction": {
        "properties": {"transaction_id": {"type": "string"}},
        "required": ["transaction_id"],
    },
    "search_policy": {
        "properties": {"query": {"type": "string"}, "k": {"type": "integer"}},
        "required": ["query"],
    },
    "propose_resolution": {
        "properties": {
            "dispute_id": {"type": "string"},
            "disposition": {"type": "string"},
            "rationale": {"type": "string"},
            "citations": {"type": "string"},
            "amount": {"type": "number"},
        },
        "required": ["dispute_id", "disposition", "rationale", "citations"],
    },
    "no_args": {"properties": {}},
}


def test_repairs_the_observed_failure():
    """`id` -> `transaction_id`, the exact call qwen kept getting wrong."""
    args, repair = coerce_arguments("get_transaction", {"id": "T-2001"}, SCHEMAS)
    assert args == {"transaction_id": "T-2001"}
    assert repair == "get_transaction: 'id' -> 'transaction_id'"


@pytest.mark.parametrize(
    ("tool", "args"),
    [
        # Correct already, and a correct call with an extra optional set.
        ("get_transaction", {"transaction_id": "T-2001"}),
        ("search_policy", {"query": "duplicate", "k": 3}),
        # Two required missing, so which one the stray key meant is a guess.
        ("propose_resolution", {"dispute_id": "D-1", "foo": "x"}),
        # Two unknown keys — same problem from the other side.
        ("get_transaction", {"a": "1", "b": "2"}),
        # Nothing to repair against.
        ("nope", {"x": 1}),
        ("no_args", {"x": 1}),
        # Required is satisfied, so a stray key is the model's error to own,
        # not ours to paper over.
        ("search_policy", {"query": "x", "kk": 3}),
    ],
    ids=[
        "already-correct",
        "optional-set",
        "two-required-missing",
        "two-extras",
        "unknown-tool",
        "no-params",
        "required-satisfied",
    ],
)
def test_leaves_everything_it_cannot_unambiguously_repair(tool, args):
    """The rule fires only when exactly one required param is missing AND exactly
    one supplied key is unknown. Everything else must pass through untouched and
    fail loudly downstream, rather than be silently invented."""
    out, repair = coerce_arguments(tool, dict(args), SCHEMAS)
    assert repair is None
    assert out == args
