"""Scoring correctness — the eval harness measuring what it claims to measure.

A wrong metric is worse than a missing one: it produces a number that looks like
evidence. Both bugs guarded here were silent. Retrieval recall read high because
the corpus cross-references itself, and the labelled amounts were never compared
to anything at all.
"""

from __future__ import annotations

import json

from evals.metrics import CaseResult, aggregate, load_cases, score


def _search_entry(*hits: dict) -> dict:
    """A search_policy trace entry in the shape the loop actually records.

    MCP renders each hit as its own content block and `result_to_text` joins them
    with newlines, so the payload is a *run* of JSON objects rather than one
    array. Reproduce that exactly — a tidier fixture would not exercise the
    parsing this is here to protect.
    """
    return {
        "tool": "search_policy",
        "args": {"query": "q"},
        "result": "\n".join(json.dumps(h, indent=2) for h in hits),
    }


def _proposal(**over) -> dict:
    return {
        "disposition": "provisional_credit",
        "status": "pending",
        "citations": "POL-030",
        "amount": 100.0,
        **over,
    }


# --- retrieval: read the field, never the clause text ----------------------


def test_policy_ids_quoted_inside_clause_text_are_not_retrieved():
    """The bug that made hallucinated_citation unreachable.

    POL-030 Clause 30.1 cites POL-001 in its body. A regex over the whole payload
    counts POL-001 as retrieved, so a model that cites a policy it only read
    *about* scores as grounded, and retrieval_recall is inflated on top.
    """
    trace = [
        _search_entry(
            {
                "policy_id": "POL-030",
                "section": "Clause 30.1",
                "score": 0.4,
                "text": "Partial dispense is handled under POL-030, not the POL-001 timeline.",
            }
        )
    ]

    # Gold is POL-001, which was never actually returned.
    res = score(
        {"id": "D-X", "expected_disposition": "provisional_credit", "gold_policies": ["POL-001"]},
        {"trace": trace},
        _proposal(citations="POL-001"),
    )
    assert res.searched is True
    assert res.retrieval_hit is False, "POL-001 was mentioned, not retrieved"
    assert res.hallucinated_citation is True


def test_retrieved_policy_ids_come_from_every_hit():
    trace = [
        _search_entry(
            {"policy_id": "POL-022", "section": "Scope", "score": 0.4, "text": "..."},
            {"policy_id": "POL-031", "section": "Clause 31.2", "score": 0.2, "text": "..."},
        )
    ]
    res = score(
        {"id": "D-X", "expected_disposition": "escalate", "gold_policies": ["POL-031"]},
        {"trace": trace},
        _proposal(disposition="escalate", citations="POL-022,POL-031", amount=None),
    )
    assert res.retrieval_hit is True
    assert res.hallucinated_citation is False


def test_unparseable_search_payload_retrieves_nothing():
    """An error string is not a retrieval. Better to score zero than to guess."""
    trace = [{"tool": "search_policy", "args": {}, "result": "ERROR calling search_policy: boom"}]
    res = score(
        {"id": "D-X", "expected_disposition": "deny", "gold_policies": ["POL-020"]},
        {"trace": trace},
        _proposal(disposition="deny", citations="POL-020", amount=None),
    )
    assert res.retrieval_hit is False
    assert res.hallucinated_citation is True


# --- amounts: labelled means scored ----------------------------------------


def test_labelled_amount_is_actually_compared():
    """D-1008 credits the difference (100), not the full debit (300).

    Before this, expected_amount was labelled on three cases and never read, so a
    full-300 credit passed as correct.
    """
    case = {
        "id": "D-1008",
        "expected_disposition": "provisional_credit",
        "expected_amount": 100.0,
        "gold_policies": ["POL-030"],
    }
    trace = [_search_entry({"policy_id": "POL-030", "section": "s", "score": 0.4, "text": "..."})]

    full_debit = score(case, {"trace": trace}, _proposal(amount=300.0))
    assert full_debit.correct is True, "the disposition is right on its own"
    assert full_debit.amount_correct is False, "but the amount is not"

    difference = score(case, {"trace": trace}, _proposal(amount=100.0))
    assert difference.amount_correct is True


def test_none_means_unlabelled_never_passed():
    """The distinction the metric depends on: `None` is 'this case labels no
    amount', and a labelled case that produced no number is a failure."""
    unlabelled = {"id": "D-1005", "expected_disposition": "escalate", "gold_policies": ["POL-031"]}
    res = score(unlabelled, {"trace": []}, _proposal(disposition="escalate", amount=None))
    assert res.expected_amount is None and res.amount_correct is None

    labelled = {
        "id": "D-1004",
        "expected_disposition": "provisional_credit",
        "expected_amount": 189.0,
        "gold_policies": ["POL-022"],
    }
    assert score(labelled, {"trace": []}, _proposal(amount=None)).amount_correct is False
    # No proposal at all is a failure on the amount too, not a blank.
    assert score(labelled, {"trace": []}, None).amount_correct is False


def test_amount_accuracy_averages_only_over_labelled_cases():
    labelled_pass = CaseResult("A", False, expected_amount=189.0, amount_correct=True)
    labelled_fail = CaseResult("B", False, expected_amount=100.0, amount_correct=False)
    unlabelled = CaseResult("C", False)

    agg = aggregate([labelled_pass, labelled_fail, unlabelled])
    assert agg["amount_cases"] == 2
    assert agg["amount_accuracy"] == 0.5

    assert aggregate([unlabelled])["amount_accuracy"] is None


def test_every_labelled_amount_in_the_suite_is_a_real_number():
    """Guards the labels themselves — a string or a typo would score everything wrong."""
    labelled = [c for c in load_cases() if "expected_amount" in c]
    assert labelled, "the amount metric has nothing to measure"
    for case in labelled:
        assert isinstance(case["expected_amount"], (int, float))
        assert case["expected_amount"] > 0
