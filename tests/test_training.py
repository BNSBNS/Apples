"""Training-data hygiene. Nothing here trains anything or asserts model quality.

The two things that can silently invalidate a fine-tune are a leaky split and a
generator that restates the source. Both produce *better* numbers, which is what
makes them worth a test.
"""

from __future__ import annotations

from server.retrieval import Chunk
from training.build_pairs import looks_like_the_clause
from training.train_rerank import group_by_query, split_by_document


def _pair(query: str, doc: str, label: int, passage: str = "text") -> dict:
    return {
        "query": query,
        "passage": passage,
        "label": label,
        "query_policy_id": doc,
        "policy_id": doc,
        "heading": "h",
    }


def test_split_keeps_a_documents_queries_on_one_side():
    """Split by document, never by query.

    Queries generated from one clause are near-duplicates. Split them randomly
    and the train half answers the test half from memory — validation then reads
    far better than the model is, which is the failure that looks like success.
    """
    pairs = [_pair(f"q{i}", f"POL-{i % 5:03d}", i % 2) for i in range(40)]
    train, val = split_by_document(pairs, holdout=0.4, seed=0)

    train_docs = {p["query_policy_id"] for p in train}
    val_docs = {p["query_policy_id"] for p in val}
    assert train_docs and val_docs
    assert not (train_docs & val_docs), "a document appeared on both sides"
    assert len(train) + len(val) == len(pairs), "pairs were lost or duplicated"


def test_split_is_deterministic():
    pairs = [_pair(f"q{i}", f"POL-{i % 6:03d}", 1) for i in range(30)]
    assert split_by_document(pairs, seed=7) == split_by_document(pairs, seed=7)


def test_grouping_needs_both_a_positive_and_negatives():
    """The listwise loss scores one positive against its negatives, so a query
    missing either side is not trainable and must be dropped, not padded."""
    pairs = [
        _pair("has both", "POL-001", 1, "pos"),
        _pair("has both", "POL-002", 0, "neg"),
        _pair("positive only", "POL-003", 1, "pos"),
        _pair("negatives only", "POL-004", 0, "neg"),
    ]
    groups = group_by_query(pairs)

    assert [q for q, _, _ in groups] == ["has both"]
    query, pos, negs = groups[0]
    assert pos == "pos" and negs == ["neg"]


def test_near_verbatim_queries_are_rejected():
    """A query that restates the clause measures the retriever against itself."""
    chunk = Chunk(
        "POL-030",
        "Clause 30.1",
        "A partial dispense is credited as the difference between the amount debited "
        "and the amount actually dispensed.",
    )
    assert looks_like_the_clause(
        "A partial dispense is credited as the difference between the amount debited", chunk
    )
    assert not looks_like_the_clause("atm gave me less cash than it took, debit account", chunk)
