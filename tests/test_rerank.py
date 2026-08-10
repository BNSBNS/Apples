"""Reranker wiring.

The reranker measured *worse* than no reranker on this corpus and ships
disabled. These tests cover the wiring, not its usefulness — the usefulness
question is answered by `evals/retrieval_eval.py`, and the answer is no.

Kept because a negative result you cannot reproduce is not a result.
"""

from __future__ import annotations

import pytest

from server.retrieval import Chunk, KeywordRetriever, get_retriever


def _reranker_usable() -> bool:
    """Skip unless torch, transformers AND the cached model are all present.

    Checking imports alone is not enough: the weights are a separate download,
    and tests run with HF_HUB_OFFLINE=1 (see conftest). Without this the suite
    fails on a clean machine instead of skipping.
    """
    try:
        from server.rerank import CrossEncoderReranker

        CrossEncoderReranker()._load()
        return True
    except Exception:
        return False


needs_torch = pytest.mark.skipif(
    not _reranker_usable(), reason="torch/transformers or the cached cross-encoder unavailable"
)


# --- wiring, no model needed ----------------------------------------------


def test_defaults_follow_the_measurement():
    """Keyword base, reranker opt-in. Both are measured choices, not fashion —
    see the retrieval table in README.md."""
    assert isinstance(get_retriever(), KeywordRetriever)
    assert not hasattr(get_retriever(), "reranker")


def test_reranker_failure_degrades_to_base(monkeypatch):
    """A missing/broken reranker must not take retrieval down with it."""
    import sys

    monkeypatch.setitem(sys.modules, "server.rerank", None)

    r = get_retriever("keyword", rerank=True)
    # Falls back rather than raising.
    assert isinstance(r, KeywordRetriever)
    assert r.search("duplicate charge", k=2)


# --- with the real cross-encoder ------------------------------------------


@needs_torch
def test_cross_encoder_separates_relevant_from_irrelevant():
    """Sanity: the model must at least rank an obviously-relevant passage higher."""
    from server.rerank import CrossEncoderReranker

    scores = CrossEncoderReranker().score(
        "debit card provisional credit deadline",
        [
            "Provisional credit must be issued within 10 business days of receiving notice.",
            "Chargeback representment is a network-level process between issuer and acquirer.",
        ],
    )
    assert scores[0] > scores[1]


@needs_torch
def test_reranking_retriever_satisfies_the_protocol():
    """The wrapper must be indistinguishable from what it wraps."""
    from server.rerank import RerankingRetriever

    base = KeywordRetriever()
    wrapped = RerankingRetriever(base, fetch_k=10)

    hits = wrapped.search("charged twice by the same merchant", k=3)
    assert len(hits) == 3
    assert all(isinstance(c, Chunk) for c, _ in hits)

    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)


@needs_torch
def test_reranker_returns_subset_of_base_results():
    """A reranker reorders; it cannot conjure documents the base never found.

    This is why it could not fix D-1010: the embedding retriever never returned
    POL-040 at any depth, so no amount of reranking could surface it.
    """
    from server.rerank import RerankingRetriever

    base = KeywordRetriever()
    fetch_k = 10
    wrapped = RerankingRetriever(base, fetch_k=fetch_k)

    query = "business account, goods not received"
    base_ids = {(c.policy_id, c.heading) for c, _ in base.search(query, k=fetch_k)}
    reranked_ids = {(c.policy_id, c.heading) for c, _ in wrapped.search(query, k=5)}

    assert reranked_ids <= base_ids
