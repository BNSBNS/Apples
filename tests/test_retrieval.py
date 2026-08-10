"""Retrieval tests. The keyword path needs no network; the embedding path
is skipped automatically when Ollama isn't running.
"""

from __future__ import annotations

import pytest

from server.retrieval import (
    POLICY_DIR,
    Chunk,
    EmbeddingRetriever,
    KeywordRetriever,
    chunk_markdown,
    load_corpus,
)


def _ollama_up() -> bool:
    try:
        EmbeddingRetriever()
        return True
    except Exception:
        return False


# --- chunking --------------------------------------------------------------


def test_chunking_splits_on_sections():
    path = POLICY_DIR / "POL-001-provisional-credit-debit.md"
    chunks = chunk_markdown(path)
    assert len(chunks) >= 4
    assert all(c.policy_id == "POL-001" for c in chunks)
    assert any("Clause 1.1" in c.heading for c in chunks)


def test_chunking_keeps_the_preamble():
    """The preamble carries the debit/credit scope line — losing it would make
    POL-001 and POL-002 far harder to tell apart."""
    chunks = chunk_markdown(POLICY_DIR / "POL-002-provisional-credit-credit.md")
    scope = next(c for c in chunks if c.heading == "Scope")
    assert "credit card" in scope.text.lower()


def test_corpus_covers_every_document():
    docs = {p.stem.split("-")[0] + "-" + p.stem.split("-")[1] for p in POLICY_DIR.glob("*.md")}
    assert {c.policy_id for c in load_corpus()} == docs
    assert Chunk("POL-001", "Clause 1.1", "body").citation == "POL-001 — Clause 1.1"


# --- keyword retriever -----------------------------------------------------


def test_keyword_retriever_returns_k_sorted():
    r = KeywordRetriever()
    hits = r.search("provisional credit debit", k=5)
    assert len(hits) == 5
    scores = [s for _, s in hits]
    assert scores == sorted(scores, reverse=True)


def test_keyword_retriever_finds_duplicate_policy():
    r = KeywordRetriever()
    hits = r.search("charged twice duplicate same merchant same amount", k=5)
    assert any(c.policy_id == "POL-022" for c, _ in hits)


# --- embedding retriever (skipped if Ollama is down) -----------------------


@pytest.mark.skipif(not _ollama_up(), reason="Ollama not reachable")
def test_embedding_wins_on_this_one_query_but_not_in_aggregate():
    """A cautionary test, kept deliberately.

    On THIS query the dense retriever looks clearly better: TF-IDF returns the
    generic 'Scope' preamble while the embedder returns Clause 1.1, the rule
    that actually answers the question. That single example was originally
    written up as proof that embeddings were the right choice.

    Measured across all 12 labelled cases, the conclusion reverses — TF-IDF
    reaches 100% recall@5 to the embedder's 75%. See `evals/retrieval_eval.py`.

    The test stays to mark the trap: a cherry-picked example is an anecdote,
    and anecdotes and measurements disagree often enough to matter.
    """
    q = "debit card unauthorised transaction, when must provisional credit be issued?"

    kw_top = KeywordRetriever().search(q, k=1)[0][0]
    emb_top = EmbeddingRetriever().search(q, k=1)[0][0]

    assert emb_top.policy_id == "POL-001"
    assert "1.1" in emb_top.heading
    assert kw_top.heading == "Scope"  # weaker here — and better on average


@pytest.mark.skipif(not _ollama_up(), reason="Ollama not reachable")
def test_embedding_dimension_is_768():
    r = EmbeddingRetriever()
    assert r.matrix.shape[1] == 768
    assert r.matrix.shape[0] == len(r.chunks)
