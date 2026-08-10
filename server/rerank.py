"""Cross-encoder reranking — the second stage of two-stage retrieval.

Why this exists, in the order the evidence arrived:

    recall@8  91.7%   the gold clause is almost always retrieved
    recall@1  66.7%   ...but it ranks first only two thirds of the time

That 25-point gap is the entire justification. A reranker can only reorder what
the first stage already found — it cannot rescue a miss — so it is worth
building precisely when recall is high and precision@1 is not. Had gold already
ranked first, this file would be measuring noise and should not have been
written. (`python -m evals.retrieval_eval` prints that verdict.)

The architecture is the standard one:

    bi-encoder    embeds query and documents SEPARATELY, compares vectors.
                  Cheap — documents are embedded once, offline. Weak at fine
                  distinctions, because the document was embedded without ever
                  seeing the query.

    cross-encoder reads (query, document) TOGETHER in one forward pass and
                  scores the pair. Far more accurate, and far too slow to run
                  over a whole corpus — so it runs over the top-k the
                  bi-encoder already shortlisted.

Fetch 20, rerank to 4. That is the whole idea.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from server.retrieval import Chunk, Retriever

MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
FETCH_K = int(os.getenv("RERANK_FETCH_K", "20"))


class CrossEncoderReranker:
    """Scores (query, passage) pairs with a PyTorch cross-encoder.

    Model and tokenizer load lazily on first use: importing this module must
    stay cheap, because the MCP server imports it at startup and most runs
    never rerank anything.
    """

    def __init__(self, model_name: str = MODEL) -> None:
        self.model_name = model_name
        self._tok = None
        self._model = None
        self._torch = None

    def _load(self) -> None:
        if self._model is not None:
            return
        # huggingface_hub ships its own HTTP client and its own trust bundle,
        # so it fails on a TLS-inspecting network even when other libraries
        # succeed. Must run before the model download is attempted.
        import tls_trust

        tls_trust.warn_if_unavailable("downloading the reranker model")

        # Imported here, not at module scope: torch is a heavyweight optional
        # dependency and the keyword/embedding paths must work without it.
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._torch = torch

        # Try the network first, then fall back to the local cache.
        #
        # `HF_HUB_OFFLINE` is not reliable here: huggingface_hub reads it into a
        # module constant at import time, so setting it later — from a conftest,
        # say — is a race against whatever imported transformers first. Passing
        # `local_files_only` explicitly is immune to import ordering.
        #
        # Consequence that matters: on a TLS-inspecting network the revalidation
        # request fails and surfaces as "Can't load the configuration of ...",
        # which reads like a missing model rather than a network problem. With
        # this fallback a cached model just works.
        last: Exception | None = None
        for local_only in (False, True):
            try:
                self._tok = AutoTokenizer.from_pretrained(
                    self.model_name, local_files_only=local_only
                )
                self._model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name, local_files_only=local_only
                )
                self._model.eval()  # disable dropout; inference only
                return
            except Exception as exc:
                last = exc
        raise RuntimeError(
            f"could not load reranker {self.model_name!r} from network or cache: {last}"
        ) from last

    def score(self, query: str, passages: list[str]) -> list[float]:
        """Relevance logit per passage. Higher is more relevant."""
        if not passages:
            return []
        self._load()
        torch = self._torch

        features = self._tok(
            [query] * len(passages),
            passages,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        with torch.no_grad():  # no gradients, no autograd graph, less memory
            logits = self._model(**features).logits

        # ms-marco cross-encoders emit a single relevance logit per pair.
        return logits.squeeze(-1).tolist() if logits.shape[-1] == 1 else logits[:, -1].tolist()


class RerankingRetriever:
    """Wraps any Retriever and reorders its results with a cross-encoder.

    Satisfies the same `Retriever` protocol as the thing it wraps, so nothing
    downstream — the MCP tool, the agent, the eval harness — knows or cares
    that it is there. That is the payoff of having defined retrieval as an
    interface in the first place.
    """

    def __init__(
        self,
        base: Retriever,
        reranker: CrossEncoderReranker | None = None,
        fetch_k: int = FETCH_K,
    ) -> None:
        self.base = base
        self.reranker = reranker or CrossEncoderReranker()
        self.fetch_k = fetch_k

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        # Stage 1: cheap and wide.
        candidates = self.base.search(query, k=max(self.fetch_k, k))
        if len(candidates) <= 1:
            return candidates[:k]

        # Stage 2: expensive and narrow.
        chunks = [c for c, _ in candidates]
        scores = self.reranker.score(query, [c.text for c in chunks])

        ranked = sorted(zip(chunks, scores, strict=True), key=lambda p: p[1], reverse=True)
        return ranked[:k]
