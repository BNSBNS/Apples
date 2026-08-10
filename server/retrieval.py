"""Retrieval over the policy corpus.

Two implementations behind one Protocol:

    KeywordRetriever    stdlib only, zero deps. Always works.
    EmbeddingRetriever  nomic-embed-text via Ollama (768-dim) + numpy cosine.

The Protocol *is* the lesson: retrieval is a swappable component. Nothing that
calls `search()` knows or cares which one is behind it. In production you'd add
a third implementation backed by pgvector or OpenSearch and change one env var.

Note there is no LLM SDK imported here. Embedding is an HTTP POST to an
OpenAI-shaped `/v1/embeddings` endpoint — the same request body works against
Ollama locally and OpenAI in the cloud; only the base URL and auth header
differ. Keeping it stdlib is what lets `server/` stay provider-agnostic.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

POLICY_DIR = Path(__file__).parent.parent / "data" / "policies"
CACHE_DIR = Path(__file__).parent.parent / "data" / ".cache"

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Chunk:
    """One retrievable unit: a single `##` section of one policy document."""

    policy_id: str
    heading: str
    text: str

    @property
    def citation(self) -> str:
        return f"{self.policy_id} — {self.heading}"


def chunk_markdown(path: Path) -> list[Chunk]:
    """Split a policy doc on `##` headings.

    Section-level chunking suits this corpus because each clause is already a
    self-contained rule. For prose you would use a sliding window with overlap.
    """
    policy_id = path.stem.split("-")[0] + "-" + path.stem.split("-")[1]
    raw = path.read_text(encoding="utf-8")

    # Text before the first '##' is the doc preamble (title + scope) — keep it,
    # it carries the debit/credit distinction that makes this corpus confusable.
    parts = re.split(r"^## ", raw, flags=re.MULTILINE)
    chunks: list[Chunk] = []

    preamble = parts[0].strip()
    if preamble:
        chunks.append(Chunk(policy_id, "Scope", preamble))

    for part in parts[1:]:
        lines = part.strip().splitlines()
        if not lines:
            continue
        heading = lines[0].strip()
        body = "\n".join(lines[1:]).strip()
        chunks.append(Chunk(policy_id, heading, f"{heading}\n{body}"))

    return chunks


def load_corpus(policy_dir: Path = POLICY_DIR) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(policy_dir.glob("*.md")):
        chunks.extend(chunk_markdown(path))
    return chunks


# --------------------------------------------------------------------------
# The interface
# --------------------------------------------------------------------------


class Retriever(Protocol):
    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        """Return the k best-matching chunks, highest score first."""
        ...


# --------------------------------------------------------------------------
# Implementation 1: keyword (TF-IDF cosine), stdlib only
# --------------------------------------------------------------------------


class KeywordRetriever:
    """Classic TF-IDF. No network, no model, no dependencies.

    Kept as the default fallback so the demo never hard-blocks on Ollama being
    up. It is also the honest baseline the embedding retriever has to beat.
    """

    def __init__(self, chunks: list[Chunk] | None = None) -> None:
        self.chunks = chunks if chunks is not None else load_corpus()
        self._tf: list[Counter[str]] = [Counter(_tokenize(c.text)) for c in self.chunks]
        n = len(self.chunks)
        df: Counter[str] = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        self._idf = {t: math.log((n + 1) / (d + 1)) + 1.0 for t, d in df.items()}

    def _vec(self, tf: Counter[str]) -> dict[str, float]:
        return {t: c * self._idf.get(t, 0.0) for t, c in tf.items()}

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        q = self._vec(Counter(_tokenize(query)))
        qn = math.sqrt(sum(v * v for v in q.values())) or 1.0

        scored: list[tuple[Chunk, float]] = []
        for chunk, tf in zip(self.chunks, self._tf, strict=True):
            d = self._vec(tf)
            dn = math.sqrt(sum(v * v for v in d.values())) or 1.0
            dot = sum(w * d.get(t, 0.0) for t, w in q.items())
            scored.append((chunk, dot / (qn * dn)))

        scored.sort(key=lambda p: p[1], reverse=True)
        return scored[:k]


# --------------------------------------------------------------------------
# Implementation 2: dense embeddings via an OpenAI-shaped /v1/embeddings API
# --------------------------------------------------------------------------


def embed(
    texts: list[str], *, base_url: str, model: str, api_key: str | None = None
) -> list[list[float]]:
    """POST to an OpenAI-compatible /v1/embeddings endpoint.

    Works unchanged against Ollama (no key) and OpenAI (key required).
    """
    body = json.dumps({"model": model, "input": texts}).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    req = urllib.request.Request(  # noqa: S310 - fixed, operator-configured endpoint
        f"{base_url.rstrip('/')}/embeddings", data=body, headers=headers, method="POST"
    )
    with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310
        payload = json.load(resp)
    return [row["embedding"] for row in payload["data"]]


class EmbeddingRetriever:
    """Dense retrieval. Embeds the corpus once and caches it on disk.

    The cache key is a hash of the chunk texts + model name, so editing a policy
    or switching models invalidates it automatically.
    """

    def __init__(self, chunks: list[Chunk] | None = None) -> None:
        import numpy as np

        self.np = np
        self.chunks = chunks if chunks is not None else load_corpus()
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        self.model = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

        texts = [c.text for c in self.chunks]
        key = hashlib.sha256(("||".join(texts) + self.model).encode()).hexdigest()[:16]
        cache = CACHE_DIR / f"emb-{key}.npy"

        if cache.exists():
            self.matrix = np.load(cache)
        else:
            vectors = embed(texts, base_url=self.base_url, model=self.model)
            self.matrix = np.array(vectors, dtype="float32")
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            np.save(cache, self.matrix)

        # Pre-normalise so cosine similarity is a single matrix-vector product.
        norms = np.linalg.norm(self.matrix, axis=1, keepdims=True)
        self.matrix = self.matrix / np.clip(norms, 1e-9, None)

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        np = self.np
        qv = np.array(embed([query], base_url=self.base_url, model=self.model)[0], dtype="float32")
        qv = qv / max(float(np.linalg.norm(qv)), 1e-9)
        scores = self.matrix @ qv
        top = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top]


# --------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------


def get_retriever(kind: str | None = None, rerank: bool | None = None) -> Retriever:
    """Pick an implementation. Falls back to keyword if embeddings are unreachable.

    Setting RERANK=1 wraps whichever retriever is chosen in a cross-encoder
    reranker. The wrapper satisfies the same protocol, so nothing downstream
    changes — that is the point of the protocol.
    """
    # Default is keyword because it MEASURED better on this corpus, not because
    # it is simpler: TF-IDF reaches 100% recall@5 to the embedder's 75%, with a
    # better MRR (0.792 vs 0.713) and zero dependencies. See the retrieval table
    # in README.md, reproducible with `python -m evals.retrieval_eval`.
    # Exact legal/numeric terminology is lexical-matching's home ground.
    kind = (kind or os.getenv("RETRIEVER", "keyword")).lower()

    if kind == "keyword":
        base: Retriever = KeywordRetriever()
    else:
        try:
            base = EmbeddingRetriever()
        except Exception as exc:  # Ollama down, model not pulled, numpy missing
            print(f"[retrieval] embedding retriever unavailable ({exc}); using keyword", flush=True)
            base = KeywordRetriever()

    if rerank is None:
        rerank = os.getenv("RERANK", "").lower() in {"1", "true", "yes"}
    if not rerank:
        return base

    try:
        from server.rerank import RerankingRetriever

        return RerankingRetriever(base)
    except Exception as exc:  # torch/transformers absent — not fatal
        print(f"[retrieval] reranker unavailable ({exc}); using {type(base).__name__}", flush=True)
        return base
