"""Fine-tune the cross-encoder on this corpus. The PyTorch loop, written by hand.

    python -m training.train_rerank                 # BCE, the pretrained head's own loss
    python -m training.train_rerank --loss listwise # softmax CE over 1 positive + K negatives

Deliberately not `sentence-transformers.fit()`, for the same reason
`agent/loop_manual.py` is not an agent framework: the interesting part is the
loss and the gradient, and a one-line `.fit()` hides both.

**The two losses, and why they differ.**

`bce` treats each (query, passage) pair independently: sigmoid the logit, punish
the distance from 0/1. It matches how `ms-marco-MiniLM` was trained, so it is the
honest baseline. Its weakness is that ranking is relative — a model can score
every passage 0.9 and still order them correctly, or score them all differently
and order them wrong, and BCE barely distinguishes those.

`listwise` scores one positive against K hard negatives **as a group** and applies
softmax cross-entropy over the group. The gradient then depends on the *margin*
between the true clause and its nearest competitors, which is the quantity that
actually decides recall@1. This is why hard negatives matter: with random
negatives the softmax is already saturated, the loss is ~0, and the update is
noise. The negatives here come from the retriever's own top hits, so they sit
close to the positive and the gradient stays informative.

Saves to `models/rerank-finetuned/`. `server/rerank.py` already reads
`RERANK_MODEL`, so nothing under `server/` changes:

    RERANK_MODEL=./models/rerank-finetuned python -m evals.retrieval_eval --rerank
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

PAIRS = Path(__file__).parent.parent / "data" / "train_pairs.jsonl"
OUT = Path(__file__).parent.parent / "models" / "rerank-finetuned"
BASE = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def load_pairs(path: Path = PAIRS) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(
            f"no training data at {path} — run `python -m training.build_pairs` first."
        )
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def split_by_document(
    pairs: list[dict[str, Any]], holdout: float = 0.2, seed: int = 0
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split on the document a QUERY came from, never on the query itself.

    Queries generated from one clause are near-duplicates of each other. Split
    them randomly and the train half answers the test half from memory, and the
    validation loss reads far better than the model is. Grouping by
    `query_policy_id` keeps every query about a document on the same side.

    Residual optimism, stated rather than hidden: 15 documents is a small pool,
    and the eval cases in `evals/cases.jsonl` are backed by documents that appear
    in *training*. What is never trained on is the eval queries themselves —
    those are dispute narratives from `data/seed.py`, structurally unlike the
    generated questions. So the held-out retrieval score is honest about query
    generalisation and mildly optimistic about document generalisation.
    """
    docs = sorted({p["query_policy_id"] for p in pairs})
    random.Random(seed).shuffle(docs)
    n_val = max(1, round(len(docs) * holdout))
    val_docs = set(docs[:n_val])
    train = [p for p in pairs if p["query_policy_id"] not in val_docs]
    val = [p for p in pairs if p["query_policy_id"] in val_docs]
    return train, val


def group_by_query(pairs: list[dict[str, Any]]) -> list[tuple[str, str, list[str]]]:
    """(query, positive_passage, [negatives]) — the unit the listwise loss needs."""
    buckets: dict[str, dict[str, Any]] = defaultdict(lambda: {"pos": None, "neg": []})
    for p in pairs:
        b = buckets[p["query"]]
        if p["label"] == 1:
            b["pos"] = p["passage"]
        else:
            b["neg"].append(p["passage"])
    return [(q, b["pos"], b["neg"]) for q, b in buckets.items() if b["pos"] and b["neg"]]


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tune the cross-encoder reranker.")
    ap.add_argument("--loss", default="bce", choices=["bce", "listwise"])
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    import tls_trust

    tls_trust.warn_if_unavailable("downloading the base reranker")

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    pairs = load_pairs()
    train, val = split_by_document(pairs, seed=args.seed)
    print(f"\n{len(pairs)} pairs · train {len(train)} / val {len(val)} · loss={args.loss}")
    print(f"held-out documents: {sorted({p['query_policy_id'] for p in val})}\n")

    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForSequenceClassification.from_pretrained(BASE)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    def encode(queries: list[str], passages: list[str]):
        return tok(
            queries,
            passages,
            padding=True,
            truncation=True,
            max_length=args.max_length,
            return_tensors="pt",
        )

    if args.loss == "bce":
        # Each pair judged on its own — the pretrained head's own objective.
        data = [(p["query"], p["passage"], float(p["label"])) for p in train]
        loader = DataLoader(data, batch_size=args.batch_size, shuffle=True, collate_fn=list)

        for epoch in range(1, args.epochs + 1):
            total = 0.0
            for batch in loader:
                q, p, y = zip(*batch, strict=True)
                logits = model(**encode(list(q), list(p))).logits.squeeze(-1)
                loss = F.binary_cross_entropy_with_logits(logits, torch.tensor(y))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item()
            print(f"  epoch {epoch}  train loss {total / max(len(loader), 1):.4f}")

    else:
        # One positive against its hard negatives, scored as a group. The target
        # is always index 0 because the positive is placed first.
        groups = group_by_query(train)
        loader = DataLoader(groups, batch_size=1, shuffle=True, collate_fn=list)

        for epoch in range(1, args.epochs + 1):
            total = 0.0
            for batch in loader:
                query, pos, negs = batch[0]
                passages = [pos, *negs]
                logits = model(**encode([query] * len(passages), passages)).logits.squeeze(-1)
                # softmax over the group; the true clause must outrank its
                # nearest competitors, not merely score high in isolation.
                loss = F.cross_entropy(logits.unsqueeze(0), torch.tensor([0]))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += loss.item()
            print(f"  epoch {epoch}  train loss {total / max(len(loader), 1):.4f}")

    # Held-out check, in the terms that matter: how often does the true clause
    # outrank its hard negatives? Accuracy on individual pairs would flatter it.
    model.eval()
    val_groups = group_by_query(val)
    wins = 0
    with torch.no_grad():
        for query, pos, negs in val_groups:
            passages = [pos, *negs]
            logits = model(**encode([query] * len(passages), passages)).logits.squeeze(-1)
            wins += int(torch.argmax(logits).item() == 0)
    if val_groups:
        print(
            f"\n  held-out top-1 over hard negatives: {wins}/{len(val_groups)} "
            f"({100 * wins / len(val_groups):.0f}%)"
        )

    OUT.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT)
    tok.save_pretrained(OUT)
    print(f"\n  saved to {OUT}")
    print("  measure it end to end — no code under server/ changes:")
    print(
        f"    RERANK_MODEL=./{OUT.relative_to(Path.cwd())} python -m evals.retrieval_eval --rerank"
    )


if __name__ == "__main__":
    main()
