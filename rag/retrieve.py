"""Retrieval hybrid: BM25 (keyword) + vector (semantik), digabung dengan Reciprocal Rank Fusion.

RRF dipakai karena skor BM25 dan cosine punya skala berbeda; RRF hanya memakai peringkat.
"""
from __future__ import annotations

from . import embeddings
from .index import Index

RRF_K = 60


def retrieve(index: Index, query: str, k: int, *, doc_types: list[str] | None = None,
             mode: str = "hybrid") -> list[dict]:
    if mode not in ("hybrid", "bm25", "vector"):
        raise ValueError(f"RETRIEVAL_MODE tidak dikenal: {mode}")

    candidates = [i for i, c in enumerate(index.chunks) if not doc_types or c.doc_type in doc_types]
    if not candidates:
        return []

    ranks: dict[str, dict[int, int]] = {}
    raw: dict[str, list[float]] = {}

    if mode in ("bm25", "hybrid"):
        scores = index.bm25.scores(query)
        order = [i for i in sorted(candidates, key=lambda i: scores[i], reverse=True) if scores[i] > 0]
        ranks["bm25"] = {i: r for r, i in enumerate(order, start=1)}
        raw["bm25"] = scores

    if mode in ("vector", "hybrid"):
        qv = embeddings.embed_query(query)
        sims = (index.vectors @ qv).tolist()
        order = sorted(candidates, key=lambda i: sims[i], reverse=True)
        ranks["vector"] = {i: r for r, i in enumerate(order, start=1)}
        raw["vector"] = sims

    fused: dict[int, float] = {}
    for method_ranks in ranks.values():
        for i, r in method_ranks.items():
            fused[i] = fused.get(i, 0.0) + 1.0 / (RRF_K + r)

    top = sorted(fused, key=fused.get, reverse=True)[:k]
    results = []
    for rank, i in enumerate(top, start=1):
        results.append({
            "rank": rank,
            **index.chunks[i].to_dict(),
            "scores": {"rrf": round(fused[i], 5),
                       **{m: round(raw[m][i], 4) for m in raw}},
            "method_ranks": {m: ranks[m].get(i) for m in ranks},
        })
    return results
