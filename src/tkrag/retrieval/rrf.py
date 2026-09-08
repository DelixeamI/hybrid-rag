"""Reciprocal Rank Fusion (Cormack et al., 2009): score(d) = sum over rankings of w / (k + rank)."""

from collections.abc import Sequence
from dataclasses import replace

from tkrag.retrieval.types import Hit


def rrf_scores(
    rankings: Sequence[Sequence[int]], k: int = 60, weights: Sequence[float] | None = None
) -> dict[int, float]:
    """`rankings` are lists of ids, best first. Rank starts at 1. Weights default to 1 each."""
    if weights is not None and len(weights) != len(rankings):
        raise ValueError("one weight per ranking")
    scores: dict[int, float] = {}
    for index, ranking in enumerate(rankings):
        weight = 1.0 if weights is None else weights[index]
        for rank, ident in enumerate(ranking, start=1):
            scores[ident] = scores.get(ident, 0.0) + weight / (k + rank)
    return scores


def fuse(
    rankings: Sequence[Sequence[Hit]], k: int = 60, weights: Sequence[float] | None = None
) -> list[Hit]:
    """Fuse hit lists by chunk_id; returns hits sorted by RRF score (stored in `score`)."""
    by_id = {hit.chunk_id: hit for ranking in rankings for hit in ranking}
    scores = rrf_scores([[h.chunk_id for h in ranking] for ranking in rankings], k, weights)
    ordered = sorted(scores, key=lambda ident: (-scores[ident], ident))
    return [replace(by_id[ident], score=scores[ident]) for ident in ordered]
