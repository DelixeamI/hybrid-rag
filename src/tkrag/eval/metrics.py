"""Retrieval metrics. A hit is an article: at least one expected article is in the top k."""

import math
from collections.abc import Sequence

from tkrag.retrieval.types import Hit


def ranked_articles(hits: Sequence[Hit]) -> list[str]:
    """Article numbers in rank order, each article once (a long article may give 2 chunks)."""
    return list(dict.fromkeys(h.article_number for h in hits))


def recall_at_k(ranked: Sequence[str], expected: set[str], k: int) -> float:
    return 1.0 if expected & set(ranked[:k]) else 0.0


def reciprocal_rank(ranked: Sequence[str], expected: set[str]) -> float:
    for rank, article in enumerate(ranked, start=1):
        if article in expected:
            return 1.0 / rank
    return 0.0


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile (p in 0..100). Simple and explainable."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(math.ceil(p / 100 * len(ordered)) - 1, 0)
    return ordered[index]
