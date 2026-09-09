"""Retrieval metrics at the level of articles.

Two kinds of questions, two definitions (`need_all`):

* one answer, possibly findable in several articles (`expected_articles` are alternatives):
  a question is found when ANY expected article is in the top k;
* `multi_article` (`expected_articles` are all required): Recall@k is the SHARE of expected
  articles in the top k, and the reciprocal rank is that of the rank where the LAST required
  article appears (0 when one is missing). Finding only article 115 of {115, 125} is half
  an answer, not a success.
"""

import math
from collections.abc import Sequence

from tkrag.retrieval.types import Hit


def ranked_articles(hits: Sequence[Hit]) -> list[str]:
    """Article numbers in rank order, each article once (a long article may give 2 chunks)."""
    return list(dict.fromkeys(h.article_number for h in hits))


def recall_at_k(ranked: Sequence[str], expected: set[str], k: int, need_all: bool = False) -> float:
    found = expected & set(ranked[:k])
    if not need_all:
        return 1.0 if found else 0.0
    return len(found) / len(expected) if expected else 0.0


def reciprocal_rank(ranked: Sequence[str], expected: set[str], need_all: bool = False) -> float:
    positions = {article: rank for rank, article in enumerate(ranked, start=1)}
    if not need_all:
        hits = [positions[a] for a in expected if a in positions]
        return 1.0 / min(hits) if hits else 0.0
    if not expected or any(a not in positions for a in expected):
        return 0.0
    return 1.0 / max(positions[a] for a in expected)


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile (p in 0..100). Simple and explainable."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(math.ceil(p / 100 * len(ordered)) - 1, 0)
    return ordered[index]
