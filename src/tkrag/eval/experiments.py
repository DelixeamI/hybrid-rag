"""Experiment: can a different sparse channel or fusion make the hybrid better than dense alone?

Honest protocol: the questions are split into a DEV half and a TEST half (stratified by type,
fixed seed). Every configuration is scored on DEV, one winner is picked on DEV, and only that
winner and the baselines are reported on TEST. With ~38 questions per half the noise is large
(one question moves Recall@1 by ~2.6 points), so small differences mean nothing.
"""

import random
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import psycopg

from tkrag.eval.golden import GoldenItem
from tkrag.eval.metrics import mean, recall_at_k, reciprocal_rank
from tkrag.retrieval.rrf import rrf_scores
from tkrag.retrieval.search import build_tsquery


def split_dev_test(
    items: Sequence[GoldenItem], seed: int = 13
) -> tuple[list[GoldenItem], list[GoldenItem]]:
    """Stratified 50/50 split by question type; the same seed always gives the same split."""
    rng = random.Random(seed)
    by_type: dict[str, list[GoldenItem]] = defaultdict(list)
    for item in items:
        by_type[item.type].append(item)
    dev: list[GoldenItem] = []
    test: list[GoldenItem] = []
    for kind in sorted(by_type):
        group = sorted(by_type[kind], key=lambda i: i.id)
        rng.shuffle(group)
        half = (len(group) + 1) // 2
        dev += group[:half]
        test += group[half:]
    return dev, test


# --- Sparse variants ---------------------------------------------------------------------------

_VARIANT_SQL = {
    # name: SQL expression that ranks (higher is better) given the tsquery `q.query`
    "fts_cd": "ts_rank_cd(c.tsv, q.query)",  # what the system uses today
    "fts_cd_len": "ts_rank_cd(c.tsv, q.query, 1)",  # divided by 1 + log(document length)
    "fts_cd_len_unique": "ts_rank_cd(c.tsv, q.query, 1|8)",  # + divided by unique words
    "fts_rank": "ts_rank(c.tsv, q.query)",
    "title_only": "ts_rank_cd(to_tsvector('russian', c.article_title), q.query)",
}
_WHERE_EXTRA = {"title_only": "to_tsvector('russian', c.article_title) @@ q.query"}


def sparse_variant(
    conn: psycopg.Connection, document: str, question: str, variant: str, limit: int
) -> list[int]:
    query = build_tsquery(question)
    if not query:
        return []
    condition = _WHERE_EXTRA.get(variant, "c.tsv @@ q.query")
    rows = conn.execute(
        f"""
        SELECT c.id, {_VARIANT_SQL[variant]} AS score
        FROM chunks c JOIN documents d ON d.id = c.document_id,
             to_tsquery('russian', %(query)s) AS q(query)
        WHERE d.name = %(doc)s AND {condition}
        ORDER BY score DESC, c.id
        LIMIT %(limit)s
        """,
        {"query": query, "doc": document, "limit": limit},
    ).fetchall()
    return [int(r[0]) for r in rows]


# --- Scoring -----------------------------------------------------------------------------------

Ranking = list[int]


@dataclass
class Prepared:
    """Everything a configuration needs for one question, computed once."""

    item: GoldenItem
    rankings: dict[str, Ranking]  # "dense", "bm25", "fts_cd", ...


def articles_of(ranking: Sequence[int], article_of: dict[int, str]) -> list[str]:
    return list(dict.fromkeys(article_of[cid] for cid in ranking))


def score_ranking(
    prepared: Sequence[Prepared],
    make_ranking: Callable[[Prepared], Ranking],
    article_of: dict[int, str],
) -> dict[str, float]:
    rows = []
    for p in prepared:
        ranked = articles_of(make_ranking(p), article_of)
        expected = set(p.item.expected_articles)
        need_all = p.item.needs_all_articles
        rows.append(
            {
                "recall@1": recall_at_k(ranked, expected, 1, need_all),
                "recall@5": recall_at_k(ranked, expected, 5, need_all),
                "recall@20": recall_at_k(ranked, expected, 20, need_all),
                "mrr": reciprocal_rank(ranked, expected, need_all),
            }
        )
    out = {key: mean([r[key] for r in rows]) for key in rows[0]} if rows else {}
    out["n"] = float(len(rows))
    return out


@dataclass(frozen=True)
class Config:
    sparse: str  # key into Prepared.rankings
    k: int
    dense_weight: float

    @property
    def label(self) -> str:
        return f"{self.sparse}, k={self.k}, w_dense={self.dense_weight:g}"


def fused_ranking(p: Prepared, config: Config) -> Ranking:
    scores = rrf_scores(
        [p.rankings["dense"], p.rankings[config.sparse]], config.k, [config.dense_weight, 1.0]
    )
    return sorted(scores, key=lambda cid: (-scores[cid], cid))


def grid(sparse_names: Sequence[str]) -> list[Config]:
    return [
        Config(name, k, weight)
        for name in sparse_names
        for k in (10, 60)
        for weight in (1.0, 2.0, 3.0)
    ]


BASELINE = Config("fts_cd", 60, 1.0)  # the hybrid the system ships today


def pick_winner(dev_scores: dict[Config, dict[str, float]]) -> Config:
    """Best dev MRR; ties go to the higher dev Recall@5, then to the baseline-like config."""

    def key(config: Config) -> tuple[float, float, float]:
        s = dev_scores[config]
        closeness = -(abs(config.k - BASELINE.k) / 100 + abs(config.dense_weight - 1.0))
        return (round(s["mrr"], 6), round(s["recall@5"], 6), closeness)

    return max(dev_scores, key=key)
