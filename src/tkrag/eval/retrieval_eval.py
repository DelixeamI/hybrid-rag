"""Retrieval evaluation: Recall@1/5/10, MRR and latency for every mode, overall and by type."""

from collections import defaultdict
from typing import Any

from tkrag.eval.golden import GoldenItem
from tkrag.eval.metrics import (
    mean,
    percentile,
    ranked_articles,
    recall_at_k,
    reciprocal_rank,
)
from tkrag.retrieval.retriever import Retriever
from tkrag.retrieval.types import Mode

KS = (1, 5, 10)


def _summary(rows: list[dict[str, float]]) -> dict[str, float]:
    summary = {f"recall@{k}": mean([r[f"recall@{k}"] for r in rows]) for k in KS}
    summary["mrr"] = mean([r["mrr"] for r in rows])
    summary["n"] = float(len(rows))
    return summary


def evaluate_mode(retriever: Retriever, items: list[GoldenItem], mode: Mode) -> dict[str, Any]:
    """Run every answerable question through one mode. `no_answer` has no gold article."""
    questions = [i for i in items if i.type != "no_answer"]
    if questions:  # warm-up: model load and first query plans must not pollute latency
        retriever.retrieve(questions[0].question, mode, top_k=max(KS))
    rows: list[dict[str, float]] = []
    by_type: dict[str, list[dict[str, float]]] = defaultdict(list)
    latencies: list[float] = []
    per_question: list[dict[str, Any]] = []
    for item in questions:
        result = retriever.retrieve(item.question, mode, top_k=max(KS))
        ranked = ranked_articles(result.hits)
        expected = set(item.expected_articles)
        need_all = item.needs_all_articles
        row = {f"recall@{k}": recall_at_k(ranked, expected, k, need_all) for k in KS}
        row["mrr"] = reciprocal_rank(ranked, expected, need_all)
        rows.append(row)
        per_question.append(
            {
                "id": item.id,
                "type": item.type,
                "question": item.question,
                "expected": sorted(expected),
                "need_all": need_all,
                "recall@5": row["recall@5"],
                "missing_in_top5": sorted(expected - set(ranked[:5])) if need_all else [],
                "top": ranked[:5],
            }
        )
        by_type[item.type].append(row)
        latencies.append(result.total_ms)
    return {
        "overall": _summary(rows),
        "by_type": {kind: _summary(r) for kind, r in sorted(by_type.items())},
        "latency_ms": {"p50": percentile(latencies, 50), "p95": percentile(latencies, 95)},
        "questions": per_question,
    }


def evaluate_retrieval(
    retriever: Retriever, items: list[GoldenItem], modes: tuple[Mode, ...]
) -> dict[str, Any]:
    return {mode: evaluate_mode(retriever, items, mode) for mode in modes}
