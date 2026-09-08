"""Answer evaluation: refusals, answer_facts recall, share of fully verified answers."""

from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import yaml

from tkrag.eval.golden import GoldenItem
from tkrag.eval.metrics import mean
from tkrag.generation.answerer import Answerer
from tkrag.generation.schemas import AskResult
from tkrag.retrieval.retriever import Searcher
from tkrag.textnorm import content_stems, stems


def fact_found(fact: str, answer_text: str, strict: bool = False) -> bool:
    """Every number and content word of the fact appears in the answer (order is ignored).

    `strict=True` also demands the prepositions and conjunctions of the fact ("до одного года"
    is not found in "один год"). The default ignores them: they carry no fact.
    """
    wanted = stems(fact) if strict else content_stems(fact)
    return wanted <= stems(answer_text)


def facts_recall(facts: list[str], result: AskResult, strict: bool = False) -> float:
    """Share of expected facts present in an *accepted* answer. Refusals and unverified count 0."""
    if result.status != "ok" or not facts:
        return 0.0
    return mean([1.0 if fact_found(f, result.answer.answer, strict) else 0.0 for f in facts])


def evaluate_answers(
    answerer: Answerer, items: list[GoldenItem], llm_model: str, threshold: float
) -> dict[str, Any]:
    results = [(item, answerer.ask(item.question)) for item in items]
    no_answer = [r for i, r in results if i.type == "no_answer"]
    answerable = [(i, r) for i, r in results if i.type != "no_answer"]
    answered = [r for _, r in results if r.status in ("ok", "unverified")]
    counts: dict[str, int] = {}
    for _, r in results:
        counts[r.status] = counts.get(r.status, 0) + 1
    return {
        "llm_model": llm_model,
        "refusal_threshold": threshold,
        "n_no_answer": len(no_answer),
        "n_answerable": len(answerable),
        "n_answered": len(answered),
        "correct_refusal_rate": mean([1.0 if r.status == "refused" else 0.0 for r in no_answer]),
        "false_refusal_rate": mean([1.0 if r.status == "refused" else 0.0 for _, r in answerable]),
        "facts_recall": mean([facts_recall(i.answer_facts, r) for i, r in answerable]),
        "facts_recall_strict": mean(
            [facts_recall(i.answer_facts, r, strict=True) for i, r in answerable]
        ),
        "verified_rate": mean([1.0 if r.status == "ok" else 0.0 for r in answered]),
        "status_counts": counts,
        "items": [
            {
                "id": i.id,
                "type": i.type,
                "status": r.status,
                "reason": r.answer.reason,
                "best_score": round(r.best_score, 4),
                "answer": r.answer.answer,
                "articles": [c.article for c in r.answer.citations],
                "issues": r.issues,
            }
            for i, r in results
        ],
    }


@dataclass(frozen=True)
class Calibration:
    threshold: float
    curve: list[dict[str, float]]


def calibrate_threshold(scored: list[tuple[bool, float]]) -> Calibration:
    """Pick the score below which we refuse: maximises (correct refusals - false refusals).

    `scored` = (is_no_answer, best retrieval score) for every golden question.
    """
    scores = sorted({s for _, s in scored})
    candidates = [0.0] + [(a + b) / 2 for a, b in pairwise(scores)]
    candidates.append(min(scores[-1] + 1e-6, 1.0) if scores else 1.0)
    no_answer = [s for flag, s in scored if flag]
    answerable = [s for flag, s in scored if not flag]
    curve = []
    for t in candidates:
        correct = mean([1.0 if s < t else 0.0 for s in no_answer])
        false = mean([1.0 if s < t else 0.0 for s in answerable])
        curve.append(
            {
                "threshold": round(t, 4),
                "correct_refusal": correct,
                "false_refusal": false,
                "youden": correct - false,
            }
        )
    best = max(c["youden"] for c in curve)
    tied = [c["threshold"] for c in curve if c["youden"] == best]
    return Calibration(tied[len(tied) // 2], curve)


def best_scores(
    searcher: Searcher, items: list[GoldenItem], top_k: int
) -> list[tuple[bool, float]]:
    return [
        (i.type == "no_answer", searcher.retrieve(i.question, "hybrid_rerank", top_k).best_score)
        for i in items
    ]


def load_thresholds(path: Any) -> dict[str, float]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return {str(k): float(v) for k, v in loaded.items()}


def check_gate(results: dict[str, Any], thresholds: dict[str, float]) -> dict[str, Any]:
    """CI gate: the main pipeline (hybrid + rerank) must stay above the agreed floors."""
    pipeline = results["retrieval"]["hybrid_rerank"]["overall"]
    values = {
        "recall_at_5": pipeline["recall@5"],
        "correct_refusal_rate": results["answers"]["correct_refusal_rate"],
    }
    checks = [
        {
            "name": name,
            "value": values[name],
            "threshold": minimum,
            "ok": values[name] >= minimum,
        }
        for name, minimum in thresholds.items()
        if name in values
    ]
    return {"ok": all(c["ok"] for c in checks), "checks": checks}
