"""Golden set: loading and validation (including the type quotas from the task)."""

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args

from pydantic import BaseModel, field_validator

QuestionType = Literal["factual", "paraphrase", "multi_article", "no_answer"]
QUESTION_TYPES: tuple[str, ...] = get_args(QuestionType)

MIN_SHARE: dict[str, float] = {"no_answer": 0.15, "paraphrase": 0.25, "multi_article": 0.10}


class GoldenItem(BaseModel):
    id: str
    question: str
    expected_articles: list[str]
    answer_facts: list[str]
    type: QuestionType

    @property
    def needs_all_articles(self) -> bool:
        """multi_article: every expected article is required; elsewhere they are alternatives."""
        return self.type == "multi_article"

    @field_validator("expected_articles", mode="before")
    @classmethod
    def _as_strings(cls, value: object) -> object:
        return [str(v) for v in value] if isinstance(value, list) else value


@dataclass(frozen=True)
class GoldenIssues:
    errors: list[str]

    @property
    def ok(self) -> bool:
        return not self.errors


def load_golden(path: Path) -> list[GoldenItem]:
    lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return [GoldenItem.model_validate(json.loads(ln)) for ln in lines]


def validate_items(items: list[GoldenItem], known_articles: set[str] | None = None) -> GoldenIssues:
    """Structural checks. Quotas are checked in `validate_quotas` (real golden only)."""
    errors: list[str] = []
    for ident, count in Counter(i.id for i in items).items():
        if count > 1:
            errors.append(f"duplicate id {ident!r}")
    for item in items:
        if item.type == "no_answer":
            if item.expected_articles:
                errors.append(f"{item.id}: no_answer must have empty expected_articles")
            continue
        if not item.expected_articles:
            errors.append(f"{item.id}: expected_articles is empty")
        if not item.answer_facts:
            errors.append(f"{item.id}: answer_facts is empty")
        if item.type == "multi_article" and len(item.expected_articles) < 2:
            errors.append(f"{item.id}: multi_article needs at least 2 expected_articles")
        if known_articles is not None:
            for number in item.expected_articles:
                if number not in known_articles:
                    errors.append(f"{item.id}: article {number} is not in the indexed corpus")
    return GoldenIssues(errors)


def overlap_errors(items: list[GoldenItem], other: list[GoldenItem]) -> list[str]:
    """A test set must not repeat a question of the set that was used to tune the system."""
    seen = {" ".join(i.question.lower().split()): i.id for i in other}
    return [
        f"{i.id}: the same question as {seen[key]} in the other set"
        for i in items
        if (key := " ".join(i.question.lower().split())) in seen
    ]


def validate_quotas(
    items: list[GoldenItem], size_range: tuple[int, int] = (60, 100)
) -> GoldenIssues:
    errors: list[str] = []
    total = len(items)
    if not size_range[0] <= total <= size_range[1]:
        errors.append(f"golden has {total} questions, expected {size_range[0]}-{size_range[1]}")
    counts: Counter[str] = Counter(i.type for i in items)
    for kind, minimum in MIN_SHARE.items():
        share = counts[kind] / total if total else 0.0
        if share < minimum:
            errors.append(f"{kind}: {share:.0%} of questions, need at least {minimum:.0%}")
    return GoldenIssues(errors)
