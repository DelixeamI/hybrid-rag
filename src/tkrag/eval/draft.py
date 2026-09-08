"""Draft golden questions with an LLM. A human must review every line before it is used.

The draft only proposes: question text, type, and facts copied from the article. The expected
article is the one the question was generated from, which is why a person has to check that no
other article answers it equally well.
"""

import json
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

from tkrag.eval.golden import GoldenItem
from tkrag.ingest.parser import Article
from tkrag.llm import LLMClient
from tkrag.textnorm import stems

SYSTEM = """\
Ты помогаешь составить проверочный набор вопросов к Трудовому кодексу РФ. Верни только JSON \
без пояснений: {"questions": [{"question": "...", "type": "factual", "answer_facts": ["..."]}]}.
Типы: factual - вопрос с терминами кодекса; paraphrase - вопрос бытовыми словами, без терминов \
кодекса, как спросил бы человек без юридического образования; multi_article - вопрос, на который \
нельзя ответить по одной из двух статей; no_answer - вопрос не про трудовое право (answer_facts \
пустой).
answer_facts - короткие фразы с числами и сроками, которые есть в тексте статьи дословно."""


def article_text(article: Article) -> str:
    body = "\n".join(p.text if p.number is None else f"{p.number}. {p.text}" for p in article.parts)
    return f"Статья {article.number}. {article.title}\n{body}"


def single_prompt(article: Article, count: int) -> str:
    return (
        f"Предложи {count} вопроса к статье: один типа factual и один типа paraphrase "
        f"(остальные любого из этих типов).\n\n{article_text(article)}"
    )


def pair_prompt(first: Article, second: Article) -> str:
    return (
        "Предложи один вопрос типа multi_article, ответ на который требует обеих статей.\n\n"
        f"{article_text(first)}\n\n{article_text(second)}"
    )


def no_answer_prompt(count: int) -> str:
    return f"Предложи {count} вопросов типа no_answer (налоги, жилье, транспорт, документы...)."


def parse_questions(raw: str) -> list[dict[str, Any]]:
    try:
        questions = json.loads(raw)["questions"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return []
    return [q for q in questions if isinstance(q, dict) and q.get("question")]


@dataclass(frozen=True)
class DraftResult:
    items: list[GoldenItem]
    dropped: list[str]  # questions rejected by the automatic checks, with the reason


def _facts_supported(facts: list[str], articles: list[Article]) -> bool:
    """Every fact must be made of words and numbers that occur in the source article(s)."""
    available = stems(" ".join(article_text(a) for a in articles))
    return all(stems(fact) <= available for fact in facts)


def draft_golden(
    articles: list[Article],
    llm: LLMClient,
    per_article: int = 2,
    no_answer: int = 12,
    pairs: list[tuple[Article, Article]] | None = None,
) -> DraftResult:
    items: list[GoldenItem] = []
    dropped: list[str] = []

    def add(raw: str, sources: list[Article], allowed: set[str]) -> None:
        for q in parse_questions(raw):
            kind = str(q.get("type", ""))
            facts = [str(f) for f in q.get("answer_facts", [])]
            if kind not in allowed:
                dropped.append(f"wrong type {kind!r}: {q['question']}")
            elif kind != "no_answer" and (not facts or not _facts_supported(facts, sources)):
                dropped.append(f"facts not in the article: {q['question']}")
            else:
                items.append(
                    GoldenItem(
                        id=f"d{len(items) + 1:03d}",
                        question=str(q["question"]),
                        expected_articles=[]
                        if kind == "no_answer"
                        else [a.number for a in sources],
                        answer_facts=[] if kind == "no_answer" else facts,
                        type=kind,
                    )
                )

    for article in articles:
        add(llm.complete(SYSTEM, single_prompt(article, per_article)), [article],
            {"factual", "paraphrase"})  # fmt: skip
    if pairs is None:
        pairs = [(a, b) for a, b in pairwise(articles) if a.chapter == b.chapter]
    for first, second in pairs:
        add(llm.complete(SYSTEM, pair_prompt(first, second)), [first, second], {"multi_article"})
    add(llm.complete(SYSTEM, no_answer_prompt(no_answer)), [], {"no_answer"})
    return DraftResult(items, dropped)
