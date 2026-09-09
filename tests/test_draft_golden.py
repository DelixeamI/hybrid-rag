import json

from tkrag.eval.draft import draft_golden, parse_questions
from tkrag.eval.golden import validate_items, validate_quotas
from tkrag.ingest.parser import Article, Part
from tkrag.llm import FakeLLM


def article(number: str, chapter: str = "Глава 1") -> Article:
    return Article(
        number, f"Название {number}", None, chapter, (Part(None, "Срок 28 дней."),), False
    )


def test_parse_questions_tolerates_garbage() -> None:
    assert parse_questions("not json") == []
    assert parse_questions('{"questions": "oops"}') == []
    assert parse_questions('{"questions": [{"question": "q"}, {"x": 1}]}') == [{"question": "q"}]


def test_draft_assigns_expected_articles_and_stays_reviewable() -> None:
    def respond(system: str, user: str) -> str:
        if "no_answer" in user.split("\n")[0]:
            qs = [{"question": "Как оформить ипотеку?", "type": "no_answer", "answer_facts": []}]
        elif "multi_article" in user.split("\n")[0]:
            qs = [{"question": "Q-multi", "type": "multi_article", "answer_facts": ["28 дней"]}]
        else:
            qs = [{"question": "Q-single", "type": "factual", "answer_facts": ["28 дней"]}]
        return json.dumps({"questions": qs}, ensure_ascii=False)

    items = draft_golden(
        [article("1"), article("2"), article("3", "Глава 2")], FakeLLM(respond)
    ).items
    by_type = {t: [i for i in items if i.type == t] for t in {i.type for i in items}}
    assert [i.expected_articles for i in by_type["factual"]] == [["1"], ["2"], ["3"]]
    assert [i.expected_articles for i in by_type["multi_article"]] == [["1", "2"]]  # same chapter
    assert by_type["no_answer"][0].expected_articles == []
    assert validate_items(items).ok
    assert len({i.id for i in items}) == len(items)


def test_quota_validation_flags_a_bad_mix() -> None:
    items = draft_golden([article("1")], FakeLLM()).items  # FakeLLM refuses -> no questions at all
    assert not validate_quotas(items).ok


def test_facts_that_are_not_in_the_article_are_dropped() -> None:
    def respond(system: str, user: str) -> str:
        qs = [
            {"question": "ok", "type": "factual", "answer_facts": ["28 дней"]},
            {"question": "invented", "type": "factual", "answer_facts": ["35 дней"]},
            {"question": "no facts", "type": "factual", "answer_facts": []},
            {"question": "wrong type", "type": "no_answer", "answer_facts": []},
        ]
        return json.dumps({"questions": qs}, ensure_ascii=False)

    result = draft_golden([article("1")], FakeLLM(respond), no_answer=0)
    assert [i.question for i in result.items if i.type == "factual"] == ["ok"]
    assert len(result.dropped) >= 3
    assert any("invented" in d for d in result.dropped)


def test_overlap_errors_flag_repeated_questions_only() -> None:
    from tkrag.eval.golden import GoldenItem, overlap_errors

    def item(ident: str, question: str) -> GoldenItem:
        return GoldenItem(
            id=ident,
            question=question,
            expected_articles=["1"],
            answer_facts=["x"],
            type="factual",
        )

    tuned = [item("a1", "Сколько дней отпуска?"), item("a2", "Кто платит?")]
    test = [item("t1", "  сколько  ДНЕЙ отпуска? "), item("t2", "Совсем другой вопрос?")]
    assert overlap_errors(test, tuned) == ["t1: the same question as a1 in the other set"]
    assert overlap_errors([item("t2", "Совсем другой вопрос?")], tuned) == []
