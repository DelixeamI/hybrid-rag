import json

from tkrag.generation.answerer import DEGRADED_TEXT, NO_ANSWER_TEXT, Answerer, parse_reply
from tkrag.llm import FakeLLM
from tkrag.retrieval.types import Hit, Mode, Retrieval

H115 = Hit(
    1,
    "115#0",
    "115",
    "Продолжительность отпуска",
    "ТК РФ > Глава 19. Отпуска > Статья 115. Продолжительность отпуска",
    "Ежегодный основной оплачиваемый отпуск предоставляется работникам продолжительностью 28 календарных дней.",
    [],
    0.9,
)


class StubSearcher:
    def __init__(self, hits: list[Hit]) -> None:
        self._hits = hits

    def retrieve(self, question: str, mode: Mode, top_k: int = 5) -> Retrieval:
        best = self._hits[0].score if self._hits else 0.0
        return Retrieval(self._hits[:top_k], mode, best, {"retrieve": 1.0, "rerank": 2.0})


def reply(**fields: object) -> str:
    base: dict[str, object] = {"answer": "", "citations": [], "refused": False, "reason": None}
    return json.dumps({**base, **fields}, ensure_ascii=False)


GOOD = reply(
    answer="Ежегодный отпуск длится 28 календарных дней [1].",
    citations=[{"source": 1, "quote": "продолжительностью 28 календарных дней"}],
)


def make(llm: FakeLLM, hits: list[Hit] | None = None, threshold: float = 0.5) -> Answerer:
    return Answerer(StubSearcher([H115] if hits is None else hits), llm, threshold)


def test_good_answer_is_ok_and_maps_citations_to_real_chunks() -> None:
    result = make(FakeLLM(lambda s, u: GOOD)).ask("Сколько дней отпуска?")
    assert result.status == "ok"
    (citation,) = result.answer.citations
    assert (citation.chunk_id, citation.article) == ("115#0", "115")
    assert set(result.timings_ms) == {"retrieve", "rerank", "generate", "validate"}


def test_weak_retrieval_refuses_without_calling_the_llm() -> None:
    llm = FakeLLM(lambda s, u: GOOD)
    weak = Hit(**{**H115.__dict__, "score": 0.1})
    result = make(llm, [weak], threshold=0.5).ask("Какой налог?")
    assert result.status == "refused"
    assert result.answer.reason == "low_retrieval_score"
    assert result.answer.answer == NO_ANSWER_TEXT
    assert llm.calls == []


def test_no_hits_refuses() -> None:
    assert make(FakeLLM(), hits=[]).ask("вопрос").status == "refused"


def test_model_can_refuse_itself() -> None:
    llm = FakeLLM(lambda s, u: reply(refused=True, reason="в контексте нет ответа"))
    result = make(llm).ask("вопрос")
    assert result.status == "refused"
    assert result.answer.reason == "в контексте нет ответа"


def test_invented_number_degrades_to_the_relevant_articles() -> None:
    bad = reply(
        answer="Отпуск длится 35 календарных дней [1].",
        citations=[{"source": 1, "quote": "продолжительностью 28 календарных дней"}],
    )
    result = make(FakeLLM(lambda s, u: bad)).ask("Сколько дней отпуска?")
    assert result.status == "unverified"
    assert result.answer.answer == DEGRADED_TEXT
    assert "35" not in result.answer.answer
    assert [c.article for c in result.answer.citations] == ["115"]
    assert any("35 day" in issue for issue in result.issues)


def test_invalid_json_degrades_instead_of_crashing() -> None:
    result = make(FakeLLM(lambda s, u: "Конечно! Вот ответ: 28 дней")).ask("вопрос")
    assert result.status == "unverified"
    assert result.answer.reason == "invalid_llm_output"


def test_citation_to_a_non_existent_fragment_is_dropped_then_fails_validation() -> None:
    bad = reply(answer="28 календарных дней [5].", citations=[{"source": 5, "quote": "x"}])
    assert make(FakeLLM(lambda s, u: bad)).ask("вопрос").status == "unverified"


def test_markdown_fenced_json_is_accepted() -> None:
    assert parse_reply(f"```json\n{GOOD}\n```") is not None
    assert parse_reply("not json") is None
    assert parse_reply('{"citations": "oops"}') is None


def test_prompt_contains_numbered_fragments_and_the_question() -> None:
    llm = FakeLLM(lambda s, u: GOOD)
    make(llm).ask("Сколько дней отпуска?")
    system, user = llm.calls[0]
    assert "[1] ТК РФ > Глава 19. Отпуска > Статья 115." in user
    assert user.endswith("ВОПРОС: Сколько дней отпуска?")
    assert "refused" in system
