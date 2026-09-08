"""Parser checks against the real Labor Code in data/raw/ (skipped when the file is absent).

They pin facts that must hold for any reasonable edition of the code: the structure the parser
has to recognise, and a few well-known norms. They do not pin the full text.
"""

from pathlib import Path

import pytest

from tkrag.ingest.chunker import chunk_article
from tkrag.ingest.parser import Article, live_articles, parse_articles, read_text

CORPUS = Path(__file__).resolve().parents[1] / "data" / "raw" / "tk_rf.txt"

pytestmark = pytest.mark.skipif(not CORPUS.exists(), reason="data/raw/tk_rf.txt is not present")


@pytest.fixture(scope="module")
def code() -> dict[str, Article]:
    return {a.number: a for a in parse_articles(read_text(CORPUS))}


def words(text: str) -> int:
    return int(len(text.split()) * 1.8)  # rough token proxy, enough for a size sanity check


def test_the_code_is_parsed_as_hundreds_of_articles(code: dict[str, Article]) -> None:
    assert len(code) > 500
    live = live_articles(list(code.values()))
    assert 0 < len(code) - len(live) < 30  # a handful of repealed articles are skipped
    assert all(a.chapter and a.section for a in live)
    assert all(a.parts for a in live), "every live article must have text"


def test_article_115_has_three_parts_and_the_disabled_get_30_days(code: dict[str, Article]) -> None:
    a115 = code["115"]
    assert [p.number for p in a115.parts] == ["1", "2", "3"]
    assert "28 календарных дней" in a115.parts[0].text
    assert "инвалидам" in a115.parts[2].text and "30 календарных дней" in a115.parts[2].text
    assert a115.chapter is not None and a115.chapter.startswith("Глава 19.")


def test_article_81_part_six_forbids_dismissal_during_illness_or_leave(
    code: dict[str, Article],
) -> None:
    a81 = code["81"]
    assert a81.parts[0].text.startswith("Трудовой договор может быть расторгнут работодателем")
    assert "1) ликвидации организации" in a81.parts[0].text  # list items stay in part one
    part_six = a81.parts[5]
    assert part_six.number == "6"
    assert part_six.text.startswith("Не допускается увольнение работника по инициативе")
    assert "временной нетрудоспособности" in part_six.text


def test_hyphenated_and_dotted_article_numbers_exist(code: dict[str, Article]) -> None:
    assert {"84.1", "341.1-1", "312.1"} <= set(code)
    assert code["341.1-1"].title.startswith("Организации, имеющие право")


def test_known_repealed_articles_are_not_indexed(code: dict[str, Article]) -> None:
    repealed = {a.number for a in code.values() if a.repealed}
    assert "7" in repealed
    assert all(a.number not in repealed for a in live_articles(list(code.values())))


def test_every_live_article_chunks_into_pieces_that_fit(code: dict[str, Article]) -> None:
    for article in live_articles(list(code.values())):
        chunks = chunk_article(article, words, max_tokens=500)
        assert chunks
        # The proxy rounds each piece down, so breadcrumbs + text can come out a couple of tokens
        # above the budget; with the real e5 tokenizer the largest chunk is 498 (limit 512).
        assert all(c.token_count <= 500 + 3 for c in chunks), article.number
        covered = {n for c in chunks for n in c.part_numbers}
        assert covered == {p.number for p in article.parts if p.number}, article.number
