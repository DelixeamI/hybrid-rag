import itertools
import re

from tkrag.ingest.chunker import chunk_article
from tkrag.ingest.parser import Article, Part


def words(text: str) -> int:
    return len(re.findall(r"\w+", text))


def make_article(parts: list[Part]) -> Article:
    return Article("7", "Тестовая статья", None, "Глава 1. Тест", tuple(parts), False)


def test_short_article_is_a_single_chunk(articles: list[Article]) -> None:
    a115 = next(a for a in articles if a.number == "115")
    chunks = chunk_article(a115, words, max_tokens=500)
    assert len(chunks) == 1
    assert chunks[0].key == "115#0"
    assert chunks[0].breadcrumbs == a115.breadcrumbs
    assert chunks[0].part_numbers == ()
    assert chunks[0].embedding_text.startswith("ТК РФ > Глава 19. Отпуска")


def test_chunk_records_part_numbers(articles: list[Article]) -> None:
    a81 = next(a for a in articles if a.number == "81")
    (chunk,) = chunk_article(a81, words, max_tokens=500)
    assert chunk.part_numbers == ("1", "2", "3", "4", "5", "6")


def test_long_article_is_split_by_parts_with_one_part_overlap() -> None:
    parts = [Part(str(i), " ".join(["слово"] * 30)) for i in range(1, 7)]  # 6 parts x ~31 words
    chunks = chunk_article(make_article(parts), words, max_tokens=100)
    assert len(chunks) > 1
    for previous, current in itertools.pairwise(chunks):
        assert previous.part_numbers[-1] == current.part_numbers[0]  # overlap of one part
    covered = {n for c in chunks for n in c.part_numbers}
    assert covered == {"1", "2", "3", "4", "5", "6"}
    assert all(c.token_count <= 100 for c in chunks)


def test_every_chunk_keeps_breadcrumbs_and_unique_keys() -> None:
    parts = [Part(str(i), " ".join(["слово"] * 30)) for i in range(1, 7)]
    chunks = chunk_article(make_article(parts), words, max_tokens=100)
    assert all(c.breadcrumbs.endswith("Статья 7. Тестовая статья") for c in chunks)
    assert len({c.key for c in chunks}) == len(chunks)


def test_single_oversized_part_is_cut_at_sentences() -> None:
    sentences = " ".join(f"Предложение номер {i} содержит несколько слов." for i in range(40))
    chunks = chunk_article(make_article([Part("1", sentences)]), words, max_tokens=60)
    assert len(chunks) > 1
    assert all(c.token_count <= 60 for c in chunks)
    assert all(c.part_numbers == ("1",) for c in chunks)


def test_chunking_always_terminates_when_each_part_fills_a_window() -> None:
    parts = [Part(str(i), " ".join(["слово"] * 45)) for i in range(1, 5)]
    chunks = chunk_article(make_article(parts), words, max_tokens=60)
    assert len(chunks) == 4
