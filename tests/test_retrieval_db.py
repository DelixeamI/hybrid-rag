from pathlib import Path

import psycopg
import pytest

from tkrag.ingest.loader import ingest
from tkrag.retrieval.embedder import HashEmbedder
from tkrag.retrieval.reranker import LexicalReranker
from tkrag.retrieval.retriever import Retriever
from tkrag.retrieval.types import Mode

pytestmark = pytest.mark.db

DOC = "test-retrieval"


@pytest.fixture
def retriever(
    conn: psycopg.Connection, fixture_path: Path, hash_embedder: HashEmbedder
) -> Retriever:
    ingest(conn, fixture_path, DOC, hash_embedder)
    return Retriever(conn, DOC, hash_embedder, LexicalReranker(), rrf_k=60, candidates_top_n=20)


def articles(retriever: Retriever, question: str, mode: Mode, k: int = 3) -> list[str]:
    result = retriever.retrieve(question, mode, top_k=k)
    return [h.article_number for h in result.hits]


@pytest.mark.parametrize("mode", ["fts", "bm25"])
def test_sparse_finds_the_article_by_its_terms(retriever: Retriever, mode: Mode) -> None:
    found = articles(
        retriever, "продолжительность ежегодного основного оплачиваемого отпуска", mode
    )
    assert found[0] == "115"


@pytest.mark.parametrize("mode", ["fts", "bm25"])
def test_sparse_uses_russian_stemming(retriever: Retriever, mode: Mode) -> None:
    # "увольнения" in the question, "увольнение" / "увольнению" in the text
    assert "81" in articles(retriever, "запрет увольнения в период болезни", mode, k=5)


def test_bm25_weighs_rare_words_above_common_ones(retriever: Retriever) -> None:
    # "отпуск" is in many articles; "беременности" is only in one: it must decide
    assert articles(retriever, "отпуск беременности", "bm25", k=1) == ["255"]


def test_bm25_hits_carry_scores_in_descending_order(retriever: Retriever) -> None:
    hits = retriever.retrieve("ежегодный оплачиваемый отпуск", "bm25", top_k=5).hits
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True) and len(hits) > 1


def test_dense_returns_cosine_scores_in_order(retriever: Retriever) -> None:
    result = retriever.retrieve("отпуск по беременности и родам", "dense", top_k=5)
    scores = [h.score for h in result.hits]
    assert scores == sorted(scores, reverse=True)
    assert result.hits[0].article_number == "255"


def test_hybrid_contains_hits_from_both_searches(retriever: Retriever) -> None:
    question = "сколько дней отпуска по беременности"
    hybrid = set(articles(retriever, question, "hybrid", k=10))
    sparse = articles(retriever, question, "bm25", k=1)
    dense = articles(retriever, question, "dense", k=1)
    assert set(sparse) | set(dense) <= hybrid


def test_rerank_mode_scores_are_probabilities_and_top_k_is_respected(retriever: Retriever) -> None:
    result = retriever.retrieve("за сколько недель предупредить об увольнении", "hybrid_rerank", 3)
    assert len(result.hits) <= 3
    assert all(0.0 <= h.score <= 1.0 for h in result.hits)
    assert set(result.timings_ms) == {"retrieve", "rerank"}


def test_rerank_mode_without_reranker_fails_loudly(
    conn: psycopg.Connection, hash_embedder: HashEmbedder
) -> None:
    plain = Retriever(conn, DOC, hash_embedder, reranker=None)
    with pytest.raises(ValueError, match="needs a reranker"):
        plain.retrieve("вопрос", "hybrid_rerank")


def test_other_documents_are_invisible(retriever: Retriever) -> None:
    other = Retriever(retriever._conn, "test-no-such-doc", retriever._embedder)
    for mode in ("fts", "bm25", "dense", "hybrid"):
        assert other.retrieve("отпуск", mode).hits == []


def test_query_with_only_stop_words_returns_nothing(retriever: Retriever) -> None:
    assert retriever.retrieve("и в на", "fts").hits == []
    assert retriever.retrieve("и в на", "bm25").hits == []


def test_dense_search_is_not_starved_by_other_documents(
    conn: psycopg.Connection, fixture_path: Path, tmp_path: Path, hash_embedder: HashEmbedder
) -> None:
    """Regression: HNSW post-filters by document, so a big neighbour must not hide a small one."""
    ingest(conn, fixture_path, "test-small", hash_embedder)
    lines = ["Раздел I. Много", "Глава 1. Чужая"]
    for number in range(1, 301):
        lines += [
            f"Статья {number}. Чужая статья {number}",
            f"Текст чужой статьи про отпуск {number}.",
        ]
    big = tmp_path / "big.txt"
    big.write_text("\n".join(lines), encoding="utf-8")
    ingest(conn, big, "test-big", hash_embedder)
    small = Retriever(conn, "test-small", hash_embedder)
    result = small.retrieve("ежегодный основной оплачиваемый отпуск", "dense", top_k=10)
    assert len(result.hits) == 10
    assert result.hits[0].article_number == "115"


def test_bm25_cache_reuses_the_index_and_rebuilds_after_a_re_ingest(
    conn: psycopg.Connection, fixture_path: Path, tmp_path: Path, hash_embedder: HashEmbedder
) -> None:
    from tkrag.retrieval.bm25 import Bm25Cache, query_lexemes

    ingest(conn, fixture_path, "test-cache", hash_embedder)
    cache = Bm25Cache("test-cache")
    first = cache.get(conn)
    assert cache.get(conn) is first  # unchanged corpus: the same object, no reload
    version = cache.version

    edited = tmp_path / "edition2.txt"
    edited.write_text(
        fixture_path.read_text(encoding="utf-8") + "\nСтатья 999. Новая\n\nУникальноеслово.\n",
        encoding="utf-8",
    )
    ingest(conn, edited, "test-cache", hash_embedder)  # a new edition replaces the chunks
    rebuilt = cache.get(conn)
    assert rebuilt is not first and cache.version != version
    lexemes = query_lexemes(conn, "Уникальноеслово")  # the stemmed form Postgres stores
    assert rebuilt.search(lexemes, 5)  # the new article is searchable now
    assert not first.search(lexemes, 5)  # the stale index could not see it
