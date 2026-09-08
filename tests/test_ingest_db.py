from pathlib import Path

import psycopg
import pytest

from tkrag.ingest.loader import ingest
from tkrag.retrieval.embedder import HashEmbedder

pytestmark = pytest.mark.db


def count(conn: psycopg.Connection, document: str) -> int:
    row = conn.execute(
        "SELECT count(*) FROM chunks c JOIN documents d ON d.id = c.document_id WHERE d.name = %s",
        (document,),
    ).fetchone()
    assert row is not None
    return int(row[0])


def test_ingest_stores_chunks_without_repealed_articles(
    conn: psycopg.Connection, fixture_path: Path, hash_embedder: HashEmbedder
) -> None:
    report = ingest(conn, fixture_path, "test-fixture", hash_embedder)
    assert not report.skipped
    assert report.articles_repealed == 1
    assert count(conn, "test-fixture") == report.chunks > 0
    repealed = conn.execute("SELECT count(*) FROM chunks WHERE article_number = '83.1'").fetchone()
    assert repealed == (0,)


def test_ingest_is_idempotent(
    conn: psycopg.Connection, fixture_path: Path, hash_embedder: HashEmbedder
) -> None:
    first = ingest(conn, fixture_path, "test-idem", hash_embedder)
    second = ingest(conn, fixture_path, "test-idem", hash_embedder)
    assert second.skipped
    assert count(conn, "test-idem") == first.chunks
    docs = conn.execute("SELECT count(*) FROM documents WHERE name = 'test-idem'").fetchone()
    assert docs == (1,)


def test_new_edition_replaces_the_old_one(
    conn: psycopg.Connection, fixture_path: Path, tmp_path: Path, hash_embedder: HashEmbedder
) -> None:
    ingest(conn, fixture_path, "test-edition", hash_embedder)
    edited = tmp_path / "edition2.txt"
    edited.write_text(
        fixture_path.read_text(encoding="utf-8") + "\nСтатья 999. Новая\n\nТекст.\n",
        encoding="utf-8",
    )
    report = ingest(conn, edited, "test-edition", hash_embedder)
    assert not report.skipped
    docs = conn.execute("SELECT count(*) FROM documents WHERE name = 'test-edition'").fetchone()
    assert docs == (1,)
    assert count(conn, "test-edition") == report.chunks


def test_chunk_columns_and_indexes(
    conn: psycopg.Connection, fixture_path: Path, hash_embedder: HashEmbedder
) -> None:
    ingest(conn, fixture_path, "test-cols", hash_embedder)
    row = conn.execute(
        "SELECT breadcrumbs, part_numbers, tsv IS NOT NULL FROM chunks c "
        "JOIN documents d ON d.id = c.document_id "
        "WHERE d.name = 'test-cols' AND article_number = '81'"
    ).fetchone()
    assert row is not None
    assert row[0].endswith("Статья 81. Расторжение трудового договора по инициативе работодателя")
    assert row[1] == ["1", "2", "3", "4", "5", "6"]
    assert row[2] is True
    indexes = {
        r[0] for r in conn.execute("SELECT indexname FROM pg_indexes WHERE tablename = 'chunks'")
    }
    assert {"chunks_tsv_gin", "chunks_embedding_hnsw"} <= indexes


def test_ingest_is_visible_to_other_connections(
    conn: psycopg.Connection, fixture_path: Path, hash_embedder: HashEmbedder
) -> None:
    """Regression: a forgotten commit passed all tests on one connection but lost the data."""
    from tkrag.config import Settings

    report = ingest(conn, fixture_path, "test-commit", hash_embedder)
    with psycopg.connect(Settings().database_url) as other:
        assert count(other, "test-commit") == report.chunks > 0
