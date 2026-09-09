from pathlib import Path

import psycopg
import pytest

from tkrag.ingest.loader import IndexMismatchError, ingest
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


class NamedEmbedder(HashEmbedder):
    """Same vectors as HashEmbedder under another model name."""

    def __init__(self, name: str, dim: int = 384) -> None:
        super().__init__(dim)
        self.name = name


def test_a_different_embedding_model_rebuilds_the_index(
    conn: psycopg.Connection, fixture_path: Path
) -> None:
    first = ingest(conn, fixture_path, "test-model", NamedEmbedder("model-a"))
    again = ingest(conn, fixture_path, "test-model", NamedEmbedder("model-a"))
    switched = ingest(conn, fixture_path, "test-model", NamedEmbedder("model-b"))
    assert not first.skipped and again.skipped
    assert not switched.skipped and switched.rebuilt  # same file, new model: not skipped
    assert count(conn, "test-model") == switched.chunks
    assert "rebuilt" in switched.summary()


def test_other_chunk_size_rebuilds_the_index(
    conn: psycopg.Connection, fixture_path: Path, hash_embedder: HashEmbedder
) -> None:
    small = ingest(conn, fixture_path, "test-size", hash_embedder, max_tokens=60)
    large = ingest(conn, fixture_path, "test-size", hash_embedder, max_tokens=500)
    assert not large.skipped and large.rebuilt
    assert small.chunks > large.chunks  # smaller limit, more pieces
    assert count(conn, "test-size") == large.chunks


def test_an_index_without_a_fingerprint_is_rebuilt_once(
    conn: psycopg.Connection, fixture_path: Path, hash_embedder: HashEmbedder
) -> None:
    ingest(conn, fixture_path, "test-legacy", hash_embedder)
    conn.execute("UPDATE documents SET index_fingerprint = NULL WHERE name = 'test-legacy'")
    conn.commit()  # what rows ingested before the fingerprint column look like
    assert not ingest(conn, fixture_path, "test-legacy", hash_embedder).skipped
    assert ingest(conn, fixture_path, "test-legacy", hash_embedder).skipped


def test_an_embedder_of_another_dimension_fails_with_a_clear_message(
    conn: psycopg.Connection, fixture_path: Path
) -> None:
    with pytest.raises(IndexMismatchError, match=r"384-dimensional.*produces 128"):
        ingest(conn, fixture_path, "test-dim", HashEmbedder(dim=128))
    assert count(conn, "test-dim") == 0  # nothing half-written
