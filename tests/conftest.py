from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest

from tkrag.config import Settings
from tkrag.datasets import FIXTURE
from tkrag.db import connect
from tkrag.ingest.parser import Article, parse_articles, read_text
from tkrag.retrieval.embedder import HashEmbedder


@pytest.fixture(scope="session")
def fixture_path() -> Path:
    return FIXTURE.corpus_path


@pytest.fixture(scope="session")
def articles(fixture_path: Path) -> list[Article]:
    return parse_articles(read_text(fixture_path))


@pytest.fixture
def hash_embedder() -> HashEmbedder:
    return HashEmbedder()


@pytest.fixture
def conn() -> Iterator[psycopg.Connection]:
    """A real Postgres connection (tests marked `db`). Cleans up documents it created."""
    dsn = Settings().database_url
    try:
        connection = connect(dsn)
    except psycopg.OperationalError as exc:
        pytest.skip(f"Postgres is not reachable: {exc}")
    try:
        yield connection
    finally:
        connection.rollback()
        connection.execute("DELETE FROM documents WHERE name LIKE 'test-%'")
        connection.commit()
        connection.close()
