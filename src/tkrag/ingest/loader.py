"""Corpus file -> Postgres. Idempotent by a fingerprint of everything the index is built from."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import psycopg
from numpy import array

from tkrag.ingest.chunker import DEFAULT_MAX_TOKENS, Chunk, chunk_article
from tkrag.ingest.fingerprint import index_fingerprint
from tkrag.ingest.parser import live_articles, parse_articles, read_text
from tkrag.retrieval.embedder import Embedder

_BATCH = 32


class IndexMismatchError(RuntimeError):
    """The embedder does not fit the stored vector column (different dimension)."""


@dataclass(frozen=True)
class IngestReport:
    document: str
    sha256: str
    skipped: bool
    rebuilt: bool = False  # the same file was ingested before, but with other index settings
    articles_total: int = 0
    articles_repealed: int = 0
    chunks: int = 0

    def summary(self) -> str:
        if self.skipped:
            return (
                f"{self.document}: edition {self.sha256[:12]} already ingested with the same "
                "settings, nothing to do"
            )
        indexed = self.articles_total - self.articles_repealed
        why = " (rebuilt: the index settings changed)" if self.rebuilt else ""
        return (
            f"{self.document}{why}: {self.articles_total} articles found, "
            f"{self.articles_repealed} repealed (skipped), {indexed} indexed as "
            f"{self.chunks} chunks"
        )


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ingest(
    conn: psycopg.Connection,
    path: Path,
    document: str,
    embedder: Embedder,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> IngestReport:
    sha = file_sha256(path)
    fingerprint = index_fingerprint(sha, embedder.name, embedder.dim, max_tokens)
    _check_dimension(conn, embedder)
    stored = conn.execute(
        "SELECT sha256, index_fingerprint FROM documents WHERE name = %s", (document,)
    ).fetchall()
    if any(row[1] == fingerprint for row in stored):
        return IngestReport(document, sha, skipped=True)
    rebuilt = any(row[0] == sha for row in stored)  # same file, different index settings

    articles = parse_articles(read_text(path))
    chunks = [
        c
        for a in live_articles(articles)
        for c in chunk_article(a, embedder.count_tokens, max_tokens)
    ]
    vectors = _embed_all(chunks, embedder)

    with conn.transaction():
        conn.execute("DELETE FROM documents WHERE name = %s", (document,))  # old edition
        doc_id = conn.execute(
            "INSERT INTO documents (name, sha256, index_fingerprint) VALUES (%s, %s, %s) "
            "RETURNING id",
            (document, sha, fingerprint),
        ).fetchone()
        assert doc_id is not None
        _insert_chunks(conn, doc_id[0], chunks, vectors)
    conn.commit()  # conn.transaction() is only a savepoint when a transaction is already open
    return IngestReport(
        document,
        sha,
        skipped=False,
        rebuilt=rebuilt,
        articles_total=len(articles),
        articles_repealed=len(articles) - len(live_articles(articles)),
        chunks=len(chunks),
    )


def column_dimension(conn: psycopg.Connection) -> int:
    """Dimension of chunks.embedding as declared in the schema (pgvector keeps it in atttypmod)."""
    row = conn.execute(
        "SELECT atttypmod FROM pg_attribute WHERE attrelid = 'chunks'::regclass "
        "AND attname = 'embedding'"
    ).fetchone()
    assert row is not None, "chunks.embedding is missing: run the migrations"
    return int(row[0])


def _check_dimension(conn: psycopg.Connection, embedder: Embedder) -> None:
    declared = column_dimension(conn)
    if declared != embedder.dim:
        raise IndexMismatchError(
            f"the chunks table stores {declared}-dimensional vectors, but embedder "
            f"{embedder.name!r} produces {embedder.dim}. Add a migration that changes "
            f"chunks.embedding to vector({embedder.dim}) (and recreate the HNSW index), "
            "then ingest again."
        )


def _embed_all(chunks: list[Chunk], embedder: Embedder) -> list[list[float]]:
    vectors: list[list[float]] = []
    for start in range(0, len(chunks), _BATCH):
        batch = chunks[start : start + _BATCH]
        vectors.extend(embedder.embed_passages([c.embedding_text for c in batch]))
    return vectors


def _insert_chunks(
    conn: psycopg.Connection, doc_id: int, chunks: list[Chunk], vectors: list[list[float]]
) -> None:
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO chunks (document_id, chunk_key, article_number, article_title, section,
                                chapter, part_numbers, breadcrumbs, text, token_count, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """,
            [
                (
                    doc_id,
                    c.key,
                    c.article_number,
                    c.article_title,
                    c.section,
                    c.chapter,
                    list(c.part_numbers),
                    c.breadcrumbs,
                    c.text,
                    c.token_count,
                    array(v),
                )
                for c, v in zip(chunks, vectors, strict=True)
            ],
        )
