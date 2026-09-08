"""Corpus file -> Postgres. Idempotent by the sha256 of the source file."""

import hashlib
from dataclasses import dataclass
from pathlib import Path

import psycopg
from numpy import array

from tkrag.ingest.chunker import DEFAULT_MAX_TOKENS, Chunk, chunk_article
from tkrag.ingest.parser import live_articles, parse_articles, read_text
from tkrag.retrieval.embedder import Embedder

_BATCH = 32


@dataclass(frozen=True)
class IngestReport:
    document: str
    sha256: str
    skipped: bool
    articles_total: int = 0
    articles_repealed: int = 0
    chunks: int = 0

    def summary(self) -> str:
        if self.skipped:
            return f"{self.document}: edition {self.sha256[:12]} already ingested, nothing to do"
        indexed = self.articles_total - self.articles_repealed
        return (
            f"{self.document}: {self.articles_total} articles found, "
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
    known = conn.execute(
        "SELECT 1 FROM documents WHERE name = %s AND sha256 = %s", (document, sha)
    ).fetchone()
    if known:
        return IngestReport(document, sha, skipped=True)

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
            "INSERT INTO documents (name, sha256) VALUES (%s, %s) RETURNING id", (document, sha)
        ).fetchone()
        assert doc_id is not None
        _insert_chunks(conn, doc_id[0], chunks, vectors)
    conn.commit()  # conn.transaction() is only a savepoint when a transaction is already open
    return IngestReport(
        document,
        sha,
        skipped=False,
        articles_total=len(articles),
        articles_repealed=len(articles) - len(live_articles(articles)),
        chunks=len(chunks),
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
