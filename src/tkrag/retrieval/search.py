"""First-stage searches in plain SQL: dense (pgvector) and Postgres full-text search."""

import re
from dataclasses import replace

import psycopg

from tkrag.retrieval.types import Hit

_COLUMNS = (
    "c.id, c.chunk_key, c.article_number, c.article_title, c.breadcrumbs, c.text, c.part_numbers"
)
_WORD_RE = re.compile(r"\w+")

_DENSE_SQL = f"""
SELECT {_COLUMNS}, 1 - (c.embedding <=> %(vec)s::vector) AS score
FROM chunks c JOIN documents d ON d.id = c.document_id
WHERE d.name = %(doc)s
ORDER BY c.embedding <=> %(vec)s::vector
LIMIT %(limit)s
"""

# OR semantics: a conversational question never contains every word of the article.
_SPARSE_SQL = f"""
SELECT {_COLUMNS}, ts_rank_cd(c.tsv, q.query) AS score
FROM chunks c
JOIN documents d ON d.id = c.document_id,
     to_tsquery('russian', %(query)s) AS q(query)
WHERE d.name = %(doc)s AND c.tsv @@ q.query
ORDER BY score DESC, c.id
LIMIT %(limit)s
"""


def _hit(row: tuple[object, ...]) -> Hit:
    return Hit(
        chunk_id=int(str(row[0])),
        chunk_key=str(row[1]),
        article_number=str(row[2]),
        article_title=str(row[3]),
        breadcrumbs=str(row[4]),
        text=str(row[5]),
        part_numbers=list(row[6]) if isinstance(row[6], list) else [],
        score=float(str(row[7])),
    )


def search_dense(
    conn: psycopg.Connection, document: str, query_vector: list[float], limit: int
) -> list[Hit]:
    from numpy import array

    rows = conn.execute(
        _DENSE_SQL, {"vec": array(query_vector), "doc": document, "limit": limit}
    ).fetchall()
    return [_hit(r) for r in rows]


def build_tsquery(question: str) -> str:
    """'Сколько дней отпуска?' -> 'сколько | дней | отпуска'. Only word characters are kept."""
    words = dict.fromkeys(w for w in _WORD_RE.findall(question.lower()) if len(w) > 1)
    return " | ".join(words)


def search_sparse(conn: psycopg.Connection, document: str, question: str, limit: int) -> list[Hit]:
    query = build_tsquery(question)
    if not query:
        return []
    rows = conn.execute(_SPARSE_SQL, {"query": query, "doc": document, "limit": limit}).fetchall()
    return [_hit(r) for r in rows]


def fetch_hits(conn: psycopg.Connection, scored: list[tuple[int, float]]) -> list[Hit]:
    """Load chunks for (chunk_id, score) pairs, keeping the given order and scores."""
    if not scored:
        return []
    rows = conn.execute(
        f"SELECT {_COLUMNS}, 0.0 AS score FROM chunks c WHERE c.id = ANY(%s)",
        ([cid for cid, _ in scored],),
    ).fetchall()
    by_id = {int(r[0]): _hit(r) for r in rows}
    return [replace(by_id[cid], score=score) for cid, score in scored if cid in by_id]
