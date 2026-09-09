"""BM25 over the lexemes Postgres produces for the 'russian' text search configuration.

Why not Postgres ranking: `ts_rank_cd` ignores inverse document frequency, so a common word like
"отпуск" weighs as much as a rare one, and long articles win by sheer length. The experiment in
`scripts/experiment_hybrid.py` shows how much that costs. Postgres still does the language work
(stemming, stop words); only the scoring formula is ours. The index lives in memory, which is fine
for one code (about 700 chunks) and is the thing to replace (pg_search, Elasticsearch) at scale.
"""

import math
import threading
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field

import psycopg


@dataclass
class Bm25Index:
    """Textbook BM25 (Robertson/Sparck Jones idf, k1=1.2, b=0.75)."""

    k1: float = 1.2
    b: float = 0.75
    term_freqs: dict[int, Counter[str]] = field(default_factory=dict)
    doc_freq: Counter[str] = field(default_factory=Counter)
    lengths: dict[int, int] = field(default_factory=dict)

    def add(self, chunk_id: int, lexeme_counts: dict[str, int]) -> None:
        self.term_freqs[chunk_id] = Counter(lexeme_counts)
        self.lengths[chunk_id] = sum(lexeme_counts.values())
        self.doc_freq.update(lexeme_counts.keys())

    def search(self, query_lexemes: Sequence[str], limit: int) -> list[tuple[int, float]]:
        """(chunk_id, score) pairs, best first. Chunks without any query term are not returned."""
        total = len(self.term_freqs)
        average = sum(self.lengths.values()) / total if total else 1.0
        scores: dict[int, float] = {}
        for term in set(query_lexemes):
            df = self.doc_freq.get(term, 0)
            if not df:
                continue
            idf = math.log(1 + (total - df + 0.5) / (df + 0.5))
            for chunk_id, counts in self.term_freqs.items():
                tf = counts.get(term, 0)
                if not tf:
                    continue
                norm = tf + self.k1 * (1 - self.b + self.b * self.lengths[chunk_id] / average)
                scores[chunk_id] = scores.get(chunk_id, 0.0) + idf * tf * (self.k1 + 1) / norm
        ranked = sorted(scores, key=lambda cid: (-scores[cid], cid))
        return [(cid, scores[cid]) for cid in ranked[:limit]]


def load_bm25(conn: psycopg.Connection, document: str) -> Bm25Index:
    """Build the index from the stored chunks (title and text, as the tsvector column does)."""
    rows = conn.execute(
        """
        SELECT c.id, u.lexeme, coalesce(array_length(u.positions, 1), 1)
        FROM chunks c JOIN documents d ON d.id = c.document_id,
             LATERAL unnest(to_tsvector('russian', c.article_title || ' ' || c.text)) AS u
        WHERE d.name = %s
        """,
        (document,),
    ).fetchall()
    per_chunk: dict[int, dict[str, int]] = defaultdict(dict)
    for chunk_id, lexeme, count in rows:
        per_chunk[int(chunk_id)][str(lexeme)] = int(count)
    index = Bm25Index()
    for chunk_id, counts in per_chunk.items():
        index.add(chunk_id, counts)
    return index


def query_lexemes(conn: psycopg.Connection, question: str) -> list[str]:
    """Lexemes of the question after Postgres stemming and stop-word removal."""
    rows = conn.execute("SELECT lexeme FROM unnest(to_tsvector('russian', %s))", (question,))
    return [str(r[0]) for r in rows.fetchall()]


def document_version(conn: psycopg.Connection, document: str) -> int | None:
    """Id of the newest stored edition. A re-ingest inserts a new row, so the id changes."""
    row = conn.execute("SELECT max(id) FROM documents WHERE name = %s", (document,)).fetchone()
    return int(row[0]) if row and row[0] is not None else None


class Bm25Cache:
    """One shared BM25 index that rebuilds itself when the corpus is re-ingested.

    A long-running API process would otherwise keep ranking with the index it built at startup
    after `ingest` replaced the chunks in the database. Each request checks one cheap query.
    """

    def __init__(self, document: str) -> None:
        self._document = document
        self._index: Bm25Index | None = None
        self._version: int | None = None
        self._lock = threading.Lock()

    @property
    def version(self) -> int | None:
        return self._version

    def get(self, conn: psycopg.Connection) -> Bm25Index:
        current = document_version(conn, self._document)
        with self._lock:
            if self._index is None or current != self._version:
                self._index = load_bm25(conn, self._document)
                self._version = current
            return self._index
