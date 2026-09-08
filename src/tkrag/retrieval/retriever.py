"""Retriever: one entry point, five modes (fts, bm25, dense, hybrid, hybrid_rerank)."""

import time
from typing import Protocol

import psycopg

from tkrag.retrieval.bm25 import Bm25Index, load_bm25, query_lexemes
from tkrag.retrieval.embedder import Embedder
from tkrag.retrieval.reranker import Reranker
from tkrag.retrieval.rrf import fuse
from tkrag.retrieval.search import fetch_hits, search_dense, search_sparse
from tkrag.retrieval.types import Hit, Mode, Retrieval

FIRST_STAGE_POOL = 50  # candidates taken from each first-stage search before fusion


class Searcher(Protocol):
    """What the answer step needs from retrieval; tests plug in a stub."""

    def retrieve(self, question: str, mode: Mode, top_k: int = 5) -> Retrieval: ...


class Retriever:
    def __init__(
        self,
        conn: psycopg.Connection,
        document: str,
        embedder: Embedder,
        reranker: Reranker | None = None,
        rrf_k: int = 60,
        candidates_top_n: int = 20,
        rrf_dense_weight: float = 1.0,
        bm25: Bm25Index | None = None,
    ) -> None:
        self._conn = conn
        self._document = document
        self._embedder = embedder
        self._reranker = reranker
        self._rrf_k = rrf_k
        self._candidates = candidates_top_n
        self._dense_weight = rrf_dense_weight
        self._bm25 = bm25

    def retrieve(self, question: str, mode: Mode, top_k: int = 5) -> Retrieval:
        timings: dict[str, float] = {}
        hits = self._first_stage(question, mode, timings)
        if mode == "hybrid_rerank":
            if self._reranker is None:
                raise ValueError("mode 'hybrid_rerank' needs a reranker")
            start = time.perf_counter()
            hits = self._reranker.rerank(question, hits[: self._candidates], top_k)
            timings["rerank"] = (time.perf_counter() - start) * 1000
        else:
            hits = hits[:top_k]
        return Retrieval(hits, mode, hits[0].score if hits else 0.0, timings)

    def _bm25_hits(self, question: str) -> list[Hit]:
        if self._bm25 is None:  # built once per Retriever; share an index to avoid rebuilding
            self._bm25 = load_bm25(self._conn, self._document)
        scored = self._bm25.search(query_lexemes(self._conn, question), FIRST_STAGE_POOL)
        return fetch_hits(self._conn, scored)

    def _first_stage(self, question: str, mode: Mode, timings: dict[str, float]) -> list[Hit]:
        start = time.perf_counter()
        if mode == "fts":
            hits = search_sparse(self._conn, self._document, question, FIRST_STAGE_POOL)
        elif mode == "bm25":
            hits = self._bm25_hits(question)
        else:
            vector = self._embedder.embed_query(question)
            dense = search_dense(self._conn, self._document, vector, FIRST_STAGE_POOL)
            if mode == "dense":
                hits = dense
            else:
                sparse = self._bm25_hits(question)
                hits = fuse([dense, sparse], self._rrf_k, [self._dense_weight, 1.0])
        timings["retrieve"] = (time.perf_counter() - start) * 1000
        return hits
