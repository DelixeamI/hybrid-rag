from dataclasses import dataclass, field
from typing import Literal

# fts = Postgres ts_rank_cd (no IDF), bm25 = BM25 over Postgres lexemes, hybrid = dense + bm25
Mode = Literal["fts", "bm25", "dense", "hybrid", "hybrid_rerank"]
MODES: tuple[Mode, ...] = ("fts", "bm25", "dense", "hybrid", "hybrid_rerank")


@dataclass(frozen=True)
class Hit:
    chunk_id: int
    chunk_key: str
    article_number: str
    article_title: str
    breadcrumbs: str
    text: str
    part_numbers: list[str]
    score: float

    @property
    def context_text(self) -> str:
        return f"{self.breadcrumbs}\n{self.text}"


@dataclass(frozen=True)
class Retrieval:
    hits: list[Hit]
    mode: Mode
    # Score of the best hit in the units of the mode: reranker probability for
    # hybrid_rerank, cosine for dense, RRF for hybrid, BM25 or ts_rank_cd for the sparse modes.
    best_score: float
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def total_ms(self) -> float:
        return sum(self.timings_ms.values())
