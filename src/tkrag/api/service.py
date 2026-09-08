"""The real service: Postgres pool + embedder + reranker + LLM. Built once at startup."""

from typing import Any

from psycopg_pool import ConnectionPool

from tkrag.api.main import Service, create_app
from tkrag.components import make_embedder, make_llm, make_reranker, make_retriever
from tkrag.config import Settings, get_settings
from tkrag.datasets import Dataset, get_dataset
from tkrag.db import connect, prepare_connection
from tkrag.generation.answerer import Answerer
from tkrag.generation.schemas import AskResult
from tkrag.retrieval.bm25 import load_bm25


def build_service(settings: Settings, dataset: Dataset) -> Service:
    with connect(settings.database_url) as startup_conn:  # applies migrations once
        bm25 = load_bm25(startup_conn, dataset.document)  # in-memory index, shared by requests
    pool = ConnectionPool(
        settings.database_url,
        min_size=1,
        max_size=4,
        configure=prepare_connection,
        open=True,
    )
    embedder = make_embedder(settings)
    reranker = make_reranker(settings)
    llm = make_llm(settings, dataset.name)

    def ask(question: str) -> AskResult:
        with pool.connection() as conn:
            retriever = make_retriever(conn, settings, dataset.document, embedder, reranker, bm25)
            answerer = Answerer(retriever, llm, settings.refusal_threshold, settings.context_chunks)
            return answerer.ask(question)

    def health() -> dict[str, Any]:
        with pool.connection() as conn:
            row = conn.execute(
                "SELECT count(*) FROM chunks c JOIN documents d ON d.id = c.document_id "
                "WHERE d.name = %s",
                (dataset.document,),
            ).fetchone()
        chunks = int(row[0]) if row else 0
        return {
            "status": "ok" if chunks else "empty",
            "document": dataset.document,
            "chunks": chunks,
            "llm_model": llm.model,
        }

    return Service(ask=ask, health=health, close=pool.close)


def app_from_env() -> Any:
    """uvicorn entry point: `uvicorn tkrag.api.service:app_from_env --factory`."""
    import os

    settings = get_settings()
    dataset = get_dataset(os.environ.get("TKRAG_DATASET", "fixture"), settings)
    return create_app(lambda: build_service(settings, dataset))
