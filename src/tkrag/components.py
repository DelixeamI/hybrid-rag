"""Factories that turn Settings into live objects (heavy imports stay inside)."""

from pathlib import Path

import psycopg

from tkrag.config import Settings
from tkrag.datasets import ROOT
from tkrag.llm import FakeLLM, LLMClient, OpenAICompatibleLLM, RecordedLLM, RecordingLLM
from tkrag.retrieval.bm25 import Bm25Index
from tkrag.retrieval.embedder import Embedder, SentenceTransformerEmbedder
from tkrag.retrieval.reranker import CrossEncoderReranker, Reranker
from tkrag.retrieval.retriever import Retriever


def make_embedder(settings: Settings) -> Embedder:
    return SentenceTransformerEmbedder(settings.embedding_model, settings.embedding_dim)


def make_reranker(settings: Settings) -> Reranker:
    return CrossEncoderReranker(settings.reranker_model)


def cassettes_dir(settings: Settings, dataset: str = "fixture") -> Path:
    """Fixture cassettes live in the base directory, full-corpus ones in its `full/` subfolder."""
    base = ROOT / settings.cassettes_dir
    return base if dataset == "fixture" else base / dataset


def make_real_llm(settings: Settings) -> OpenAICompatibleLLM:
    return OpenAICompatibleLLM(
        settings.llm_base_url,
        settings.llm_model,
        settings.llm_api_key,
        min_interval=settings.llm_min_interval,
    )


def make_llm(settings: Settings, dataset: str = "fixture") -> LLMClient:
    if settings.llm_mode == "fake":
        return FakeLLM()
    if settings.llm_mode == "recorded":
        return RecordedLLM(cassettes_dir(settings, dataset), settings.llm_model)
    return make_real_llm(settings)


def make_recording_llm(
    settings: Settings, dataset: str = "fixture", resume: bool = False
) -> LLMClient:
    """Real model whose every answer is written to a cassette (`make record`)."""
    return RecordingLLM(make_real_llm(settings), cassettes_dir(settings, dataset), resume)


def make_retriever(
    conn: psycopg.Connection,
    settings: Settings,
    document: str,
    embedder: Embedder,
    reranker: Reranker | None,
    bm25: Bm25Index | None = None,
) -> Retriever:
    """The retriever configured as the system runs it (RRF k and dense weight from settings)."""
    return Retriever(
        conn,
        document,
        embedder,
        reranker,
        settings.rrf_k,
        settings.candidates_top_n,
        settings.rrf_dense_weight,
        bm25,
    )
