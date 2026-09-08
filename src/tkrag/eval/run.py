"""Evaluation orchestration: ingest -> retrieval eval (-> answer eval -> gate) -> reports."""

from typing import Any

import psycopg

from tkrag.components import make_retriever
from tkrag.config import Settings
from tkrag.datasets import Dataset
from tkrag.eval.answer_eval import calibrate_threshold, check_gate, evaluate_answers
from tkrag.eval.golden import GoldenItem, load_golden, validate_items
from tkrag.eval.retrieval_eval import evaluate_retrieval
from tkrag.generation.answerer import Answerer
from tkrag.ingest.chunker import chunk_article
from tkrag.ingest.loader import ingest
from tkrag.ingest.parser import live_articles, parse_articles, read_text
from tkrag.llm import LLMClient
from tkrag.retrieval.embedder import Embedder
from tkrag.retrieval.reranker import Reranker
from tkrag.retrieval.types import MODES, Mode


class GoldenInvalidError(ValueError):
    pass


def prepare(
    conn: psycopg.Connection, dataset: Dataset, embedder: Embedder
) -> tuple[list[GoldenItem], dict[str, int]]:
    """Ingest the corpus (no-op if already there) and load + validate the golden set."""
    ingest(conn, dataset.corpus_path, dataset.document, embedder)
    articles = live_articles(parse_articles(read_text(dataset.corpus_path)))
    items = load_golden(dataset.golden_path)
    issues = validate_items(items, {a.number for a in articles})
    if not issues.ok:
        raise GoldenInvalidError("; ".join(issues.errors))
    n_chunks = sum(len(chunk_article(a, embedder.count_tokens)) for a in articles)
    return items, {"n_articles": len(articles), "n_chunks": n_chunks}


def dataset_info(
    dataset: Dataset, items: list[GoldenItem], sizes: dict[str, int]
) -> dict[str, Any]:
    return {
        "name": dataset.name,
        "document": dataset.document,
        "golden": dataset.golden_path.name,
        "n_questions": len(items),
        **sizes,
    }


def config_info(settings: Settings) -> dict[str, Any]:
    return {
        "embedding_model": settings.embedding_model,
        "reranker_model": settings.reranker_model,
        "rrf_k": settings.rrf_k,
        "rrf_dense_weight": settings.rrf_dense_weight,
        "candidates_top_n": settings.candidates_top_n,
        "final_top_k": settings.final_top_k,
    }


def run_retrieval(
    conn: psycopg.Connection,
    dataset: Dataset,
    settings: Settings,
    embedder: Embedder,
    reranker: Reranker | None,
    modes: tuple[Mode, ...] = MODES,
) -> dict[str, Any]:
    items, sizes = prepare(conn, dataset, embedder)
    retriever = make_retriever(conn, settings, dataset.document, embedder, reranker)
    return {
        "dataset": dataset_info(dataset, items, sizes),
        "config": config_info(settings),
        "retrieval": evaluate_retrieval(retriever, items, modes),
    }


def run_full(
    conn: psycopg.Connection,
    dataset: Dataset,
    settings: Settings,
    embedder: Embedder,
    reranker: Reranker,
    llm: LLMClient,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Retrieval eval + answer eval (+ CI gate when thresholds are given)."""
    results = run_retrieval(conn, dataset, settings, embedder, reranker)
    items = load_golden(dataset.golden_path)
    retriever = make_retriever(conn, settings, dataset.document, embedder, reranker)
    answerer = Answerer(retriever, llm, settings.refusal_threshold, settings.context_chunks)
    results["answers"] = evaluate_answers(answerer, items, llm.model, settings.refusal_threshold)
    scored = [(i["type"] == "no_answer", i["best_score"]) for i in results["answers"]["items"]]
    calibration = calibrate_threshold(scored)
    results["calibration"] = {"chosen": calibration.threshold, "curve": calibration.curve}
    if thresholds is not None:
        results["gate"] = check_gate(results, thresholds)
    return results
