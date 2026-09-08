from tkrag.eval.report import README_END, README_START, render_markdown, update_readme

RESULTS = {
    "dataset": {"name": "fixture", "n_articles": 2, "n_chunks": 2, "n_questions": 3, "golden": "g"},
    "config": {
        "embedding_model": "e",
        "reranker_model": "r",
        "rrf_k": 60,
        "candidates_top_n": 20,
        "final_top_k": 5,
    },
    "retrieval": {
        mode: {
            "overall": {"recall@1": 0.5, "recall@5": 1.0, "recall@10": 1.0, "mrr": 0.75, "n": 2.0},
            "by_type": {
                "factual": {
                    "recall@1": 0.5,
                    "recall@5": 1.0,
                    "recall@10": 1.0,
                    "mrr": 0.75,
                    "n": 2.0,
                }
            },
            "latency_ms": {"p50": 1.0, "p95": 2.0},
        }
        for mode in ("bm25", "dense", "hybrid", "hybrid_rerank")
    },
}


def test_markdown_contains_every_mode_and_the_synthetic_warning() -> None:
    text = render_markdown(RESULTS)
    assert "SYNTHETIC" in text
    for label in ("BM25", "dense", "dense + BM25", "hybrid + rerank"):
        assert label in text
    assert "0.750" in text and "50.0%" in text


def test_update_readme_replaces_only_the_marked_block() -> None:
    readme = f"# Title\nintro\n{README_START}\nold numbers\n{README_END}\noutro\n"
    updated = update_readme(readme, RESULTS, "reports/results.md")
    assert "old numbers" not in updated
    assert updated.startswith("# Title\nintro\n") and updated.endswith(f"{README_END}\noutro\n")
    assert "reports/results.md" in updated
    assert update_readme(updated, RESULTS, "reports/results.md") == updated  # idempotent


def test_update_readme_handles_several_named_blocks() -> None:
    from tkrag.eval.report import markers

    a_start, a_end = markers("METRICS")
    b_start, b_end = markers("CI-METRICS")
    readme = f"{a_start}\nold A\n{a_end}\nmiddle\n{b_start}\nold B\n{b_end}\n"
    once = update_readme(readme, RESULTS, "reports/results_full.md", "METRICS")
    assert "old A" not in once and "old B" in once and "middle" in once
    twice = update_readme(once, RESULTS, "reports/results.md", "CI-METRICS")
    assert "old B" not in twice and "reports/results.md" in twice
