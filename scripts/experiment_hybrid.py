"""Does a better sparse channel or a different fusion beat dense alone? (dev/test protocol)

Usage: uv run python scripts/experiment_hybrid.py [--dataset full] [--seed 13]
Writes reports/experiment_hybrid.md and .json. Needs Postgres with the corpus ingested.
Reranker is not involved: this measures the first stage (what the reranker gets to choose from).
"""

import argparse
import json
from pathlib import Path
from typing import Any

from tkrag.components import make_embedder
from tkrag.config import get_settings
from tkrag.datasets import get_dataset
from tkrag.db import connect
from tkrag.eval.experiments import (
    BASELINE,
    Config,
    Prepared,
    fused_ranking,
    grid,
    pick_winner,
    score_ranking,
    sparse_variant,
    split_dev_test,
)
from tkrag.eval.golden import load_golden
from tkrag.retrieval.bm25 import load_bm25, query_lexemes
from tkrag.retrieval.search import search_dense

SPARSE = ["fts_cd", "fts_cd_len", "fts_cd_len_unique", "fts_rank", "title_only", "bm25"]
POOL = 50
ROOT = Path(__file__).resolve().parents[1]


def pct(value: float) -> str:
    return f"{value:.1%}"


def row(label: str, s: dict[str, float]) -> str:
    return (
        f"| {label} | {pct(s['recall@1'])} | {pct(s['recall@5'])} | {pct(s['recall@20'])} | "
        f"{s['mrr']:.3f} |"
    )


HEADER = ["| Configuration | Recall@1 | Recall@5 | Recall@20 | MRR |", "|---|---|---|---|---|"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["fixture", "full"], default="full")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    settings = get_settings()
    dataset = get_dataset(args.dataset, settings)
    conn = connect(settings.database_url)
    embedder = make_embedder(settings)
    items = [i for i in load_golden(dataset.golden_path) if i.type != "no_answer"]
    article_of = {
        int(r[0]): str(r[1])
        for r in conn.execute(
            "SELECT c.id, c.article_number FROM chunks c JOIN documents d ON d.id = c.document_id "
            "WHERE d.name = %s",
            (dataset.document,),
        )
    }
    bm25 = load_bm25(conn, dataset.document)

    prepared: dict[str, Prepared] = {}
    for item in items:
        dense = [
            h.chunk_id
            for h in search_dense(conn, dataset.document, embedder.embed_query(item.question), POOL)
        ]
        rankings = {"dense": dense}
        for name in SPARSE:
            if name == "bm25":
                rankings[name] = [
                    cid for cid, _ in bm25.search(query_lexemes(conn, item.question), POOL)
                ]
            else:
                rankings[name] = sparse_variant(conn, dataset.document, item.question, name, POOL)
        prepared[item.id] = Prepared(item, rankings)

    dev_items, test_items = split_dev_test(items, args.seed)
    dev = [prepared[i.id] for i in dev_items]
    test = [prepared[i.id] for i in test_items]

    single = {name: (lambda p, n=name: p.rankings[n]) for name in ["dense", *SPARSE]}
    dev_single = {n: score_ranking(dev, f, article_of) for n, f in single.items()}
    configs = grid(SPARSE)
    dev_fused = {
        c: score_ranking(dev, lambda p, c=c: fused_ranking(p, c), article_of) for c in configs
    }
    winner = pick_winner(dev_fused)

    def on_test(config: Config) -> dict[str, float]:
        return score_ranking(test, lambda p: fused_ranking(p, config), article_of)

    test_rows = {
        "dense alone": score_ranking(test, single["dense"], article_of),
        "BM25 alone (true BM25 on Postgres lexemes)": score_ranking(
            test, single["bm25"], article_of
        ),
        f"today's hybrid ({BASELINE.label})": on_test(BASELINE),
        f"dev winner ({winner.label})": on_test(winner),
    }

    lines = [
        "# Hybrid experiment (first stage only, no reranker)",
        "",
        f"Dataset `{args.dataset}`, {len(items)} answerable questions, split {len(dev)} dev / "
        f"{len(test)} test (stratified by type, seed {args.seed}). Configurations were compared "
        "on DEV; the winner and the baselines are then reported on TEST. Noise is large: one "
        "question moves Recall@1 by about "
        f"{100 / len(test):.1f} points.",
        "",
        "## DEV: single channels",
        "",
        *HEADER,
        *[row(n, s) for n, s in dev_single.items()],
        "",
        "## DEV: dense + one sparse channel fused with weighted RRF",
        "",
        *HEADER,
        *[row(c.label, s) for c, s in sorted(dev_fused.items(), key=lambda x: -x[1]["mrr"])[:15]],
        "",
        f"(top 15 of {len(configs)} by MRR; winner by the rule `best MRR, then Recall@5, then "
        f"closest to the shipped setting`: **{winner.label}**)",
        "",
        "## TEST",
        "",
        *HEADER,
        *[row(label, s) for label, s in test_rows.items()],
        "",
    ]
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    (out / "experiment_hybrid.md").write_text("\n".join(lines), encoding="utf-8")
    payload: dict[str, Any] = {
        "seed": args.seed,
        "n_dev": len(dev),
        "n_test": len(test),
        "winner": winner.label,
        "dev_single": dev_single,
        "dev_fused": {c.label: s for c, s in dev_fused.items()},
        "test": test_rows,
    }
    (out / "experiment_hybrid.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
