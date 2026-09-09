"""results.json + results.md. Every number in the README comes from these files."""

import json
from pathlib import Path
from typing import Any

from tkrag.retrieval.types import MODES

MODE_LABEL = {
    "fts": "Postgres FTS (ts_rank_cd)",
    "bm25": "BM25 (Postgres lexemes)",
    "dense": "dense (e5)",
    "hybrid": "hybrid (dense + BM25, RRF)",
    "hybrid_rerank": "hybrid + rerank",
}
TYPE_LABEL = {
    "factual": "factual",
    "paraphrase": "paraphrase",
    "multi_article": "multi_article",
}


def _pct(value: float) -> str:
    return f"{value:.0%}" if value in (0.0, 1.0) else f"{value:.1%}"


def _num(value: float) -> str:
    return f"{value:.3f}"


def _retrieval_tables(retrieval: dict[str, Any]) -> list[str]:
    modes = [m for m in MODES if m in retrieval]
    lines = [
        "| Mode | Recall@1 | Recall@5 | Recall@10 | MRR | latency p50, ms | latency p95, ms |",
        "|---|---|---|---|---|---|---|",
    ]
    for mode in modes:
        r = retrieval[mode]
        o = r["overall"]
        lines.append(
            f"| {MODE_LABEL[mode]} | {_pct(o['recall@1'])} | {_pct(o['recall@5'])} | "
            f"{_pct(o['recall@10'])} | {_num(o['mrr'])} | {r['latency_ms']['p50']:.1f} | "
            f"{r['latency_ms']['p95']:.1f} |"
        )
    types = list(retrieval[modes[0]]["by_type"]) if modes else []
    for metric, title in (("recall@1", "Recall@1"), ("recall@5", "Recall@5"), ("mrr", "MRR")):
        lines += ["", f"**{title} by question type**", ""]
        lines.append("| Type | n | " + " | ".join(MODE_LABEL[m] for m in modes) + " |")
        lines.append("|---|---|" + "---|" * len(modes))
        for kind in types:
            n = int(retrieval[modes[0]]["by_type"][kind]["n"])
            cells = []
            for mode in modes:
                value = retrieval[mode]["by_type"][kind][metric]
                cells.append(_num(value) if metric == "mrr" else _pct(value))
            lines.append(f"| {TYPE_LABEL.get(kind, kind)} | {n} | " + " | ".join(cells) + " |")
    return lines


def _misses_section(
    retrieval: dict[str, Any], mode: str = "hybrid_rerank", k: int = 5
) -> list[str]:
    """Questions where the main pipeline has no expected article in its top k."""
    if mode not in retrieval or "questions" not in retrieval[mode]:
        return []
    misses = [q for q in retrieval[mode]["questions"] if q["recall@5"] < 1.0]
    lines = [
        f"Questions not fully found in the top {k} of `{mode}` (nothing found, or for "
        f"`multi_article` at least one required article missing): "
        f"{len(misses)} of {len(retrieval[mode]['questions'])}.",
    ]
    if misses:
        lines += [
            "",
            "| id | type | question | expected | found (top 5) |",
            "|---|---|---|---|---|",
        ]
        for q in misses:
            question = q["question"].replace("|", "/")
            lines.append(
                f"| {q['id']} | {q['type']} | {question[:90]} | {', '.join(q['expected'])} | "
                f"{', '.join(q['top'])} |"
            )
    return lines


def _first_table(lines: list[str]) -> list[str]:
    """The overall table only (up to the first blank line), for the README."""
    return lines[: lines.index("")] if "" in lines else lines


def _answer_section(answers: dict[str, Any]) -> list[str]:
    counts = answers["status_counts"]
    lines = [
        f"LLM: `{answers['llm_model']}`; refusal threshold (reranker score): "
        f"{answers['refusal_threshold']}",
        "",
        "| Metric | Value | n |",
        "|---|---|---|",
        f"| Correct refusals on `no_answer` | {_pct(answers['correct_refusal_rate'])} | "
        f"{answers['n_no_answer']} |",
        f"| False refusals on answerable questions | {_pct(answers['false_refusal_rate'])} | "
        f"{answers['n_answerable']} |",
        f"| `answer_facts` recall (answerable, refusals count as 0; prepositions ignored) | "
        f"{_pct(answers['facts_recall'])} | {answers['n_answerable']} |",
        f"| `answer_facts` recall, strict (also demands prepositions and conjunctions) | "
        f"{_pct(answers.get('facts_recall_strict', answers['facts_recall']))} | "
        f"{answers['n_answerable']} |",
        f"| Grounded answers: numbers, articles and quotes are in the cited text "
        f"(share of the answers given, not of all questions) | "
        f"{_pct(answers['verified_rate'])} | {answers['n_answered']} |",
        "",
        "Outcome statuses: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())),
        "",
        *_answer_caveats(),
    ]
    return lines


def _answer_caveats() -> list[str]:
    """What the answer metrics do NOT say. Kept next to the numbers on purpose."""
    return [
        "**How to read these numbers.**",
        "",
        "- *Grounded answers* is a syntactic check: the numbers with units, the article numbers "
        "and the quotes of the answer occur in the cited fragments. It is not the share of legally "
        "correct answers; a grounded answer can still draw a wrong conclusion from the right "
        "article. It also counts only the answers that were given.",
        "- *`answer_facts` recall* is lexical overlap with the expected facts (words and numbers "
        "after a crude stemming). It misses a correct answer in other words and can accept a "
        "wrong answer that contains the same words.",
        "- *Correct refusals* depend on a threshold that was chosen on this same golden set, so "
        "they are optimistic for new questions. Only a set that was not used to choose any "
        "parameter or threshold gives an honest estimate (see `eval --frozen` in the README).",
    ]


def _calibration_section(calibration: dict[str, Any]) -> list[str]:
    lines = [
        "Refuse when the best reranker score is below the threshold. "
        f"The threshold with the best (correct refusals - false refusals) is "
        f"**{calibration['chosen']}**. It is a suggestion fitted on this same golden set, "
        "not an independent estimate.",
        "",
        "| Threshold | Correct refusals | False refusals |",
        "|---|---|---|",
    ]
    for point in calibration["curve"]:
        lines.append(
            f"| {point['threshold']} | {_pct(point['correct_refusal'])} | "
            f"{_pct(point['false_refusal'])} |"
        )
    return lines


def _gate_section(gate: dict[str, Any]) -> list[str]:
    lines = ["| Check | Value | Threshold | Result |", "|---|---|---|---|"]
    for check in gate["checks"]:
        verdict = "pass" if check["ok"] else "FAIL"
        lines.append(
            f"| {check['name']} | {_pct(check['value'])} | >= {_pct(check['threshold'])} | "
            f"{verdict} |"
        )
    return lines


def render_markdown(results: dict[str, Any]) -> str:
    ds = results["dataset"]
    out = ["# Evaluation results", ""]
    out.append(
        f"Dataset: **{ds['name']}** — {ds['n_articles']} articles, {ds['n_chunks']} chunks, "
        f"{ds['n_questions']} questions (`{ds['golden']}`)."
    )
    if results.get("frozen"):
        out += [
            "",
            "> **Frozen test set.** The RRF parameters, the refusal threshold and the prompts "
            "were fixed before this run and not tuned on these questions. No threshold "
            "calibration is shown on purpose.",
        ]
    if ds["name"] == "fixture":
        out += [
            "",
            "> The `fixture` corpus is SYNTHETIC: a few dozen hand-written approximations of "
            "articles. These numbers check that the pipeline works; they say nothing about "
            "quality on the real Labor Code. Small n: one question moves a percentage by "
            "several points.",
        ]
    if "draft" in ds["golden"]:
        out += [
            "",
            "> The golden set is an LLM-written DRAFT that no human has reviewed. Expected "
            "articles are the articles the questions were generated from, so a question that "
            "another article answers equally well counts as a miss. Treat the numbers as "
            "exploratory.",
        ]
    cfg = results["config"]
    out += [
        "",
        f"Embeddings `{cfg['embedding_model']}`, reranker `{cfg['reranker_model']}`, "
        f"RRF k={cfg['rrf_k']} (dense weight {cfg.get('rrf_dense_weight', 1):g}), "
        f"rerank top-{cfg['candidates_top_n']} -> top-{cfg['final_top_k']}. "
        "Recall@k: a question counts as found when ANY expected article is among the top-k "
        "distinct articles (alternatives); for `multi_article` it is the SHARE of the required "
        "articles found, and MRR uses the rank where the last required article appears.",
        "",
        "## Retrieval",
        "",
        *_retrieval_tables(results["retrieval"]),
    ]
    misses = _misses_section(results["retrieval"])
    if misses:
        out += ["", "### Misses", "", *misses]
    if "answers" in results:
        out += ["", "## Answers", "", *_answer_section(results["answers"])]
    if "calibration" in results:
        out += ["", "## Refusal threshold", "", *_calibration_section(results["calibration"])]
    if "gate" in results:
        out += ["", "## CI gate", "", *_gate_section(results["gate"])]
    return "\n".join(out) + "\n"


def write_reports(results: dict[str, Any], directory: Path, stem: str) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    json_path = directory / f"{stem}.json"
    md_path = directory / f"{stem}.md"
    json_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    md_path.write_text(render_markdown(results), encoding="utf-8")
    return md_path, json_path


README_START = "<!-- METRICS:START -->"  # default block: the real corpus
README_END = "<!-- METRICS:END -->"


def markers(name: str) -> tuple[str, str]:
    return f"<!-- {name}:START -->", f"<!-- {name}:END -->"


def render_readme_block(results: dict[str, Any], results_path: str) -> str:
    """The metrics block of the README: same tables as results.md, plus a link to it."""
    ds = results["dataset"]
    lines = [
        f"Dataset `{ds['name']}`: {ds['n_articles']} articles, {ds['n_chunks']} chunks, "
        f"{ds['n_questions']} questions (`{ds['golden']}`). "
        f"Full report: [{results_path}]({results_path}).",
        "",
        *_first_table(_retrieval_tables(results["retrieval"])),
    ]
    if "answers" in results:
        lines += ["", *_answer_section(results["answers"])]
    return "\n".join(lines)


def update_readme(
    readme: str, results: dict[str, Any], results_path: str, name: str = "METRICS"
) -> str:
    """Replace the text between `<!-- NAME:START -->` and `<!-- NAME:END -->`."""
    start_marker, end_marker = markers(name)
    start, end = readme.index(start_marker), readme.index(end_marker)
    block = render_readme_block(results, results_path)
    return f"{readme[: start + len(start_marker)]}\n{block}\n{readme[end:]}"
