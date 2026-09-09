"""Command line: `python -m tkrag.cli <command>`; the Makefile wraps these."""

import argparse
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from tkrag.components import (
    cassettes_dir,
    make_embedder,
    make_llm,
    make_recording_llm,
    make_reranker,
    make_retriever,
)
from tkrag.config import Settings, get_settings
from tkrag.datasets import ROOT, Dataset, get_dataset
from tkrag.db import connect
from tkrag.eval.answer_eval import best_scores, calibrate_threshold, load_thresholds
from tkrag.eval.golden import load_golden, overlap_errors, validate_items, validate_quotas
from tkrag.eval.report import write_reports
from tkrag.eval.run import prepare, run_full, run_retrieval
from tkrag.generation.answerer import Answerer
from tkrag.ingest.loader import ingest
from tkrag.ingest.parser import live_articles, parse_articles, read_text
from tkrag.llm import LLMClient, RecordedLLM
from tkrag.retrieval.types import MODES, Mode


def _add_dataset(parser: argparse.ArgumentParser, default: str = "fixture") -> None:
    parser.add_argument("--dataset", choices=["fixture", "full"], default=default)
    parser.add_argument("--golden", type=Path, help="use another question set (JSONL)")
    parser.add_argument("--stem", help="report file name: reports/<stem>.md and .json")


def _dataset(args: argparse.Namespace, settings: Settings) -> Dataset:
    return get_dataset(args.dataset, settings, args.golden, args.stem)


def cmd_ingest(args: argparse.Namespace) -> int:
    settings = get_settings()
    dataset = _dataset(args, settings)
    conn = connect(settings.database_url)
    report = ingest(conn, dataset.corpus_path, dataset.document, make_embedder(settings))
    print(report.summary())
    return 0


def cmd_eval_retrieval(args: argparse.Namespace) -> int:
    settings = get_settings()
    dataset = _dataset(args, settings)
    modes: tuple[Mode, ...] = tuple(args.modes) if args.modes else MODES
    conn = connect(settings.database_url)
    embedder = make_embedder(settings)
    reranker = make_reranker(settings) if "hybrid_rerank" in modes else None
    results = run_retrieval(conn, dataset, settings, embedder, reranker, modes)
    md, js = write_reports(results, settings.reports_dir, dataset.report_stem)
    print(f"wrote {md} and {js}")
    print(md.read_text(encoding="utf-8"))
    return 0


def _print_gate(results: dict[str, Any]) -> int:
    gate = results["gate"]
    for check in gate["checks"]:
        verdict = "ok" if check["ok"] else "FAIL"
        print(f"gate {check['name']}: {check['value']:.3f} >= {check['threshold']} ... {verdict}")
    return 0 if gate["ok"] else 1


def _run_full(args: argparse.Namespace, llm: LLMClient, gate: bool) -> int:
    settings = get_settings()
    dataset = _dataset(args, settings)
    thresholds = load_thresholds(ROOT / settings.thresholds_path) if gate else None
    conn = connect(settings.database_url)
    results = run_full(
        conn,
        dataset,
        settings,
        make_embedder(settings),
        make_reranker(settings),
        llm,
        thresholds,
        frozen=getattr(args, "frozen", False),
    )
    md, js = write_reports(results, settings.reports_dir, dataset.report_stem)
    print(f"wrote {md} and {js}")
    return _print_gate(results) if gate else 0


def cmd_eval(args: argparse.Namespace) -> int:
    settings = get_settings()
    recorded = RecordedLLM(cassettes_dir(settings, args.dataset), settings.llm_model)
    llm: LLMClient = make_llm(settings, args.dataset) if args.llm == "configured" else recorded
    code = _run_full(args, llm, gate=args.dataset == "fixture")
    if args.llm == "recorded":
        stale = recorded.unused_cassettes()
        print(f"{len(stale)} cassette(s) of {recorded.model!r} were not used by this run")
        if args.prune_cassettes and code == 0:
            for path in stale:
                path.unlink()
            print(f"deleted {len(stale)} unused cassette(s)")
    return code


def cmd_record(args: argparse.Namespace) -> int:
    return _run_full(
        args, make_recording_llm(get_settings(), args.dataset, args.resume), gate=False
    )


def cmd_ask(args: argparse.Namespace) -> int:
    settings = get_settings()
    dataset = _dataset(args, settings)
    conn = connect(settings.database_url)
    retriever = make_retriever(
        conn, settings, dataset.document, make_embedder(settings), make_reranker(settings)
    )
    answerer = Answerer(retriever, make_llm(settings, args.dataset), settings.refusal_threshold)
    print(answerer.ask(args.question).model_dump_json(indent=2))
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    settings = get_settings()
    dataset = _dataset(args, settings)
    conn = connect(settings.database_url)
    embedder = make_embedder(settings)
    items, _ = prepare(conn, dataset, embedder)
    retriever = make_retriever(conn, settings, dataset.document, embedder, make_reranker(settings))
    scored = best_scores(retriever, items, settings.context_chunks)
    for (_, score), item in sorted(zip(scored, items, strict=True), key=lambda x: x[0][1]):
        print(f"{score:.4f}  {item.type:<13} {item.id}  {item.question[:60]}")
    calibration = calibrate_threshold(scored)
    print(f"\nsuggested refusal_threshold = {calibration.threshold}")
    return 0


def cmd_golden_check(args: argparse.Namespace) -> int:
    settings = get_settings()
    dataset = _dataset(args, settings)
    items = load_golden(dataset.golden_path)
    known = {a.number for a in live_articles(parse_articles(read_text(dataset.corpus_path)))}
    issues = validate_items(items, known)
    errors = list(issues.errors)
    if args.dataset == "full":  # the fixture golden is intentionally small
        errors += validate_quotas(items).errors
    counts = Counter(i.type for i in items)
    print(f"{len(items)} questions: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    if args.against:
        other = load_golden(args.against)
        errors += overlap_errors(items, other)
        shared = {a for i in items for a in i.expected_articles} & {
            a for i in other for a in i.expected_articles
        }
        print(f"{len(shared)} expected article(s) also occur in {args.against.name}")
    for error in errors:
        print(f"ERROR: {error}")
    return 1 if errors else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tkrag")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="parse, chunk, embed and store the corpus")
    _add_dataset(p)
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("eval-retrieval", help="Recall@k / MRR / latency per retrieval mode")
    _add_dataset(p)
    p.add_argument("--modes", nargs="*", choices=list(MODES))
    p.set_defaults(func=cmd_eval_retrieval)

    p = sub.add_parser("eval", help="retrieval + answers + CI gate, RecordedLLM by default")
    _add_dataset(p)
    p.add_argument("--llm", choices=["recorded", "configured"], default="recorded")
    p.add_argument(
        "--frozen",
        action="store_true",
        help="treat the question set as a test set: show no threshold calibration, mark the report",
    )
    p.add_argument(
        "--prune-cassettes",
        action="store_true",
        help="after a successful run delete cassettes of this model that nothing replayed",
    )
    p.set_defaults(func=cmd_eval)

    p = sub.add_parser("record", help="run the eval on the real LLM and write cassettes")
    _add_dataset(p)
    p.add_argument(
        "--frozen", action="store_true", help="test set: do not compute a threshold calibration"
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="reuse cassettes that exist, ask the model for the rest",
    )
    p.set_defaults(func=cmd_record)

    p = sub.add_parser("calibrate", help="best reranker score per question + suggested threshold")
    _add_dataset(p)
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("golden-check", help="validate the golden set (structure and quotas)")
    _add_dataset(p)
    p.add_argument(
        "--against", type=Path, help="another golden set this one must not repeat questions of"
    )
    p.set_defaults(func=cmd_golden_check)

    p = sub.add_parser("ask", help="ask one question")
    _add_dataset(p)
    p.add_argument("question")
    p.set_defaults(func=cmd_ask)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
