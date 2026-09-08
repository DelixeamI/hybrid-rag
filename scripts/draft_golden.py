"""Propose draft golden questions with the configured LLM -> data/golden.draft.jsonl.

Usage: uv run python scripts/draft_golden.py [--dataset fixture|full] [--clusters 11] [--seed 7]

Samples `clusters` chapters (always including the vacation and termination chapters), takes
3 consecutive articles from each: 2 questions per article, multi-article questions over
neighbours, and off-topic questions. Questions whose answer_facts are not made of words found in
the article are dropped automatically.

The output is a DRAFT. Review every line by hand (questions, expected_articles, answer_facts),
then save the result as data/golden.jsonl. `python -m tkrag.cli golden-check` validates it.
"""

import argparse
import random
from pathlib import Path

from tkrag.config import get_settings
from tkrag.datasets import get_dataset
from tkrag.eval.draft import draft_golden
from tkrag.ingest.parser import Article, live_articles, parse_articles, read_text
from tkrag.llm import LLMClient, OpenAICompatibleLLM
from tkrag.llm.openai_compat import LLMError

# The free Gemini tier allows ~20 requests per day per model, so drafting rotates over models.
# Drafts are reviewed by a human, so it does not matter which model wrote a question.
DEFAULT_MODELS = (
    "gemini-3.5-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash-lite",
    "gemini-flash-lite-latest",
)
ALWAYS_CHAPTERS = ("Глава 19.", "Глава 13.")  # vacations and termination: the task's examples
CLUSTER_SIZE = 3


class RotatingLLM:
    """Uses the first client until its daily quota is gone, then moves to the next one."""

    def __init__(self, clients: list[LLMClient]) -> None:
        self._clients = clients
        self.model = "rotating:" + ",".join(c.model for c in clients)

    def complete(self, system: str, user: str) -> str:
        while True:
            try:
                return self._clients[0].complete(system, user)
            except LLMError as error:
                if "429" not in str(error) or len(self._clients) == 1:
                    raise
                print(f"quota gone for {self._clients[0].model}, switching", flush=True)
                self._clients.pop(0)


def pick_clusters(articles: list[Article], count: int, seed: int) -> list[list[Article]]:
    """`count` runs of consecutive live articles from distinct chapters, in document order."""
    rng = random.Random(seed)
    by_chapter: dict[str, list[Article]] = {}
    for article in articles:
        by_chapter.setdefault(article.chapter or "", []).append(article)
    usable = {c: arts for c, arts in by_chapter.items() if len(arts) >= CLUSTER_SIZE}
    forced = [c for c in usable if c.startswith(ALWAYS_CHAPTERS)]
    others = [c for c in usable if c not in forced]
    chosen = forced + rng.sample(others, max(count - len(forced), 0))
    clusters = []
    for chapter in chosen:
        start = rng.randrange(len(usable[chapter]) - CLUSTER_SIZE + 1)
        clusters.append(usable[chapter][start : start + CLUSTER_SIZE])
    order = {a.number: i for i, a in enumerate(articles)}
    return sorted(clusters, key=lambda c: order[c[0].number])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["fixture", "full"], default="full")
    parser.add_argument("--clusters", type=int, default=11)
    parser.add_argument("--pairs", type=int, default=12, help="multi-article questions")
    parser.add_argument("--no-answer", type=int, default=16)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--models", nargs="*", default=list(DEFAULT_MODELS))
    parser.add_argument("--out", type=Path, default=Path("data/golden.draft.jsonl"))
    args = parser.parse_args()

    settings = get_settings()
    dataset = get_dataset(args.dataset, settings)
    articles = live_articles(parse_articles(read_text(dataset.corpus_path)))
    clusters = pick_clusters(articles, args.clusters, args.seed)
    picked = [a for cluster in clusters for a in cluster]
    neighbours = [(c[i], c[i + 1]) for c in clusters for i in range(CLUSTER_SIZE - 1)]
    pairs = random.Random(args.seed).sample(neighbours, min(args.pairs, len(neighbours)))

    result = draft_golden(
        picked,
        RotatingLLM(
            [
                OpenAICompatibleLLM(
                    settings.llm_base_url, m, settings.llm_api_key, min_interval=2.0
                )
                for m in args.models
            ]
        ),
        no_answer=args.no_answer,
        pairs=sorted(pairs, key=lambda p: p[0].number),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(i.model_dump_json() + "\n" for i in result.items), encoding="utf-8")
    for line in result.dropped:
        print(f"dropped: {line}")
    print(
        f"wrote {len(result.items)} DRAFT questions to {args.out} "
        f"({len(result.dropped)} dropped). Review them by hand before use."
    )


if __name__ == "__main__":
    main()
