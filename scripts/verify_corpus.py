"""Spot-check data/raw/tk_rf.txt against an independent source (zakonrf.info, one page per article).

For a sample of articles it fetches the page, takes the article body and compares it line by line
with what the parser produced from our corpus file. The corpus came from KonsultantPlus, so this
checks both the download and the parser against a second site.

Usage: uv run python scripts/verify_corpus.py [--sample 15] [--seed 1] [--delay 1.5]

It does NOT verify part numbering: the Labor Code does not print part numbers, so the numbers
come from our paragraph-order heuristic. Compare a few articles by eye (see PROGRESS.md).
"""

import argparse
import random
import re
import sys
import time
from html import unescape
from pathlib import Path

import httpx

from tkrag.ingest.parser import Article, live_articles, parse_articles, read_text
from tkrag.textnorm import normalize

ROOT = Path(__file__).resolve().parents[1]
URL = "https://www.zakonrf.info/tk/{number}/"
ALWAYS = ["115", "81", "80", "255", "140", "178", "84.1", "341.1-1", "22", "57"]
_TAG_RE = re.compile(r"<[^>]+>")
_NOISE_RE = re.compile(r"^\((?:в ред|введен|п\. |ч\. |абзац|часть|пункт)", re.IGNORECASE)


def site_lines(html: str) -> list[str]:
    """Paragraph lines between the article heading and the revision-history block."""
    cleaned = re.sub(r"(?s)<script.*?</script>|<style.*?</style>", "", html)
    lines = [" ".join(unescape(ln).split()) for ln in _TAG_RE.sub("\n", cleaned).splitlines()]
    lines = [ln for ln in lines if ln]
    end = next(
        i for i, ln in enumerate(lines) if ln.startswith(("История редакций", "Обзор редакций"))
    )
    # The body starts after the last "Статья N ТК РФ. Title ..." heading above the history block.
    start = max(i for i, ln in enumerate(lines[:end]) if ln.startswith("Статья ") and "ТК РФ" in ln)
    return [ln for ln in lines[start + 1 : end] if not _NOISE_RE.match(ln)]


def our_lines(article: Article) -> list[str]:
    return [ln for part in article.parts for ln in part.text.split("\n")]


def compare(ours: list[str], theirs: list[str]) -> tuple[int, list[str]]:
    """Number of identical lines (after normalisation) and descriptions of the differences."""
    a, b = [normalize(x) for x in ours], [normalize(x) for x in theirs]
    problems = []
    if len(a) != len(b):
        problems.append(f"line count: ours {len(a)}, site {len(b)}")
    same = 0
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x == y:
            same += 1
        else:
            problems.append(f"line {i + 1} differs: ours {x[:60]!r} / site {y[:60]!r}")
    return same, problems


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=15)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--delay", type=float, default=1.5)
    args = parser.parse_args()

    articles = {
        a.number: a for a in live_articles(parse_articles(read_text(ROOT / "data/raw/tk_rf.txt")))
    }
    rng = random.Random(args.seed)
    others = [n for n in articles if n not in ALWAYS]
    numbers = [n for n in ALWAYS if n in articles]
    numbers += rng.sample(others, max(args.sample - len(numbers), 0))

    failures = 0
    with httpx.Client(headers={"User-Agent": "tkrag-corpus-check/1.0"}, timeout=30) as client:
        for number in numbers:
            time.sleep(args.delay)
            response = client.get(URL.format(number=number), follow_redirects=True)
            if response.status_code != 200:
                print(f"{number}: site answered {response.status_code}, skipped")
                continue
            try:
                theirs = site_lines(response.text)
            except StopIteration:
                print(f"{number}: could not find the article body on the page, skipped")
                continue
            same, problems = compare(our_lines(articles[number]), theirs)
            status = "ok" if not problems else "DIFF"
            failures += bool(problems)
            print(f"{number}: {status} ({same} identical lines)")
            for problem in problems[:3]:
                print(f"    {problem}")
    print(f"{len(numbers)} articles checked, {failures} with differences")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
