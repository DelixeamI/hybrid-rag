"""One-time download of the Labor Code text from KonsultantPlus into data/raw/tk_rf.txt.

The official text of a law is not protected by copyright. The site allows /document/ in
robots.txt. We still keep it polite: one request at a time, a pause between requests, a cache so
an interrupted run resumes without asking the site again, and a stop on repeated errors.

Usage: uv run python scripts/fetch_corpus.py [--delay 1.2] [--limit N]
Writes data/raw/tk_rf.txt and data/raw/SOURCE.md.
"""

import argparse
import re
import sys
import time
from datetime import date
from html import unescape
from html.parser import HTMLParser
from pathlib import Path

import httpx

BASE = "https://www.consultant.ru"
DOC = "/document/cons_doc_LAW_34683/"
USER_AGENT = "tkrag-corpus-fetch/1.0 (one-time personal download for a study project)"
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "raw" / ".cache"

_LINK_RE = re.compile(
    r'<a[^>]+href="(/document/cons_doc_LAW_34683/[0-9a-f]{40}/)[^"]*"[^>]*>(.*?)</a>', re.S
)
_TAG_RE = re.compile(r"<[^>]+>")
_HEADING_RE = re.compile(r"^(Часть|Раздел|Подраздел|Глава|Статья)\s", re.IGNORECASE)
_SKIP_CLASSES = ("document__insert", "document__edit", "document__style")


class ContentParser(HTMLParser):
    """Collects <p> paragraphs of the legal text, skipping editorial blocks and the title."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.paragraphs: list[str] = []
        self._in_content = False
        self._depth = 0  # div depth inside the content block
        self._skip_depth: int | None = None
        self._buffer: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        classes = dict(attrs).get("class") or ""
        if tag == "div":
            if "document-page__content" in classes:
                self._in_content, self._depth = True, 0
            elif self._in_content:
                self._depth += 1
                if self._skip_depth is None and any(c in classes for c in _SKIP_CLASSES):
                    self._skip_depth = self._depth
        elif tag == "p" and self._in_content and self._skip_depth is None:
            self._buffer = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "p" and self._buffer is not None:
            text = " ".join("".join(self._buffer).split())
            if text:
                self.paragraphs.append(text)
            self._buffer = None
        elif tag == "div" and self._in_content:
            if self._skip_depth == self._depth:
                self._skip_depth = None
            self._depth -= 1
            if self._depth < 0:
                self._in_content = False

    def handle_data(self, data: str) -> None:
        if self._buffer is not None:
            self._buffer.append(data)


def fetch(client: httpx.Client, path: str, delay: float) -> str:
    cached = CACHE / (path.strip("/").replace("/", "_") + ".html")
    if cached.exists():
        return cached.read_text(encoding="utf-8")
    failures = 0
    while True:
        time.sleep(delay)
        response = client.get(BASE + path)
        if response.status_code == 200:
            CACHE.mkdir(parents=True, exist_ok=True)
            cached.write_text(response.text, encoding="utf-8")
            return response.text
        failures += 1
        print(f"HTTP {response.status_code} for {path} (attempt {failures})", file=sys.stderr)
        if failures >= 3 or response.status_code in (403, 429):
            sys.exit(f"stopping: the site answered {response.status_code}; try again later")
        time.sleep(10 * failures)


def toc_entries(toc_html: str) -> list[tuple[str, str]]:
    """(path, heading text) in document order, one entry per distinct heading."""
    entries: list[tuple[str, str]] = []
    seen: set[str] = set()
    for path, label in _LINK_RE.findall(toc_html):
        text = " ".join(unescape(_TAG_RE.sub("", label)).split())
        if _HEADING_RE.match(text) and path not in seen:
            seen.add(path)
            entries.append((path, text))
    return entries


def edition_line(toc_html: str) -> str:
    match = re.search(r"ред\.\s*от\s*(\d{2}\.\d{2}\.\d{4})", unescape(toc_html), re.IGNORECASE)
    return match.group(1) if match else "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delay", type=float, default=1.2, help="seconds between requests")
    parser.add_argument("--limit", type=int, default=0, help="only the first N articles")
    args = parser.parse_args()

    out_dir = ROOT / "data" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    with httpx.Client(
        headers={"User-Agent": USER_AGENT}, timeout=30, follow_redirects=True
    ) as client:
        toc_html = fetch(client, DOC, 0.0)
        entries = toc_entries(toc_html)
        edition = edition_line(toc_html)
        lines: list[str] = []
        fetched = 0
        for path, heading in entries:
            lines.append(heading)
            if not heading.startswith("Статья"):
                continue
            if re.search(r"Утратил[аи]?\s+силу", heading, re.IGNORECASE):
                continue
            if args.limit and fetched >= args.limit:
                continue
            parser_ = ContentParser()
            parser_.feed(fetch(client, path, args.delay))
            lines.extend(parser_.paragraphs)
            fetched += 1
            if fetched % 50 == 0:
                print(f"{fetched} articles fetched", flush=True)

    corpus = out_dir / "tk_rf.txt"
    corpus.write_text("\n".join(lines) + "\n", encoding="utf-8")
    articles = sum(1 for ln in lines if ln.startswith("Статья"))
    print(f"wrote {corpus}: {len(lines)} lines, {articles} article headings, edition {edition}")
    (out_dir / "SOURCE.md").write_text(
        f"""# Source of the corpus

- Document: Трудовой кодекс Российской Федерации от 30.12.2001 N 197-ФЗ (ТК РФ).
- Edition: consolidated text, "последняя редакция", Ред. от {edition} (as shown by the site).
- Taken from: {BASE}{DOC} (KonsultantPlus, free public pages), one page per article.
- Downloaded: {date.today().isoformat()} by `scripts/fetch_corpus.py`.
- Processing: only the legal text of each article is kept. Editorial notes (changes history),
  "Готовые решения" inserts and comments are dropped. Repealed articles keep only their heading.
- The official text of a law is not subject to copyright. The consolidated edition is
  maintained by the publisher; for legal work check the official publication on pravo.gov.ru.
""",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
