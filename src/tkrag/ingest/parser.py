"""Plain text of the Labor Code -> Section > Chapter > Article > Parts, with regular expressions."""

import re
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import ClassVar

CORPUS_LABEL = "ТК РФ"

_SECTION_RE = re.compile(r"^(?:Раздел|Подраздел)\s+[IVXLC]+(?:\.\d+)?\.?\s+.+$", re.IGNORECASE)
# Numbers look like 81, 84.1 and, in the real code, 341.1-1 (article 341.1-1).
_NUMBER = r"\d+(?:[.-]\d+)*"
_CHAPTER_RE = re.compile(rf"^Глава\s+({_NUMBER})\.\s*(.+)$", re.IGNORECASE)
_ARTICLE_RE = re.compile(rf"^Статья\s+({_NUMBER})\.\s*(.*)$", re.IGNORECASE)
_LIST_ITEM_RE = re.compile(r"^(\d+|[а-яa-z])\)")
_PART_RE = re.compile(r"^(\d+(?:\.\d+)?)\.\s+(\S.*)$")  # explicit: "1. text", "1.1. text"
_REPEALED_RE = re.compile(r"^утратил[аи]?\s+силу", re.IGNORECASE)
_INLINE_NOTE_RE = re.compile(
    r"\s*\((?:в ред\.|введен[аоы]?\b|абзац[^)]*введен|см\. текст в предыдущей|"
    r"п\. \d+ в ред|часть [^)]*в ред|пункт [^)]*в ред)[^)]*\)",
    re.IGNORECASE,
)
_NOTE_LINE_RE = re.compile(
    r"^\(.*(?:в ред\.|введен|утратил|Федеральн|N \d+-ФЗ).*\)$", re.IGNORECASE
)


@dataclass(frozen=True)
class Part:
    number: str | None
    text: str
    # True when the number is the paragraph's order, not printed in the text. The Labor Code
    # numbers its parts only implicitly ("часть вторая статьи 115").
    implicit: bool = False


@dataclass(frozen=True)
class Article:
    number: str
    title: str
    section: str | None
    chapter: str | None
    parts: tuple[Part, ...]
    repealed: bool

    @property
    def heading(self) -> str:
        return f"Статья {self.number}. {self.title}"

    @property
    def breadcrumbs(self) -> str:
        crumbs = [CORPUS_LABEL]
        if self.chapter:
            crumbs.append(self.chapter)
        crumbs.append(self.heading)
        return " > ".join(crumbs)


def _part_key(number: str) -> tuple[int, int]:
    major, _, minor = number.partition(".")
    return int(major), int(minor or 0)


def _is_next_part(number: str, last: str | None) -> bool:
    """A numbered line starts a new part only if it continues the sequence (1, 2, 2.1, 3...)."""
    if last is None:
        return number == "1"
    new, old = _part_key(number), _part_key(last)
    return new > old and new[0] <= old[0] + 1


def read_text(path: Path) -> str:
    """Read a .txt or .html corpus file (utf-8, falls back to cp1251) as plain text."""
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp1251")
    return _html_to_text(text) if path.suffix.lower() in {".html", ".htm"} else text


class _TextExtractor(HTMLParser):
    _BLOCK_TAGS: ClassVar[set[str]] = {
        "p",
        "div",
        "br",
        "li",
        "tr",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.chunks: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            self._skip += 1
        elif tag in self._BLOCK_TAGS:
            self.chunks.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"}:
            self._skip -= 1
        elif tag in self._BLOCK_TAGS:
            self.chunks.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.chunks.append(data)


def _html_to_text(html: str) -> str:
    extractor = _TextExtractor()
    extractor.feed(html)
    return "".join(extractor.chunks).replace("\xa0", " ")


def _clean_line(line: str) -> str:
    line = " ".join(line.replace("\xa0", " ").split())
    if _NOTE_LINE_RE.match(line):
        return ""
    return _INLINE_NOTE_RE.sub("", line).strip()


def _build_article(
    number: str,
    title: str,
    section: str | None,
    chapter: str | None,
    body: list[str],
) -> Article:
    repealed = bool(_REPEALED_RE.match(title))
    parts: list[Part] = []
    number_of_current: str | None = None
    buffer: list[str] = []
    last: str | None = None

    def flush() -> None:
        if buffer:
            parts.append(Part(number_of_current, "\n".join(buffer)))
            buffer.clear()

    for line in body:
        match = _PART_RE.match(line)
        if match and _is_next_part(match.group(1), last):
            flush()
            number_of_current = last = match.group(1)
            buffer.append(match.group(2))
        else:
            buffer.append(line)
    flush()
    if len(parts) == 1 and parts[0].number is None:
        parts = _implicit_parts(parts[0].text.split("\n"))
    return Article(number, title, section, chapter, tuple(parts), repealed)


def _continues_part(previous: str, line: str) -> bool:
    """A line belongs to the previous part if it is a list item or follows a list lead-in."""
    return (
        previous.rstrip().endswith((":", ";", ","))
        or line[0].islower()
        or bool(_LIST_ITEM_RE.match(line))
        or line.startswith(("-", "–", "—"))
    )


def _implicit_parts(lines: list[str]) -> list[Part]:
    """Split an unnumbered article into its paragraphs (= parts); list items stay with them."""
    groups: list[list[str]] = []
    for line in lines:
        if groups and _continues_part(groups[-1][-1], line):
            groups[-1].append(line)
        else:
            groups.append([line])
    if len(groups) == 1:
        return [Part(None, "\n".join(groups[0]))]
    return [Part(str(i), "\n".join(g), implicit=True) for i, g in enumerate(groups, start=1)]


def parse_articles(text: str) -> list[Article]:
    """Parse the whole text. Repealed articles are returned with `repealed=True`."""
    articles: dict[str, Article] = {}
    section: str | None = None
    chapter: str | None = None
    current: tuple[str, str] | None = None
    body: list[str] = []

    def close() -> None:
        if current is None:
            return
        article = _build_article(current[0], current[1], section, chapter, body)
        known = articles.get(article.number)
        # A table of contents repeats headings with no text: keep the longer version.
        if known is None or _size(article) >= _size(known):
            articles[article.number] = article

    for raw in text.splitlines():
        line = _clean_line(raw)
        if not line:
            continue
        article_match = _ARTICLE_RE.match(line)
        chapter_match = _CHAPTER_RE.match(line)
        if article_match:
            close()
            body = []
            current = (article_match.group(1), article_match.group(2).strip())
        elif chapter_match:
            close()
            current, body = None, []
            chapter = f"Глава {chapter_match.group(1)}. {chapter_match.group(2).strip()}"
        elif _SECTION_RE.match(line):
            close()
            current, body = None, []
            section = line
        elif current is not None:
            body.append(line)
    close()
    return list(articles.values())


def _size(article: Article) -> int:
    return sum(len(p.text) for p in article.parts)


def live_articles(articles: list[Article]) -> list[Article]:
    """Articles that are still in force (repealed ones are not indexed)."""
    return [a for a in articles if not a.repealed]
