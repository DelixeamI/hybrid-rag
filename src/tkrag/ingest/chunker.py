"""Articles -> chunks. One article = one chunk unless it is too long; then split by parts."""

import re
from collections.abc import Callable
from dataclasses import dataclass, replace

from tkrag.ingest.parser import Article, Part

TokenCounter = Callable[[str], int]
DEFAULT_MAX_TOKENS = 500

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.;!?])\s+")


@dataclass(frozen=True)
class Chunk:
    key: str
    article_number: str
    article_title: str
    section: str | None
    chapter: str | None
    part_numbers: tuple[str, ...]
    breadcrumbs: str
    text: str
    token_count: int

    @property
    def embedding_text(self) -> str:
        """What gets embedded and shown to the LLM: the breadcrumbs plus the legal text."""
        return f"{self.breadcrumbs}\n{self.text}"


def _render(part: Part) -> str:
    return f"{part.number}. {part.text}" if part.number and not part.implicit else part.text


def _split_oversized(part: Part, budget: int, count: TokenCounter) -> list[Part]:
    """A single part longer than the budget is cut at sentence boundaries (then at words)."""
    if count(_render(part)) <= budget:
        return [part]
    pieces: list[str] = []
    current = ""
    for sentence in _SENTENCE_SPLIT_RE.split(part.text):
        for unit in _hard_split(sentence, budget, count):
            candidate = f"{current} {unit}".strip()
            if current and count(candidate) > budget:
                pieces.append(current)
                current = unit
            else:
                current = candidate
    if current:
        pieces.append(current)
    return [Part(part.number, p) for p in pieces]


def _hard_split(sentence: str, budget: int, count: TokenCounter) -> list[str]:
    if count(sentence) <= budget:
        return [sentence]
    words = sentence.split()
    out: list[str] = []
    current: list[str] = []
    for word in words:
        if current and count(" ".join([*current, word])) > budget:
            out.append(" ".join(current))
            current = []
        current.append(word)
    if current:
        out.append(" ".join(current))
    return out


def _make_chunk(article: Article, index: int, parts: list[Part], count: TokenCounter) -> Chunk:
    text = "\n".join(_render(p) for p in parts)
    numbers = tuple(dict.fromkeys(p.number for p in parts if p.number))
    chunk = Chunk(
        key=f"{article.number}#{index}",
        article_number=article.number,
        article_title=article.title,
        section=article.section,
        chapter=article.chapter,
        part_numbers=numbers,
        breadcrumbs=article.breadcrumbs,
        text=text,
        token_count=0,
    )
    return replace(chunk, token_count=count(chunk.embedding_text))


def chunk_article(
    article: Article, count: TokenCounter, max_tokens: int = DEFAULT_MAX_TOKENS
) -> list[Chunk]:
    """Whole article if it fits; else windows of parts overlapping by one part."""
    header_cost = count(article.breadcrumbs)
    budget = max(max_tokens - header_cost, 1)

    def fits(parts: list[Part]) -> bool:
        return count("\n".join(_render(p) for p in parts)) <= budget

    if fits(list(article.parts)):
        return [_make_chunk(article, 0, list(article.parts), count)]

    units = [u for p in article.parts for u in _split_oversized(p, budget, count)]
    chunks: list[Chunk] = []
    start = 0
    while start < len(units):
        end = start + 1
        while end < len(units) and fits(units[start : end + 1]):
            end += 1
        chunks.append(_make_chunk(article, len(chunks), units[start:end], count))
        if end >= len(units):
            break
        start = max(end - 1, start + 1)  # overlap one part, but always move forward
    return chunks
