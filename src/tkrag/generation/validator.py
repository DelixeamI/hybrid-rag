"""Deterministic post-validator: the answer may not contain facts the cited chunks do not have.

Checks (all string/regex, no model):
  * at least one citation, and every quote is a real passage of its chunk. A quote the model
    reworded is replaced by the passage it most resembles (when the resemblance is high), so the
    user only ever sees verbatim text; a quote that resembles nothing rejects the answer;
  * the answer has [n] markers and each points to a cited fragment;
  * every number with a unit (28 дней, 2 месяца, 10 %, ...) is in the cited chunks
    (two weeks and 14 days count as the same);
  * every article number mentioned (статья 81, ст. 140) is one of the cited articles or is
    referred to inside the cited chunks.

Not checked: numbers without a unit, dates, and whether the claim *means* what the chunk
means. That is the job of answer-correctness evaluation, not of this guard.
"""

import re
from dataclasses import dataclass, field

from tkrag.retrieval.types import Hit
from tkrag.textnorm import (
    Quantity,
    content_stems,
    extract_article_refs,
    extract_quantities,
    squash,
)

_MARKER_RE = re.compile(r"\[(\d+)\]")
_ELLIPSIS_RE = re.compile(r"\s*(?:…|\.{3})\s*")
_SENTENCE_RE = re.compile(r"(?<=[.;!?])\s+")
MIN_QUOTE_OVERLAP = 0.6  # Jaccard overlap of content words needed to repair a reworded quote
_MAX_WINDOW = 3  # a repaired quote spans at most this many consecutive lines


@dataclass(frozen=True)
class CitedSource:
    number: int  # 1-based number of the fragment in the prompt
    hit: Hit
    quote: str


@dataclass(frozen=True)
class Validation:
    ok: bool
    issues: list[str]
    # Quotes to show, aligned with `cited`: the model's own when verbatim, else the repaired one.
    quotes: list[str] = field(default_factory=list)


def validate_answer(answer_text: str, cited: list[CitedSource]) -> Validation:
    if not cited:
        return Validation(False, ["no citations"])
    quote_issues, quotes = _check_quotes(cited)
    issues = list(quote_issues)
    issues += _check_markers(answer_text, cited)
    plain = _MARKER_RE.sub(" ", answer_text)
    issues += _check_quantities(plain, cited)
    issues += _check_articles(plain, cited)
    return Validation(not issues, issues, quotes)


def _spans(text: str) -> list[str]:
    """Candidate passages: sentences, lines, and windows of a few consecutive lines."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    spans = list(lines)
    for line in lines:
        spans += [s for s in _SENTENCE_RE.split(line) if s and s != line]
    for size in range(2, _MAX_WINDOW + 1):
        spans += [" ".join(lines[i : i + size]) for i in range(len(lines) - size + 1)]
    return spans


def locate_quote(quote: str, text: str) -> str | None:
    """A verbatim passage of `text` that stands for `quote`, or None if nothing resembles it.

    Verbatim quotes (an ellipsis may skip words) are returned unchanged. A reworded one is
    replaced by the passage with the highest Jaccard overlap of content words, if that overlap
    reaches MIN_QUOTE_OVERLAP.
    """
    segments = [s for s in _ELLIPSIS_RE.split(quote) if squash(s)]
    if not segments:
        return None
    flat = squash(text)
    if all(squash(s) in flat for s in segments):
        return quote
    wanted = content_stems(quote)
    best, best_score = None, 0.0
    for span in _spans(text):
        have = content_stems(span)
        score = len(wanted & have) / len(wanted | have)
        if score > best_score:
            best, best_score = span, score
    return best if best_score >= MIN_QUOTE_OVERLAP else None


def _check_quotes(cited: list[CitedSource]) -> tuple[list[str], list[str]]:
    issues: list[str] = []
    quotes: list[str] = []
    for source in cited:
        found = locate_quote(source.quote, source.hit.text)
        if found is None:
            issues.append(f"[{source.number}] quote does not match the fragment")
            quotes.append(source.quote)
        else:
            quotes.append(found)
    return issues, quotes


def _check_markers(answer_text: str, cited: list[CitedSource]) -> list[str]:
    markers = {int(m) for m in _MARKER_RE.findall(answer_text)}
    if not markers:
        return ["answer has no [n] references"]
    cited_numbers = {s.number for s in cited}
    return [
        f"reference [{m}] points to an uncited fragment" for m in sorted(markers - cited_numbers)
    ]


def _with_equivalents(quantities: set[Quantity]) -> set[Quantity]:
    """Add the other spelling of the same duration: 2 weeks = 14 days (weeks are always 7 days)."""
    extra: set[Quantity] = set()
    for value, unit in quantities:
        number = float(value)
        if unit == "week":
            extra.add((f"{number * 7:g}", "day"))
        elif unit == "day" and number % 7 == 0:
            extra.add((f"{number / 7:g}", "week"))
    return quantities | extra


def _check_quantities(plain: str, cited: list[CitedSource]) -> list[str]:
    allowed = _with_equivalents(
        set().union(*(extract_quantities(s.hit.context_text) for s in cited))
    )
    return [
        f"{value} {unit} is not in the cited fragments"
        for value, unit in sorted(extract_quantities(plain) - allowed)
    ]


def _check_articles(plain: str, cited: list[CitedSource]) -> list[str]:
    allowed = {s.hit.article_number for s in cited}
    for source in cited:
        allowed |= extract_article_refs(source.hit.context_text)
    return [
        f"article {number} is not in the cited fragments"
        for number in sorted(extract_article_refs(plain) - allowed)
    ]
