"""retrieve -> refuse-if-weak -> generate -> validate -> answer, refusal or degradation."""

import json
import re
import time

from pydantic import ValidationError

from tkrag.generation.prompt import SYSTEM_PROMPT, build_user_prompt
from tkrag.generation.schemas import Answer, AskResult, Citation, LLMReply
from tkrag.generation.validator import CitedSource, validate_answer
from tkrag.llm import LLMClient
from tkrag.retrieval.retriever import Searcher
from tkrag.retrieval.types import Hit

NO_ANSWER_TEXT = "В Трудовом кодексе не найдено ответа на этот вопрос."
DEGRADED_TEXT = "Не удалось подтвердить ответ цитатами. Вот статьи, которые относятся к вопросу:"
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_FALLBACK_QUOTE_CHARS = 300


def parse_reply(raw: str) -> LLMReply | None:
    """Parse the model reply; None when it is not the JSON we asked for."""
    try:
        return LLMReply.model_validate(json.loads(_FENCE_RE.sub("", raw.strip())))
    except (json.JSONDecodeError, ValidationError):
        return None


def _short(text: str, limit: int = _FALLBACK_QUOTE_CHARS) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[:limit].rsplit(" ", 1)[0] + "…"


def _citation(hit: Hit, quote: str) -> Citation:
    return Citation(
        chunk_id=hit.chunk_key, article=hit.article_number, parts=hit.part_numbers, quote=quote
    )


class Answerer:
    def __init__(
        self, retriever: Searcher, llm: LLMClient, threshold: float, context_chunks: int = 5
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._threshold = threshold
        self._context_chunks = context_chunks

    def ask(self, question: str) -> AskResult:
        retrieval = self._retriever.retrieve(question, "hybrid_rerank", self._context_chunks)
        timings = dict(retrieval.timings_ms)
        hits = retrieval.hits
        if not hits or retrieval.best_score < self._threshold:
            return self._refusal("low_retrieval_score", retrieval.best_score, timings)

        start = time.perf_counter()
        raw = self._llm.complete(SYSTEM_PROMPT, build_user_prompt(question, hits))
        timings["generate"] = (time.perf_counter() - start) * 1000

        reply = parse_reply(raw)
        if reply is None:
            return self._degrade(hits, "invalid_llm_output", ["reply is not valid JSON"], timings)
        if reply.refused:
            return self._refusal(
                reply.reason or "model_found_no_answer", retrieval.best_score, timings
            )

        start = time.perf_counter()
        cited = self._cited_sources(reply, hits)
        verdict = validate_answer(reply.answer, cited)
        timings["validate"] = (time.perf_counter() - start) * 1000
        if not verdict.ok:
            return self._degrade(hits, "unverified", verdict.issues, timings, retrieval.best_score)

        citations = [_citation(s.hit, q) for s, q in zip(cited, verdict.quotes, strict=True)]
        answer = Answer(answer=reply.answer, citations=citations)
        return AskResult(
            status="ok", answer=answer, best_score=retrieval.best_score, timings_ms=timings
        )

    @staticmethod
    def _cited_sources(reply: LLMReply, hits: list[Hit]) -> list[CitedSource]:
        """Map the model's fragment numbers to real chunks; unknown numbers are dropped."""
        return [
            CitedSource(c.source, hits[c.source - 1], c.quote)
            for c in reply.citations
            if 1 <= c.source <= len(hits)
        ]

    @staticmethod
    def _refusal(reason: str, score: float, timings: dict[str, float]) -> AskResult:
        answer = Answer(answer=NO_ANSWER_TEXT, refused=True, reason=reason)
        return AskResult(status="refused", answer=answer, best_score=score, timings_ms=timings)

    @staticmethod
    def _degrade(
        hits: list[Hit],
        reason: str,
        issues: list[str],
        timings: dict[str, float],
        score: float = 0.0,
    ) -> AskResult:
        """Do not show a claim we could not verify: show the relevant articles instead."""
        citations = [_citation(h, _short(h.text)) for h in hits[:3]]
        answer = Answer(answer=DEGRADED_TEXT, citations=citations, reason=reason)
        return AskResult(
            status="unverified",
            answer=answer,
            issues=issues,
            best_score=score or (hits[0].score if hits else 0.0),
            timings_ms=timings,
        )
