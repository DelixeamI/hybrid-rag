"""FastAPI app: POST /ask, GET /health. The app depends on a `Service`, not on Postgres."""

import logging
import time
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel, Field

from tkrag.api.jsonlog import configure_logging
from tkrag.generation.schemas import AskResult

log = logging.getLogger("tkrag.api")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


@dataclass(frozen=True)
class Service:
    ask: Callable[[str], AskResult]
    health: Callable[[], dict[str, Any]]
    close: Callable[[], None] = lambda: None


def create_app(service_factory: Callable[[], Service]) -> FastAPI:
    """`service_factory` runs at startup, so importing this module loads no models."""
    configure_logging()
    state: dict[str, Service] = {}

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        state["service"] = service_factory()
        yield
        state["service"].close()

    app = FastAPI(title="Labor Code QA", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return state["service"].health()

    @app.post("/ask", response_model=AskResult)
    def ask(request: AskRequest) -> AskResult:
        request_id = uuid.uuid4().hex[:12]
        started = time.perf_counter()
        result = state["service"].ask(request.question)
        total_ms = (time.perf_counter() - started) * 1000
        log.info(
            "ask",
            extra={
                "request_id": request_id,
                "status": result.status,
                "reason": result.answer.reason,
                "best_score": round(result.best_score, 4),
                "articles": [c.article for c in result.answer.citations],
                "latency_ms": {k: round(v, 1) for k, v in result.timings_ms.items()},
                "total_ms": round(total_ms, 1),
                "question_chars": len(request.question),
            },
        )
        return result

    return app
