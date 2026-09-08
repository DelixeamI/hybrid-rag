from typing import Literal

from pydantic import BaseModel, Field

Status = Literal["ok", "refused", "unverified"]


class Citation(BaseModel):
    chunk_id: str
    article: str
    parts: list[str] = Field(default_factory=list)
    quote: str


class Answer(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    refused: bool = False
    reason: str | None = None


class AskResult(BaseModel):
    """What the API returns. `status` tells whether the answer passed the validator."""

    status: Status
    answer: Answer
    issues: list[str] = Field(default_factory=list)
    best_score: float = 0.0
    timings_ms: dict[str, float] = Field(default_factory=dict)


class LLMCitation(BaseModel):
    """A citation as the model writes it: by the number of the context fragment."""

    source: int
    quote: str


class LLMReply(BaseModel):
    answer: str = ""
    citations: list[LLMCitation] = Field(default_factory=list)
    refused: bool = False
    reason: str | None = None
