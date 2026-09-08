"""Application settings, loaded from environment variables and an optional .env file."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Database
    database_url: str = "postgresql://tkrag:tkrag@localhost:5432/tkrag"

    # Models
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_dim: int = 384
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    # Retrieval
    # k and the dense weight were chosen on the DEV half of the golden set
    # (reports/experiment_hybrid.md, DESIGN D5). Set RRF_K=60, RRF_DENSE_WEIGHT=1 for classic RRF.
    rrf_k: int = Field(default=10, ge=1)
    rrf_dense_weight: float = Field(default=3.0, gt=0)
    candidates_top_n: int = Field(default=20, ge=1)
    final_top_k: int = Field(default=5, ge=1)

    # LLM: "fake" and "recorded" need no network; "openai" calls llm_base_url
    llm_mode: Literal["fake", "recorded", "openai"] = "recorded"
    llm_base_url: str = "https://generativelanguage.googleapis.com/v1beta/openai/"
    llm_model: str = "gemini-3.8-flash"
    llm_api_key: str = "not-needed"
    llm_min_interval: float = Field(default=0.0, ge=0.0)  # seconds between calls (rate limits)

    # Generation
    # Chosen by `make calibrate` on the fixture golden; DESIGN.md D6 explains how fragile it is.
    refusal_threshold: float = Field(default=0.003, ge=0.0, le=1.0)  # reranker score
    context_chunks: int = Field(default=5, ge=1)

    # Paths
    raw_data_dir: Path = Path("data/raw")
    golden_path: Path = Path("data/golden.jsonl")
    cassettes_dir: Path = Path("tests/cassettes")
    reports_dir: Path = Path("reports")
    thresholds_path: Path = Path("eval_thresholds.yaml")


@lru_cache
def get_settings() -> Settings:
    return Settings()
