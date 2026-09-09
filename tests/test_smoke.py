import pytest

import tkrag
from tkrag.config import Settings, get_settings


def test_package_has_version() -> None:
    assert tkrag.__version__


def test_settings_defaults() -> None:
    settings = Settings(_env_file=None)
    assert settings.rrf_k == 10
    assert settings.rrf_dense_weight == 3.0
    assert settings.embedding_model == "intfloat/multilingual-e5-small"
    assert settings.llm_mode == "recorded"


def test_settings_read_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RRF_K", "30")
    assert Settings(_env_file=None).rrf_k == 30


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()


def test_dataset_overrides_point_the_same_corpus_at_another_question_set() -> None:
    from pathlib import Path

    from tkrag.datasets import get_dataset

    base = get_dataset("fixture", Settings(_env_file=None))
    other = get_dataset("fixture", Settings(_env_file=None), Path("x.jsonl"), "results_test")
    assert other.corpus_path == base.corpus_path and other.document == base.document
    assert other.golden_path == Path("x.jsonl") and other.report_stem == "results_test"
    assert base.report_stem == "results"
