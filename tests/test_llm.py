from pathlib import Path

import pytest

from tkrag.llm import (
    CassetteMissingError,
    FakeLLM,
    RecordedLLM,
    RecordingLLM,
    cassette_key,
)


def test_cassette_key_depends_on_model_and_both_prompts() -> None:
    base = cassette_key("m", "sys", "user")
    assert base == cassette_key("m", "sys", "user")
    assert (
        len(
            {
                base,
                cassette_key("m2", "sys", "user"),
                cassette_key("m", "sys2", "user"),
                cassette_key("m", "sys", "user2"),
            }
        )
        == 4
    )


def test_fake_llm_is_deterministic_and_records_calls() -> None:
    llm = FakeLLM(lambda system, user: user.upper())
    assert llm.complete("s", "abc") == "ABC"
    assert llm.calls == [("s", "abc")]


def test_record_then_replay(tmp_path: Path) -> None:
    real = FakeLLM(lambda s, u: '{"answer": "ok"}', model="real-model")
    RecordingLLM(real, tmp_path).complete("sys", "question")
    replay = RecordedLLM(tmp_path, model="real-model")
    assert replay.complete("sys", "question") == '{"answer": "ok"}'


def test_missing_cassette_has_a_helpful_error(tmp_path: Path) -> None:
    with pytest.raises(CassetteMissingError, match="make record"):
        RecordedLLM(tmp_path, model="m").complete("sys", "never recorded")


def test_cassette_for_another_model_is_not_reused(tmp_path: Path) -> None:
    RecordingLLM(FakeLLM(model="model-a"), tmp_path).complete("s", "u")
    with pytest.raises(CassetteMissingError):
        RecordedLLM(tmp_path, model="model-b").complete("s", "u")


def test_cassettes_are_kept_apart_per_dataset() -> None:
    from tkrag.components import cassettes_dir
    from tkrag.config import Settings

    settings = Settings(_env_file=None)
    fixture, full = cassettes_dir(settings, "fixture"), cassettes_dir(settings, "full")
    assert full == fixture / "full"
    assert fixture.name == "cassettes"


def test_resume_replays_existing_cassettes_and_asks_only_for_missing_ones(tmp_path: Path) -> None:
    real = FakeLLM(lambda s, u: f"answer to {u}", model="m")
    RecordingLLM(real, tmp_path).complete("sys", "first")
    assert len(real.calls) == 1

    resumed = RecordingLLM(real, tmp_path, resume=True)
    assert resumed.complete("sys", "first") == "answer to first"
    assert len(real.calls) == 1  # replayed, the model was not asked again
    assert resumed.complete("sys", "second") == "answer to second"
    assert len(real.calls) == 2  # missing one was recorded

    RecordingLLM(real, tmp_path).complete("sys", "first")
    assert len(real.calls) == 3  # without resume, record asks the model again


def test_unused_cassettes_lists_only_stale_ones_of_the_same_model(tmp_path: Path) -> None:
    other = FakeLLM(lambda s, u: "x", model="other-model")
    mine = FakeLLM(lambda s, u: "y", model="my-model")
    RecordingLLM(other, tmp_path).complete("sys", "belongs to another model")
    recorder = RecordingLLM(mine, tmp_path)
    recorder.complete("sys", "still used")
    recorder.complete("sys", "stale after a prompt change")

    replay = RecordedLLM(tmp_path, model="my-model")
    assert len(replay.unused_cassettes()) == 2  # nothing replayed yet; the other model's is hidden
    replay.complete("sys", "still used")
    stale = replay.unused_cassettes()
    assert len(stale) == 1
    assert "stale after a prompt change" in stale[0].read_text(encoding="utf-8")
