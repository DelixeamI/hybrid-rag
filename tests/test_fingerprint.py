import pytest

from tkrag.ingest import fingerprint
from tkrag.ingest.fingerprint import index_fingerprint

BASE = ("sha", "model-a", 384, 500)


def test_fingerprint_is_stable_and_sensitive_to_every_input() -> None:
    base = index_fingerprint(*BASE)
    assert base == index_fingerprint(*BASE)
    variants = [
        index_fingerprint("other-sha", "model-a", 384, 500),  # the document changed
        index_fingerprint("sha", "model-b", 384, 500),  # embedding model
        index_fingerprint("sha", "model-a", 768, 500),  # vector dimension
        index_fingerprint("sha", "model-a", 384, 300),  # chunk size
    ]
    assert len({base, *variants}) == 5


def test_bumping_a_code_version_changes_the_fingerprint(monkeypatch: pytest.MonkeyPatch) -> None:
    base = index_fingerprint(*BASE)
    monkeypatch.setattr(fingerprint, "PARSER_VERSION", fingerprint.PARSER_VERSION + 1)
    after_parser = index_fingerprint(*BASE)
    monkeypatch.setattr(fingerprint, "CHUNKER_VERSION", fingerprint.CHUNKER_VERSION + 1)
    after_both = index_fingerprint(*BASE)
    assert len({base, after_parser, after_both}) == 3
