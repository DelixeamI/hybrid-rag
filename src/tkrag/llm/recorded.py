"""Cassette-based LLM: replay recorded answers offline, or record them from a real model."""

import json
from pathlib import Path

from tkrag.llm.base import LLMClient, cassette_key


class CassetteMissingError(LookupError):
    """No recorded answer for this model + prompt. Run `make record` to create it."""


def _path(directory: Path, key: str) -> Path:
    return directory / f"{key}.json"


class RecordedLLM:
    def __init__(self, directory: Path, model: str) -> None:
        self.model = model
        self._directory = directory
        self._used: set[str] = set()

    def unused_cassettes(self) -> list[Path]:
        """Cassettes of THIS model that no call has replayed so far (stale after prompt changes).

        Cassettes of other models in the same folder are never listed.
        """
        stale = []
        for path in sorted(self._directory.glob("*.json")):
            if path.stem in self._used:
                continue
            if json.loads(path.read_text(encoding="utf-8")).get("model") == self.model:
                stale.append(path)
        return stale

    def complete(self, system: str, user: str) -> str:
        key = cassette_key(self.model, system, user)
        path = _path(self._directory, key)
        self._used.add(key)
        if not path.exists():
            raise CassetteMissingError(
                f"no cassette {key[:12]}… for model {self.model!r} in {self._directory}. "
                "The prompt or the model changed: run `make record`."
            )
        return str(json.loads(path.read_text(encoding="utf-8"))["response"])


class RecordingLLM:
    """Wraps a real client and writes a cassette for every call (used by `make record`).

    With `resume=True` an existing cassette is replayed instead of asking the model again, so an
    interrupted recording can continue where it stopped.
    """

    def __init__(self, inner: LLMClient, directory: Path, resume: bool = False) -> None:
        self.model = inner.model
        self._inner = inner
        self._directory = directory
        self._resume = resume

    def complete(self, system: str, user: str) -> str:
        key = cassette_key(self.model, system, user)
        existing = _path(self._directory, key)
        if self._resume and existing.exists():
            return str(json.loads(existing.read_text(encoding="utf-8"))["response"])
        response = self._inner.complete(system, user)
        self._directory.mkdir(parents=True, exist_ok=True)
        record = {"model": self.model, "system": system, "user": user, "response": response}
        _path(self._directory, key).write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return response
