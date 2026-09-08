"""Deterministic LLM for unit tests."""

import json
from collections.abc import Callable


def refuse_everything(system: str, user: str) -> str:
    return json.dumps(
        {"answer": "", "citations": [], "refused": True, "reason": "fake_llm"}, ensure_ascii=False
    )


class FakeLLM:
    """Answers with a user-supplied function; records every call for assertions."""

    def __init__(
        self,
        responder: Callable[[str, str], str] = refuse_everything,
        model: str = "fake-llm",
    ) -> None:
        self.model = model
        self._responder = responder
        self.calls: list[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self._responder(system, user)
