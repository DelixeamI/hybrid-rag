"""LLM interface. The rest of the code depends only on this, never on a vendor SDK."""

import hashlib
from typing import Protocol


class LLMClient(Protocol):
    model: str

    def complete(self, system: str, user: str) -> str:
        """Return the raw text of the model reply. Must be deterministic where possible."""
        ...


def cassette_key(model: str, system: str, user: str) -> str:
    """sha256 of model + prompt. The system prompt is part of the prompt."""
    payload = "\x1f".join((model, system, user)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
