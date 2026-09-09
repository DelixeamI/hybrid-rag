"""Text embedders. e5 models need 'query: ' / 'passage: ' prefixes; hiding that here."""

import hashlib
import math
import re
from typing import Protocol


class Embedder(Protocol):
    name: str  # identifies the model; part of the index fingerprint
    dim: int

    def embed_passages(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...

    def count_tokens(self, text: str) -> int: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, dim: int) -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.name = model_name
        self.dim = dim

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            [f"passage: {t}" for t in texts],
            batch_size=16,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [[float(x) for x in v] for v in vectors]

    def embed_query(self, text: str) -> list[float]:
        vector = self._model.encode(
            f"query: {text}", normalize_embeddings=True, show_progress_bar=False
        )
        return [float(x) for x in vector]

    def count_tokens(self, text: str) -> int:
        return len(self._model.tokenizer.encode(text, add_special_tokens=False))


_WORD_RE = re.compile(r"\w+")


class HashEmbedder:
    """Bag-of-words hashing embedder for tests. Not semantic: only word overlap matters."""

    def __init__(self, dim: int = 384) -> None:
        self.name = f"hash-{dim}"
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for word in _WORD_RE.findall(text.lower()):
            digest = hashlib.md5(word[:6].encode("utf-8"), usedforsecurity=False).digest()
            vector[int.from_bytes(digest[:4], "big") % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in vector)) or 1.0
        return [x / norm for x in vector]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def count_tokens(self, text: str) -> int:
        return len(_WORD_RE.findall(text))
