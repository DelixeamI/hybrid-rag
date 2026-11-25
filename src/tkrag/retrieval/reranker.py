"""Second-stage reranking with a cross-encoder."""

from dataclasses import replace
from typing import Protocol

from tkrag.retrieval.types import Hit
from tkrag.textnorm import stems


class Reranker(Protocol):
    def rerank(self, query: str, hits: list[Hit], top_k: int) -> list[Hit]:
        """Return the `top_k` best hits, best first; `score` becomes a probability in [0, 1]."""
        ...


class CrossEncoderReranker:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder

        # One output label -> sentence-transformers applies a sigmoid: score is in [0, 1].
        try:  # offline first: a cached model needs no network
            self._model = CrossEncoder(model_name, max_length=512, local_files_only=True)
        except OSError:  # not downloaded yet
            self._model = CrossEncoder(model_name, max_length=512)
        self.runtime = self._describe_runtime()

    def _describe_runtime(self) -> str:
        """Device, dtype and free GPU memory at load time, written into the eval report.

        Latency depends on these: on Windows a GPU short of memory silently spills into system
        RAM and the reranker gets several times slower without any error.
        """
        import torch

        device = self._model.model.device
        dtype = str(next(self._model.model.parameters()).dtype).removeprefix("torch.")
        if device.type != "cuda":
            return f"{device}, {dtype}"
        free, total = torch.cuda.mem_get_info(device)
        name = torch.cuda.get_device_name(device)
        return (
            f"{device}, {dtype}, {name}; free VRAM after loading "
            f"{free / 2**30:.1f} of {total / 2**30:.1f} GiB"
        )

    def rerank(self, query: str, hits: list[Hit], top_k: int) -> list[Hit]:
        if not hits:
            return []
        scores = self._model.predict([(query, h.context_text) for h in hits], batch_size=8)
        rescored = [replace(h, score=float(s)) for h, s in zip(hits, scores, strict=True)]
        return sorted(rescored, key=lambda h: -h.score)[:top_k]


class LexicalReranker:
    """Deterministic stand-in for tests: share of query word stems found in the chunk."""

    def rerank(self, query: str, hits: list[Hit], top_k: int) -> list[Hit]:
        query_stems = stems(query)
        rescored = [
            replace(h, score=len(query_stems & stems(h.context_text)) / max(len(query_stems), 1))
            for h in hits
        ]
        return sorted(rescored, key=lambda h: (-h.score, h.chunk_id))[:top_k]
