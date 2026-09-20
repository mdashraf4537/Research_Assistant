from collections.abc import Sequence
from typing import Protocol

from app.retrieval.models import RetrievedChunk


class SemanticChunkStore(Protocol):
    def search(self, embedding: Sequence[float], limit: int) -> list[RetrievedChunk]: ...
