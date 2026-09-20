from collections.abc import Sequence
from typing import Protocol

from app.ingestion.models import ChunkRecord


class EmbeddingClient(Protocol):
    async def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class ChunkStore(Protocol):
    def upsert(
        self, chunks: Sequence[ChunkRecord], embeddings: Sequence[Sequence[float]]
    ) -> None: ...
