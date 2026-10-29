from collections.abc import Sequence

from app.ingestion.models import ChunkRecord
from app.ingestion.protocols import ChunkStore
from app.retrieval.lexical import SqliteBm25Index


class DualWriteChunkStore:
    """Keeps semantic and lexical indexes in sync at ingestion time."""

    def __init__(self, vector_store: ChunkStore, lexical_index: SqliteBm25Index) -> None:
        self._vector_store = vector_store
        self._lexical_index = lexical_index

    def upsert(self, chunks: Sequence[ChunkRecord], embeddings: Sequence[Sequence[float]]) -> None:
        self._vector_store.upsert(chunks, embeddings)
        self._lexical_index.upsert(chunks)

    def delete_document(self, filename: str) -> int:
        if hasattr(self._vector_store, "delete"):
            self._vector_store.delete(filename)
        return self._lexical_index.delete_document(filename)

