from collections.abc import Sequence

import chromadb
import httpx

from app.ingestion.models import ChunkRecord
from app.retrieval.models import RetrievedChunk


class OllamaEmbeddingClient:
    def __init__(self, base_url: str, model: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self._base_url}/api/embed", json={"model": self._model, "input": list(texts)}
                )
                response.raise_for_status()
            embeddings = response.json().get("embeddings")
        except httpx.HTTPError as exc:
            raise RuntimeError("Ollama embedding service is unavailable.") from exc
        if not isinstance(embeddings, list) or len(embeddings) != len(texts):
            raise RuntimeError("Ollama returned an invalid embedding response.")
        return embeddings


class ChromaChunkStore:
    def __init__(self, persist_directory: str, collection_name: str) -> None:
        client = chromadb.PersistentClient(path=persist_directory)
        self._collection = client.get_or_create_collection(
            name=collection_name,
            embedding_function=None,
            metadata={"description": "Citation-ready research document chunks"},
        )

    def upsert(self, chunks: Sequence[ChunkRecord], embeddings: Sequence[Sequence[float]]) -> None:
        self._collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            embeddings=[list(vector) for vector in embeddings],
            metadatas=[chunk.metadata for chunk in chunks],
        )

    def search(self, embedding: Sequence[float], limit: int) -> list[RetrievedChunk]:
        result = self._collection.query(
            query_embeddings=[list(embedding)],
            n_results=limit,
            include=["documents", "metadatas", "distances"],
        )
        return [
            RetrievedChunk(
                chunk_id=chunk_id,
                text=text,
                metadata=metadata,
                source="semantic",
                score=float(distance),
            )
            for chunk_id, text, metadata, distance in zip(
                result["ids"][0],
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
                strict=True,
            )
        ]

    def delete(self, filename: str) -> None:
        try:
            self._collection.delete(where={"filename": filename})
        except Exception:
            pass

