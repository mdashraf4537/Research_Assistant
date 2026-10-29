import asyncio
from collections.abc import Sequence

from app.ingestion.protocols import EmbeddingClient
from app.retrieval.lexical import SqliteBm25Index
from app.retrieval.models import RetrievedChunk
from app.retrieval.protocols import SemanticChunkStore


class HybridRetriever:
    def __init__(
        self,
        embedding_client: EmbeddingClient,
        semantic_store: SemanticChunkStore,
        lexical_index: SqliteBm25Index,
        candidate_count: int,
        final_count: int,
        rrf_k: int,
    ) -> None:
        self._embedding_client = embedding_client
        self._semantic_store = semantic_store
        self._lexical_index = lexical_index
        self._candidate_count = candidate_count
        self._final_count = final_count
        self._rrf_k = rrf_k

    async def search(self, query: str) -> list[RetrievedChunk]:
        query = query.strip()
        if not query:
            return []
        lexical_task = asyncio.to_thread(self._lexical_index.search, query, self._candidate_count)
        try:
            embeddings = await self._embedding_client.embed([query])
            semantic_hits = await asyncio.to_thread(
                self._semantic_store.search, embeddings[0], self._candidate_count
            )
        except Exception:
            semantic_hits = []
        lexical_hits = await lexical_task
        if not semantic_hits:
            return lexical_hits[: self._final_count]
        return self._reciprocal_rank_fusion((lexical_hits, semantic_hits))

    def _reciprocal_rank_fusion(
        self, ranked_lists: Sequence[Sequence[RetrievedChunk]]
    ) -> list[RetrievedChunk]:
        merged: dict[str, RetrievedChunk] = {}
        scores: dict[str, float] = {}
        sources: dict[str, set[str]] = {}
        for hits in ranked_lists:
            for rank, hit in enumerate(hits, start=1):
                merged.setdefault(hit.chunk_id, hit)
                scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + 1 / (self._rrf_k + rank)
                sources.setdefault(hit.chunk_id, set()).add(hit.source)
        ranked = sorted(scores, key=scores.get, reverse=True)[: self._final_count]
        return [
            RetrievedChunk(
                chunk_id=chunk_id,
                text=merged[chunk_id].text,
                metadata=merged[chunk_id].metadata,
                source="+".join(sorted(sources[chunk_id])),
                score=scores[chunk_id],
            )
            for chunk_id in ranked
        ]
