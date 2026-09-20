from collections.abc import Sequence
from typing import Protocol

from app.retrieval.models import RetrievedChunk


class Reranker(Protocol):
    async def rerank(
        self, query: str, chunks: Sequence[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]: ...


class LanguageModel(Protocol):
    async def complete(self, system_prompt: str, user_prompt: str) -> str: ...
