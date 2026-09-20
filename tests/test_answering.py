import asyncio

from app.answering.service import ResearchAnswerService
from app.retrieval.models import RetrievedChunk


class FakeRetriever:
    async def search(self, _: str) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                chunk_id="chunk-1",
                text="Stroke research needs explainable models.",
                metadata={"filename": "paper.pdf", "page_number": 3},
                source="bm25+semantic",
            )
        ]


class FakeReranker:
    async def rerank(
        self, _: str, chunks: list[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        assert top_n == 8
        return chunks


class FakeLlm:
    async def complete(self, _: str, __: str) -> str:
        return "The paper supports explainable models [S1] and makes an unsupported claim [S99]."


def test_answer_keeps_only_known_citations() -> None:
    service = ResearchAnswerService(FakeRetriever(), FakeReranker(), FakeLlm())
    result = asyncio.run(service.answer("What does the paper recommend?"))

    assert "[S1]" in result.answer
    assert "[S99]" not in result.answer
    assert result.citations[0].page_number == 3
