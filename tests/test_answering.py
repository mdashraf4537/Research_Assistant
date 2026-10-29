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


def test_deepseek_fallback_to_non_thinking_on_token_limit() -> None:
    import json
    import unittest.mock
    import httpx
    from app.answering.adapters import DeepSeekChatClient

    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        calls.append(payload)
        if payload.get("thinking", {}).get("type") == "enabled":
            return httpx.Response(
                status_code=200,
                json={
                    "choices": [
                        {
                            "finish_reason": "length",
                            "message": {
                                "role": "assistant",
                                "content": "",
                                "reasoning_content": "Thinking steps that exhausted tokens...",
                            },
                        }
                    ]
                },
            )
        return httpx.Response(
            status_code=200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": "| Metric | Value |\n|---|---|\n| Accuracy | 98% |",
                        },
                    }
                ]
            },
        )

    transport = httpx.MockTransport(handler)
    client = DeepSeekChatClient("test_key", "deepseek-v4-flash", 8192, thinking_enabled=True, reasoning_effort="low")

    class PatchedAsyncClient(httpx.AsyncClient):
        def __init__(self, *args, **kwargs):
            kwargs["transport"] = transport
            super().__init__(*args, **kwargs)

    with unittest.mock.patch("httpx.AsyncClient", PatchedAsyncClient):
        res = asyncio.run(client.complete("system", "user"))

    assert "| Metric | Value |" in res
    assert len(calls) == 2
    assert calls[0]["thinking"]["type"] == "enabled"
    assert calls[0]["reasoning_effort"] == "low"
    assert calls[0]["max_tokens"] == 8192
    assert calls[1]["thinking"]["type"] == "disabled"
    assert "reasoning_effort" not in calls[1]
    assert calls[1]["max_tokens"] == 8192

