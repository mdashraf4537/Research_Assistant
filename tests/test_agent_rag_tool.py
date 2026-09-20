import asyncio
import json

from app.agent.rag_tool import LlamaIndexRagTool
from app.answering.models import AnswerResult, Citation


class FakeAnswerService:
    async def answer(self, _: str) -> AnswerResult:
        return AnswerResult(
            answer="The finding is supported [S1].",
            citations=[
                Citation(
                    citation_id="S1",
                    chunk_id="chunk-1",
                    filename="paper.pdf",
                    page_number=4,
                    excerpt="Supporting evidence.",
                )
            ],
            candidate_count=12,
        )


def test_llama_index_tool_payload_preserves_citation_metadata() -> None:
    tool = LlamaIndexRagTool(FakeAnswerService())
    payload = json.loads(asyncio.run(tool.query("What did the study find?")))

    assert payload["answer"] == "The finding is supported [S1]."
    assert payload["citations"][0]["page_number"] == 4
    assert payload["retrieval_candidate_count"] == 12
