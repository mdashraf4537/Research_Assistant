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


class FakeRetriever:
    async def search(self, _: str):
        from app.retrieval.models import RetrievedChunk
        return [
            RetrievedChunk(
                chunk_id="chunk-1",
                text="Accuracy was 98.00% on the stroke dataset.",
                metadata={"filename": "paper.pdf", "page_number": 4},
                source="bm25",
                score=0.95,
            )
        ]


class FakeReranker:
    async def rerank(self, _: str, chunks: list, top_n: int):
        return chunks[:top_n]


class FakeLlm:
    async def complete(self, _: str, __: str) -> str:
        return "Summary with findings [S1]."


class FakeLexicalIndex:
    def get_chunks_by_document(self, filename: str, page_number: int | None = None, limit: int = 50):
        from app.retrieval.models import RetrievedChunk
        return [
            RetrievedChunk(
                chunk_id="chunk-1",
                text="Detailed page content for reading.",
                metadata={"filename": filename, "page_number": page_number or 1},
                source="bm25",
                score=1.0,
            )
        ]

    def get_chunk_by_id(self, chunk_id: str):
        from app.retrieval.models import RetrievedChunk
        return RetrievedChunk(
            chunk_id=chunk_id,
            text="Exact chunk text.",
            metadata={"filename": "paper.pdf", "page_number": 1},
            source="bm25",
            score=1.0,
        )


def test_expanded_agent_tools() -> None:
    answer_svc = FakeAnswerService()
    answer_svc._retriever = FakeRetriever()
    answer_svc._reranker = FakeReranker()
    lex_index = FakeLexicalIndex()
    llm = FakeLlm()

    tool = LlamaIndexRagTool(answer_svc, lexical_index=lex_index, llm=llm)

    # Test search_corpus
    search_res = json.loads(asyncio.run(tool.search_corpus("stroke accuracy")))
    assert search_res["total_candidates"] == 1
    assert search_res["results"][0]["filename"] == "paper.pdf"

    # Test read_chunk
    read_res = json.loads(asyncio.run(tool.read_chunk("paper.pdf", page_number=2)))
    assert "Detailed page content" in read_res["content"]

    # Test summarize_document
    summary_res = json.loads(asyncio.run(tool.summarize_document("paper.pdf")))
    assert "Summary with findings [S1]." in summary_res["answer"]
    assert len(summary_res["citations"]) == 1

    # Test compare_documents
    compare_res = json.loads(asyncio.run(tool.compare_documents("accuracy", "paper1.pdf", "paper2.pdf")))
    assert "[S1]" in compare_res["answer"]

    # Test extract_metrics_table
    metrics_res = json.loads(asyncio.run(tool.extract_metrics_table("accuracy")))
    assert "Summary with findings [S1]." in metrics_res["answer"]
    assert len(metrics_res["citations"]) == 1

    # Test LlamaIndex FunctionTools exposure
    tools = tool.as_llama_index_tools()
    assert len(tools) == 6
    names = {t.metadata.name for t in tools}
    assert "research_documents" in names
    assert "search_corpus" in names
    assert "read_chunk" in names
    assert "summarize_document" in names
    assert "compare_documents" in names
    assert "extract_metrics_table" in names

