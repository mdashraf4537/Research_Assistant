import json

from app.answering.service import ResearchAnswerService


class LlamaIndexRagTool:
    """LlamaIndex tool facade over the citation-first RAG answer service."""

    def __init__(self, answer_service: ResearchAnswerService) -> None:
        self._answer_service = answer_service

    async def query(self, question: str) -> str:
        result = await self._answer_service.answer(question)
        return json.dumps(
            {
                "answer": result.answer,
                "citations": [
                    {
                        "id": citation.citation_id,
                        "filename": citation.filename,
                        "page_number": citation.page_number,
                        "excerpt": citation.excerpt,
                    }
                    for citation in result.citations
                ],
                "retrieval_candidate_count": result.candidate_count,
            }
        )

    def as_llama_index_tool(self):
        """Create the native LlamaIndex FunctionTool only when the agent is enabled."""
        from llama_index.core.tools import FunctionTool

        return FunctionTool.from_defaults(
            async_fn=self.query,
            name="query_research_corpus",
            description=(
                "Answer a research question from the indexed documents and return "
                "citation-backed evidence."
            ),
        )
