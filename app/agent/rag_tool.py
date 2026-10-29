import json
from collections.abc import Sequence

from app.answering.models import Citation
from app.answering.protocols import LanguageModel
from app.answering.service import ResearchAnswerService
from app.retrieval.lexical import SqliteBm25Index


class LlamaIndexRagTool:
    """LlamaIndex tool facade exposing specialized research and aggregation tools."""

    def __init__(
        self,
        answer_service: ResearchAnswerService,
        lexical_index: SqliteBm25Index | None = None,
        llm: LanguageModel | None = None,
    ) -> None:
        self._answer_service = answer_service
        self._retriever = getattr(answer_service, "_retriever", None)
        self._llm = llm or getattr(answer_service, "_llm", None)
        self._lexical_index = lexical_index or getattr(self._retriever, "_lexical_index", None)

    async def query(self, question: str) -> str:
        """Answer a research question from the indexed corpus with cited evidence."""
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

    async def search_corpus(self, query: str, limit: int = 5) -> str:
        """Retrieve raw matching passages from indexed documents without synthesis."""
        if not self._retriever:
            return json.dumps({"results": [], "total_candidates": 0})
        candidates = await self._retriever.search(query)
        top_hits = candidates[:limit]
        return json.dumps(
            {
                "results": [
                    {
                        "chunk_id": c.chunk_id,
                        "filename": str(c.metadata.get("filename", "")),
                        "page_number": int(c.metadata.get("page_number", 0)),
                        "excerpt": c.text[:400],
                        "source": c.source,
                        "score": round(c.score, 4),
                    }
                    for c in top_hits
                ],
                "total_candidates": len(candidates),
            }
        )

    async def read_chunk(
        self, filename: str, page_number: int = 1, chunk_id: str = ""
    ) -> str:
        """Read specific passages or full pages from a paper by filename and page_number."""
        if not self._lexical_index:
            return json.dumps({"error": "Lexical index not available", "content": ""})
        if chunk_id:
            hit = self._lexical_index.get_chunk_by_id(chunk_id)
            chunks = [hit] if hit else []
        else:
            chunks = self._lexical_index.get_chunks_by_document(
                filename, page_number=page_number, limit=10
            )
        if not chunks:
            return json.dumps(
                {"error": f"No content found for {filename} (page {page_number})", "content": ""}
            )
        combined_text = "\n\n---\n\n".join(c.text for c in chunks)
        return json.dumps(
            {
                "filename": filename,
                "page_number": page_number,
                "chunk_count": len(chunks),
                "content": combined_text,
            }
        )

    async def summarize_document(self, filename: str) -> str:
        """Generate a comprehensive structured summary of a specific research paper."""
        if not self._lexical_index or not self._llm:
            return json.dumps(
                {"answer": f"Cannot summarize {filename}: dependencies unavailable.", "citations": []}
            )
        chunks = self._lexical_index.get_chunks_by_document(filename, limit=14)
        if not chunks:
            return json.dumps(
                {"answer": f"Document '{filename}' not found in index.", "citations": []}
            )
        citations = [
            Citation(
                citation_id=f"S{idx}",
                chunk_id=c.chunk_id,
                filename=str(c.metadata.get("filename", filename)),
                page_number=int(c.metadata.get("page_number", 1)),
                excerpt=c.text,
            )
            for idx, c in enumerate(chunks, start=1)
        ]
        evidence = "\n\n".join(
            f"[{c.citation_id}] {c.filename}, page {c.page_number}\n{c.excerpt}"
            for c in citations
        )
        system_prompt = (
            "You are an expert scientific researcher. Provide a comprehensive structured summary "
            "of the document based strictly on the provided evidence. Include:\n"
            "1. Research Objective & Domain\n"
            "2. Proposed Methodology / Architecture\n"
            "3. Key Findings & Quantitative Performance\n"
            "4. Significance and Clinical / Technical Impact\n"
            "Cite all claims with bracketed citation IDs [S#]. Do not invent facts."
        )
        user_prompt = f"Document: {filename}\n\nEvidence:\n{evidence}"
        summary = await self._llm.complete(system_prompt, user_prompt)
        allowed_ids = {c.citation_id for c in citations}
        summary = ResearchAnswerService._remove_unknown_citations(summary, allowed_ids)
        return json.dumps(
            {
                "answer": summary,
                "citations": [
                    {
                        "id": c.citation_id,
                        "filename": c.filename,
                        "page_number": c.page_number,
                        "excerpt": c.excerpt,
                    }
                    for c in citations
                ],
                "retrieval_candidate_count": len(chunks),
            }
        )

    async def compare_documents(
        self, topic: str, paper_a: str = "", paper_b: str = ""
    ) -> str:
        """Compare methodologies, models, and findings across multiple research papers."""
        prompt = f"Compare papers on {topic}"
        if paper_a or paper_b:
            prompt += f": {paper_a} vs {paper_b}"
        result = await self._answer_service.answer(prompt)
        return json.dumps(
            {
                "answer": result.answer,
                "citations": [
                    {
                        "id": c.citation_id,
                        "filename": c.filename,
                        "page_number": c.page_number,
                        "excerpt": c.excerpt,
                    }
                    for c in result.citations
                ],
                "retrieval_candidate_count": result.candidate_count,
            }
        )

    async def extract_metrics_table(
        self, focus: str = "performance metrics, accuracy, precision, recall, F1, FID, dataset size"
    ) -> str:
        """Extract quantitative performance metrics across all papers into a cited Markdown table."""
        if not self._retriever or not self._llm:
            return json.dumps({
                "answer": "Metrics extraction is unavailable without retriever or language model.",
                "citations": [],
                "retrieval_candidate_count": 0,
            })
        query = f"quantitative evaluation metrics accuracy precision recall F1 FID dataset {focus}".strip()
        candidates = await self._retriever.search(query)
        if not candidates:
            return json.dumps({
                "answer": "No quantitative metrics found in the indexed corpus.",
                "citations": [],
                "retrieval_candidate_count": 0,
            })
        reranked = await self._answer_service._reranker.rerank(query, candidates, top_n=10)
        citations = [
            Citation(
                citation_id=f"S{idx}",
                chunk_id=c.chunk_id,
                filename=str(c.metadata.get("filename", "")),
                page_number=int(c.metadata.get("page_number", 1)),
                excerpt=c.text,
            )
            for idx, c in enumerate(reranked, start=1)
        ]
        evidence = "\n\n".join(
            f"[{c.citation_id}] {c.filename}, page {c.page_number}\n{c.excerpt}"
            for c in citations
        )
        system_prompt = (
            "You are an expert scientific data aggregator. Extract all quantitative performance metrics, "
            "accuracies (e.g., 95%, 96%, 98.00%), error rates, FID scores, dataset sample sizes, and model architectures "
            "from the evidence. Format the findings as a clean, complete Markdown table with exact numbers and citations:\n\n"
            "| Paper / Reference | Model / Method | Dataset & Modality | Metric | Reported Value | Citation |\n"
            "|---|---|---|---|---|---|\n\n"
            "Follow the table with a concise summary comparing the results across papers. "
            "Cite every single row and claim using bracketed citation IDs [S#]. "
            "Do not round or modify any numbers. Do not invent any metrics."
        )
        user_prompt = f"Focus:\n{focus}\n\nEvidence:\n{evidence}"
        table_result = await self._llm.complete(system_prompt, user_prompt)
        allowed_ids = {c.citation_id for c in citations}
        table_result = ResearchAnswerService._remove_unknown_citations(table_result, allowed_ids)
        return json.dumps(
            {
                "answer": table_result,
                "citations": [
                    {
                        "id": c.citation_id,
                        "filename": c.filename,
                        "page_number": c.page_number,
                        "excerpt": c.excerpt,
                    }
                    for c in citations
                ],
                "retrieval_candidate_count": len(candidates),
            }
        )

    def as_llama_index_tools(self) -> list:
        """Create the suite of LlamaIndex FunctionTools for agent orchestration."""
        from llama_index.core.tools import FunctionTool

        return [
            FunctionTool.from_defaults(
                async_fn=self.query,
                name="research_documents",
                description="Answer a research question from the indexed corpus with cited evidence.",
            ),
            FunctionTool.from_defaults(
                async_fn=self.search_corpus,
                name="search_corpus",
                description="Retrieve raw matching passages from indexed documents without LLM synthesis.",
            ),
            FunctionTool.from_defaults(
                async_fn=self.read_chunk,
                name="read_chunk",
                description="Read specific passages or full pages from a paper by filename and page_number.",
            ),
            FunctionTool.from_defaults(
                async_fn=self.summarize_document,
                name="summarize_document",
                description="Generate a comprehensive structured summary of a specific research paper.",
            ),
            FunctionTool.from_defaults(
                async_fn=self.compare_documents,
                name="compare_documents",
                description="Compare methodologies, models, and findings across multiple research papers.",
            ),
            FunctionTool.from_defaults(
                async_fn=self.extract_metrics_table,
                name="extract_metrics_table",
                description="Extract quantitative performance metrics across all papers into a cited Markdown table.",
            ),
        ]

    def as_llama_index_tool(self):
        """Create the primary LlamaIndex FunctionTool for single-tool mode."""
        return self.as_llama_index_tools()[0]
