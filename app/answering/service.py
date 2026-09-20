import re

from app.answering.models import AnswerResult, Citation
from app.answering.protocols import LanguageModel, Reranker
from app.retrieval.service import HybridRetriever

SYSTEM_PROMPT = """You are a rigorous research assistant.
Answer only from the supplied evidence. Cite every factual claim using the supplied
bracketed citation IDs, such as [S1]. Do not invent citations or use knowledge not
present in the evidence. If the evidence is insufficient, say so clearly."""


class ResearchAnswerService:
    def __init__(self, retriever: HybridRetriever, reranker: Reranker, llm: LanguageModel) -> None:
        self._retriever = retriever
        self._reranker = reranker
        self._llm = llm

    async def answer(self, question: str) -> AnswerResult:
        candidates = await self._retriever.search(question)
        if not candidates:
            return AnswerResult(
                answer="I could not find relevant evidence in the indexed documents.",
                citations=[],
                candidate_count=0,
            )
        reranked = await self._reranker.rerank(question, candidates, top_n=8)
        citations = [
            Citation(
                citation_id=f"S{index}",
                chunk_id=chunk.chunk_id,
                filename=str(chunk.metadata["filename"]),
                page_number=int(chunk.metadata["page_number"]),
                excerpt=chunk.text[:500],
            )
            for index, chunk in enumerate(reranked, start=1)
        ]
        evidence = "\n\n".join(
            f"[{citation.citation_id}] {citation.filename}, page {citation.page_number}\n"
            f"{chunk.text}"
            for citation, chunk in zip(citations, reranked, strict=True)
        )
        answer = await self._llm.complete(
            SYSTEM_PROMPT,
            f"Question:\n{question}\n\nEvidence:\n{evidence}",
        )
        allowed_ids = {citation.citation_id for citation in citations}
        answer = self._remove_unknown_citations(answer, allowed_ids)
        return AnswerResult(answer=answer, citations=citations, candidate_count=len(candidates))

    @staticmethod
    def _remove_unknown_citations(answer: str, allowed_ids: set[str]) -> str:
        return re.sub(
            r"\[S(\d+)\]",
            lambda match: match.group(0) if f"S{match.group(1)}" in allowed_ids else "",
            answer,
        )
