import re
from typing import TYPE_CHECKING

from app.answering.models import AnswerResult, Citation
from app.answering.protocols import LanguageModel, Reranker
from app.retrieval.service import HybridRetriever

if TYPE_CHECKING:
    from app.answering.web_rag import WebSearchRagService

SYSTEM_PROMPT = """You are a rigorous research assistant.
Answer only from the supplied evidence. Cite every factual claim using the supplied
bracketed citation IDs, such as [S1]. Do not invent citations or use knowledge not
present in the evidence. If the evidence is insufficient, say so clearly. Copy numbers/terms exactly. No synonyms, no elaboration.
One factual sentence = one [Sx]. Do not add held-out/corresponding-author type details unless in evidence."""

class ResearchAnswerService:
    def __init__(
        self,
        retriever: HybridRetriever,
        reranker: Reranker,
        llm: LanguageModel,
        final_count: int = 8,
        web_rag: "WebSearchRagService | None" = None,
    ) -> None:
        self._retriever = retriever
        self._reranker = reranker
        self._llm = llm
        self._final_count = final_count
        self._web_rag = web_rag

    async def answer(
        self,
        question: str,
        enable_reflective_retry: bool = True,
        search_mode: str = "auto",
    ) -> AnswerResult:
        if search_mode == "web_only" and self._web_rag is not None:
            return await self._web_rag.answer(question)

        result = await self._generate_answer(question)
        if enable_reflective_retry and self._needs_reflective_retry(result):
            reformulated = await self._reformulate_query(question)
            if reformulated and reformulated.strip().lower() != question.strip().lower():
                retry_result = await self._generate_answer(reformulated, rerank_query=question)
                if not self._needs_reflective_retry(retry_result) or len(retry_result.citations) > len(result.citations):
                    result = retry_result

        if search_mode != "kb_only" and self._web_rag is not None and self._needs_web_fallback(result):
            web_result = await self._web_rag.answer(question)
            if web_result.citations:
                return web_result

        return result

    async def _generate_answer(self, query: str, rerank_query: str | None = None) -> AnswerResult:
        candidates = await self._retriever.search(query)
        if not candidates:
            return AnswerResult(
                answer="I could not find relevant evidence in the indexed documents.",
                citations=[],
                candidate_count=0,
            )
        rerank_target = rerank_query or query
        reranked = await self._reranker.rerank(rerank_target, candidates, top_n=self._final_count)
        citations = [
            Citation(
                citation_id=f"S{index}",
                chunk_id=chunk.chunk_id,
                filename=str(chunk.metadata["filename"]),
                page_number=int(chunk.metadata["page_number"]),
                excerpt=chunk.text,
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
            f"Question:\n{rerank_target}\n\nEvidence:\n{evidence}",
        )
        allowed_ids = {citation.citation_id for citation in citations}
        answer = self._remove_unknown_citations(answer, allowed_ids)
        return AnswerResult(answer=answer, citations=citations, candidate_count=len(candidates))

    async def _reformulate_query(self, question: str) -> str:
        try:
            system_prompt = (
                "You are an expert scientific query reformulator. The research query did not yield "
                "sufficient evidence in our medical and AI research corpus. "
                "Rewrite the question into a concise, keyword-rich retrieval query including key terms, "
                "domain synonyms, dataset names, or model architectures. "
                "Output ONLY the reformulated query text, without explanations or quotes."
            )
            reformulated = await self._llm.complete(
                system_prompt, f"Original question:\n{question}\n\nReformulated query:"
            )
            cleaned = re.sub(r'["\n]', " ", reformulated).strip()
            return cleaned if cleaned else question
        except Exception:
            return question

    @staticmethod
    def _needs_reflective_retry(result: AnswerResult) -> bool:
        if not result.citations:
            return True
        if len(result.citations) >= 3:
            return False
        lowered = result.answer.lower()
        return (
            "could not find relevant evidence" in lowered
            or "insufficient evidence" in lowered
            or "not enough evidence" in lowered
        )

    @staticmethod
    def _needs_web_fallback(result: AnswerResult) -> bool:
        if not result.citations:
            return True
        if not re.search(r"\[S\d+\]", result.answer):
            return True
        lowered = result.answer.lower()
        insufficiency_markers = (
            "could not find relevant evidence",
            "insufficient",
            "not present in the",
            "not mentioned in the",
            "does not contain",
            "does not provide",
            "do not contain",
            "do not provide",
            "no information",
            "cannot answer",
            "not covered",
            "supplied evidence does not",
            "provided evidence does not",
            "evidence does not specify",
        )
        return any(marker in lowered for marker in insufficiency_markers)

    @staticmethod
    def _remove_unknown_citations(answer: str, allowed_ids: set[str]) -> str:
        return re.sub(
            r"\[S(\d+)\]",
            lambda match: match.group(0) if f"S{match.group(1)}" in allowed_ids else "",
            answer,
        )

