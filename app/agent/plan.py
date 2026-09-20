"""Plan-and-execute research agent.

A self-contained addition over the existing controlled agent: the question is
decomposed into sub-questions, each is answered by the existing citation-first
ResearchAnswerService, and the sub-answers are merged into one cited answer.
"""

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from app.answering.models import AnswerResult, Citation
from app.answering.protocols import LanguageModel
from app.answering.service import ResearchAnswerService

PLAN_SYSTEM_PROMPT = (
    "You are a research planning assistant. Decompose the user's research question "
    "into a small list of focused sub-questions that, when answered from the indexed "
    "research corpus, together fully answer the original question. Each sub-question "
    "must be self-contained and answerable from scientific documents. Return ONLY a "
    "JSON object with a single key \"sub_questions\" whose value is a list of 1 to "
    "{max_subquestions} strings. Do not include code fences, prose, or markdown."
)

SYNTHESIS_SYSTEM_PROMPT = (
    "You are a rigorous research assistant. Merge the supplied sub-answers into one "
    "coherent cited answer for the original question. Cite every factual claim using "
    "the supplied bracketed citation IDs, such as [S1]. Use only citation IDs that are "
    "listed in the available sources and used by the sub-answers. Do not invent "
    "citations or use knowledge not present in the evidence. If the evidence is "
    "insufficient, say so clearly."
)

_INSUFFICIENT_EVIDENCE = "I could not find relevant evidence in the indexed documents."

_SUB_QUESTION_RE = re.compile(r"\[S(\d+)\]")


@dataclass(frozen=True)
class PlanResult:
    answer: str
    citations: list[Citation]
    candidate_count: int
    sub_questions: list[str]
    sub_answers: list[str]


class ResearchPlanner(Protocol):
    async def plan(self, question: str) -> list[str]:
        """Return focused sub-questions for the given research question."""
        ...


class GeminiResearchPlanner:
    """Decompose a question using Google Gemini and a structured JSON response."""

    def __init__(self, api_key: str, model: str, max_subquestions: int) -> None:
        self._api_key = api_key
        self._model = model
        self._max_subquestions = max(1, max_subquestions)

    async def plan(self, question: str) -> list[str]:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model=self._model,
            google_api_key=self._api_key,
            temperature=0,
        )
        response = await llm.ainvoke(
            [
                SystemMessage(
                    content=PLAN_SYSTEM_PROMPT.format(max_subquestions=self._max_subquestions)
                ),
                HumanMessage(content=question),
            ]
        )
        content = response.content
        if not isinstance(content, str):
            return []
        return self._parse_sub_questions(content) or []

    @staticmethod
    def _parse_sub_questions(content: str) -> list[str] | None:
        cleaned = re.sub(r"^```(?:json)?\s*", "", content.strip())
        cleaned = re.sub(r"\s*```$", "", cleaned)
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        try:
            payload = json.loads(cleaned[start : end + 1])
        except (json.JSONDecodeError, TypeError):
            return None
        if not isinstance(payload, dict):
            return None
        questions = payload.get("sub_questions")
        if not isinstance(questions, list):
            return None
        return [str(q).strip() for q in questions if isinstance(q, str) and q.strip()]


class PlanAndExecuteAgentService:
    """Decompose, answer each part with the citation-first service, then synthesize."""

    def __init__(
        self,
        answer_service: ResearchAnswerService,
        planner: ResearchPlanner,
        synthesizer: LanguageModel,
        max_subquestions: int,
    ) -> None:
        self._answer_service = answer_service
        self._planner = planner
        self._synthesizer = synthesizer
        self._max_subquestions = max(1, max_subquestions)

    async def ask(self, question: str) -> PlanResult:
        sub_questions = (await self._planner.plan(question))[: self._max_subquestions]
        if not sub_questions:
            sub_questions = [question]

        partial_results = [await self._answer_service.answer(sub) for sub in sub_questions]
        if not any(result.citations for result in partial_results):
            return PlanResult(
                answer=_INSUFFICIENT_EVIDENCE,
                citations=[],
                candidate_count=sum(r.candidate_count for r in partial_results),
                sub_questions=sub_questions,
                sub_answers=[],
            )

        merged_citations, sub_answers = self._merge_partials(partial_results)
        user_prompt = self._build_synthesis_prompt(
            question, sub_questions, sub_answers, merged_citations
        )
        answer = await self._synthesizer.complete(SYNTHESIS_SYSTEM_PROMPT, user_prompt)
        allowed_ids = {citation.citation_id for citation in merged_citations}
        answer = ResearchAnswerService._remove_unknown_citations(answer, allowed_ids)
        return PlanResult(
            answer=answer,
            citations=merged_citations,
            candidate_count=sum(r.candidate_count for r in partial_results),
            sub_questions=sub_questions,
            sub_answers=sub_answers,
        )

    def _merge_partials(
        self, partial_results: Sequence[AnswerResult], /
    ) -> tuple[list[Citation], list[str]]:
        """Deduplicate citations by chunk and rewrite [S#] to global IDs."""
        merged: list[Citation] = []
        id_by_chunk: dict[str, str] = {}
        sub_answers: list[str] = []

        for result in partial_results:
            local_map: dict[str, str] = {}
            for citation in result.citations:
                if citation.chunk_id and citation.chunk_id in id_by_chunk:
                    local_map[citation.citation_id] = id_by_chunk[citation.chunk_id]
                    continue
                global_id = f"S{len(merged) + 1}"
                if citation.chunk_id:
                    id_by_chunk[citation.chunk_id] = global_id
                local_map[citation.citation_id] = global_id
                merged.append(
                    Citation(
                        citation_id=global_id,
                        chunk_id=citation.chunk_id,
                        filename=citation.filename,
                        page_number=citation.page_number,
                        excerpt=citation.excerpt,
                    )
                )
            sub_answers.append(self._rewrite_citations(result.answer, local_map))

        return merged, sub_answers

    @staticmethod
    def _rewrite_citations(answer: str, local_map: dict[str, str]) -> str:
        if not local_map:
            return answer

        def _replace(match: re.Match[str]) -> str:
            global_id = local_map.get(f"S{match.group(1)}", "")
            return f"[{global_id}]" if global_id else ""

        return _SUB_QUESTION_RE.sub(_replace, answer)

    @staticmethod
    def _build_synthesis_prompt(
        question: str,
        sub_questions: list[str],
        sub_answers: list[str],
        citations: list[Citation],
    ) -> str:
        lines = [f"Original question:\n{question}", "", "Sub-answers:"]
        for index, (sub_question, sub_answer) in enumerate(
            zip(sub_questions, sub_answers, strict=True), start=1
        ):
            lines.append(f"Sub-question {index}: {sub_question}")
            lines.append(f"Sub-answer: {sub_answer}")
            lines.append("")
        lines.append("Available sources:")
        for citation in citations:
            lines.append(
                f"{citation.citation_id} | {citation.filename}, "
                f"page {citation.page_number} | {citation.excerpt[:400]}"
            )
        return "\n".join(lines)