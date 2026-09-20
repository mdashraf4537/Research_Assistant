import asyncio

from app.agent.plan import (
    GeminiResearchPlanner,
    PlanAndExecuteAgentService,
)
from app.answering.models import AnswerResult, Citation


class FakePlanner:
    def __init__(self, sub_questions: list[str]) -> None:
        self._sub_questions = sub_questions

    async def plan(self, _: str) -> list[str]:
        return self._sub_questions


class FakeAnswerServiceWithCitations:
    async def answer(self, question: str) -> AnswerResult:
        if question == "Q1?":
            return AnswerResult(
                answer="First finding uses [S1].",
                citations=[
                    Citation(
                        citation_id="S1",
                        chunk_id="chunk-1",
                        filename="paper-a.pdf",
                        page_number=3,
                        excerpt="Excerpt one.",
                    )
                ],
                candidate_count=10,
            )
        return AnswerResult(
            answer="Second finding uses [S1] and [S2].",
            citations=[
                Citation(
                    citation_id="S1",
                    chunk_id="chunk-1",
                    filename="paper-a.pdf",
                    page_number=3,
                    excerpt="Excerpt one.",
                ),
                Citation(
                    citation_id="S2",
                    chunk_id="chunk-2",
                    filename="paper-b.pdf",
                    page_number=7,
                    excerpt="Excerpt two.",
                ),
            ],
            candidate_count=14,
        )


class FakeSynthesizer:
    def __init__(self, answer: str) -> None:
        self._answer = answer

    async def complete(self, _: str, __: str) -> str:
        return self._answer


class FakeEmptyAnswerService:
    async def answer(self, _: str) -> AnswerResult:
        return AnswerResult(answer="", citations=[], candidate_count=0)


def _service(
    planner: FakePlanner,
    answering: object | None = None,
    synthesizer=None,
) -> PlanAndExecuteAgentService:
    return PlanAndExecuteAgentService(
        answer_service=answering or FakeAnswerServiceWithCitations(),
        planner=planner,
        synthesizer=synthesizer or FakeSynthesizer("Final [S1] with [S2] and [S99]."),
        max_subquestions=4,
    )


def test_plan_merges_and_deduplicates_citations() -> None:
    result = asyncio.run(_service(FakePlanner(["Q1?", "Q2?"])).ask("Original?"))

    assert result.sub_questions == ["Q1?", "Q2?"]
    assert [citation.chunk_id for citation in result.citations] == ["chunk-1", "chunk-2"]
    assert result.citations[0].citation_id == "S1"
    assert result.citations[1].citation_id == "S2"
    assert "[S99]" not in result.answer
    assert "Final [S1] with [S2] and ." == result.answer
    assert result.sub_answers[0] == "First finding uses [S1]."
    assert result.sub_answers[1] == "Second finding uses [S1] and [S2]."
    assert result.candidate_count == 24


def test_plan_falls_back_to_single_question_when_planner_is_empty() -> None:
    result = asyncio.run(_service(FakePlanner([])).ask("Original?"))

    assert result.sub_questions == ["Original?"]
    assert len(result.sub_answers) == 1


def test_plan_reports_insufficient_evidence_without_calling_synthesizer() -> None:
    class ExplodingSynthesizer:
        async def complete(self, _: str, __: str) -> str:
            raise AssertionError("Synthesizer must not run without evidence.")

    service = PlanAndExecuteAgentService(
        answer_service=FakeEmptyAnswerService(),
        planner=FakePlanner(["Q1?"]),
        synthesizer=ExplodingSynthesizer(),
        max_subquestions=4,
    )
    result = asyncio.run(service.ask("Original?"))

    assert result.citations == []
    assert result.answer == "I could not find relevant evidence in the indexed documents."


def test_parse_sub_questions_extracts_code_fenced_json() -> None:
    planner = GeminiResearchPlanner(api_key="key", model="model", max_subquestions=4)
    content = '```json\n{"sub_questions": ["a?", "b?"]}\n```'

    assert planner._parse_sub_questions(content) == ["a?", "b?"]


def test_parse_sub_questions_rejects_invalid_and_malformed_payloads() -> None:
    planner = GeminiResearchPlanner(api_key="key", model="model", max_subquestions=4)

    assert planner._parse_sub_questions("No JSON here at all.") is None
    assert planner._parse_sub_questions('{"other_key": []}') is None
    assert planner._parse_sub_questions('{"sub_questions": "not a list"}') is None