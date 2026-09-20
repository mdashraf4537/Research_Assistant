import json
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from pydantic import BaseModel

from app.agent.plan import GeminiResearchPlanner, PlanAndExecuteAgentService
from app.agent.service import ResearchAgentService
from app.answering.adapters import CohereReranker, DeepSeekChatClient
from app.answering.service import ResearchAnswerService
from app.core.config import get_settings
from app.core.errors import UpstreamServiceError
from app.evaluation.runner import ground_truth_sha256, load_cached_report
from app.ingestion.adapters import ChromaChunkStore, OllamaEmbeddingClient
from app.ingestion.extractors import IngestionValidationError
from app.ingestion.service import IngestionService
from app.retrieval.lexical import SqliteBm25Index
from app.retrieval.service import HybridRetriever
from app.retrieval.store import DualWriteChunkStore

router = APIRouter(tags=["system"])


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class IngestionResponse(BaseModel):
    document_id: str
    filename: str
    page_count: int
    chunk_count: int
    content_sha256: str


class SearchResult(BaseModel):
    chunk_id: str
    text: str
    filename: str
    page_number: int
    retrieval_sources: list[str]
    fused_score: float


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResult]


class CitationResponse(BaseModel):
    citation_id: str
    filename: str
    page_number: int
    excerpt: str


class AnswerResponse(BaseModel):
    answer: str
    citations: list[CitationResponse]
    retrieval_candidate_count: int


class AskRequest(BaseModel):
    question: str


class PlanAnswerResponse(AnswerResponse):
    sub_questions: list[str]
    sub_answers: list[str]


class EvaluationSummaryResponse(BaseModel):
    created_at: str
    dataset_size: int
    averages: dict[str, float]
    targets: dict[str, float]
    passed: bool


def get_ingestion_service() -> IngestionService:
    settings = get_settings()
    return IngestionService(
        embedding_client=OllamaEmbeddingClient(
            base_url=settings.ollama_base_url,
            model=settings.ollama_embedding_model,
        ),
        chunk_store=DualWriteChunkStore(
            vector_store=ChromaChunkStore(
                persist_directory=settings.chroma_persist_directory,
                collection_name=settings.chroma_collection_name,
            ),
            lexical_index=SqliteBm25Index(settings.lexical_index_path),
        ),
        max_file_size_bytes=settings.ingestion_max_file_size_bytes,
        chunk_size=settings.ingestion_chunk_size,
        chunk_overlap=settings.ingestion_chunk_overlap,
    )


def get_hybrid_retriever() -> HybridRetriever:
    settings = get_settings()
    return HybridRetriever(
        embedding_client=OllamaEmbeddingClient(
            base_url=settings.ollama_base_url,
            model=settings.ollama_embedding_model,
        ),
        semantic_store=ChromaChunkStore(
            persist_directory=settings.chroma_persist_directory,
            collection_name=settings.chroma_collection_name,
        ),
        lexical_index=SqliteBm25Index(settings.lexical_index_path),
        candidate_count=settings.retrieval_candidate_count,
        final_count=settings.retrieval_final_count,
        rrf_k=settings.retrieval_rrf_k,
    )


def get_answer_service() -> ResearchAnswerService:
    settings = get_settings()
    if settings.cohere_api_key is None or settings.deepseek_api_key is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Answer generation is not configured.",
        )
    return ResearchAnswerService(
        retriever=get_hybrid_retriever(),
        reranker=CohereReranker(
            api_key=settings.cohere_api_key.get_secret_value(),
            model=settings.cohere_rerank_model,
        ),
        llm=DeepSeekChatClient(
            api_key=settings.deepseek_api_key.get_secret_value(),
            model=settings.deepseek_model,
            max_output_tokens=settings.answer_max_completion_tokens,
        ),
    )


def get_research_agent_service() -> ResearchAgentService:
    settings = get_settings()
    if settings.gemini_api_key is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Agent orchestration is not configured.",
        )
    return ResearchAgentService(
        answer_service=get_answer_service(),
        gemini_api_key=settings.gemini_api_key.get_secret_value(),
        model=settings.gemini_model,
        recursion_limit=settings.agent_recursion_limit,
    )


def get_plan_agent_service() -> PlanAndExecuteAgentService:
    settings = get_settings()
    if settings.gemini_api_key is None or settings.deepseek_api_key is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Plan-and-execute orchestration is not configured.",
        )
    return PlanAndExecuteAgentService(
        answer_service=get_answer_service(),
        planner=GeminiResearchPlanner(
            api_key=settings.gemini_api_key.get_secret_value(),
            model=settings.gemini_model,
            max_subquestions=settings.agent_plan_max_subquestions,
        ),
        synthesizer=DeepSeekChatClient(
            api_key=settings.deepseek_api_key.get_secret_value(),
            model=settings.deepseek_model,
            max_output_tokens=settings.agent_plan_max_completion_tokens,
        ),
        max_subquestions=settings.agent_plan_max_subquestions,
    )


@router.get("/health/live", response_model=HealthResponse, summary="Liveness probe")
async def liveness() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(status="ok", service=settings.app_name, version=settings.app_version)


@router.get("/health/ready", response_model=HealthResponse, summary="Readiness probe")
async def readiness() -> HealthResponse:
    # External dependency checks are added with the retrieval layer.
    settings = get_settings()
    return HealthResponse(status="ready", service=settings.app_name, version=settings.app_version)


@router.post(
    "/documents/ingest",
    response_model=IngestionResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a research document",
)
async def ingest_document(
    file: Annotated[UploadFile, File(...)],
    service: Annotated[IngestionService, Depends(get_ingestion_service)],
) -> IngestionResponse:
    try:
        result = await service.ingest(filename=file.filename or "", content=await file.read())
    except IngestionValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    return IngestionResponse(**result.__dict__)


@router.get("/search", response_model=SearchResponse, summary="Hybrid document search")
async def hybrid_search(
    query: Annotated[str, Query(min_length=1, max_length=1000)],
    retriever: Annotated[HybridRetriever, Depends(get_hybrid_retriever)],
) -> SearchResponse:
    results = await retriever.search(query)
    return SearchResponse(
        query=query,
        results=[
            SearchResult(
                chunk_id=hit.chunk_id,
                text=hit.text,
                filename=str(hit.metadata["filename"]),
                page_number=int(hit.metadata["page_number"]),
                retrieval_sources=hit.source.split("+"),
                fused_score=hit.score,
            )
            for hit in results
        ],
    )


@router.post("/ask", response_model=AnswerResponse, summary="Answer from cited research evidence")
async def ask_research_assistant(
    request: AskRequest,
    service: Annotated[ResearchAnswerService, Depends(get_answer_service)],
) -> AnswerResponse:
    question = request.question.strip()
    if not 3 <= len(question) <= 4000:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Question must be between 3 and 4000 characters.",
        )
    try:
        result = await service.answer(question)
    except UpstreamServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    return AnswerResponse(
        answer=result.answer,
        citations=[
            CitationResponse(
                citation_id=citation.citation_id,
                filename=citation.filename,
                page_number=citation.page_number,
                excerpt=citation.excerpt,
            )
            for citation in result.citations
        ],
        retrieval_candidate_count=result.candidate_count,
    )


@router.post(
    "/agent/ask", response_model=AnswerResponse, summary="Ask through the controlled agent"
)
async def ask_research_agent(
    request: AskRequest,
    service: Annotated[ResearchAgentService, Depends(get_research_agent_service)],
) -> AnswerResponse:
    question = request.question.strip()
    if not 3 <= len(question) <= 4000:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Question must be between 3 and 4000 characters.",
        )
    try:
        result = await service.ask(question)
    except UpstreamServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    return AnswerResponse(
        answer=result.answer,
        citations=[
            CitationResponse(
                citation_id=citation.citation_id,
                filename=citation.filename,
                page_number=citation.page_number,
                excerpt=citation.excerpt,
            )
            for citation in result.citations
        ],
        retrieval_candidate_count=result.retrieval_candidate_count,
    )


@router.post(
    "/agent/plan",
    response_model=PlanAnswerResponse,
    summary="Ask through the plan-and-execute agent",
)
async def ask_plan_research_agent(
    request: AskRequest,
    service: Annotated[PlanAndExecuteAgentService, Depends(get_plan_agent_service)],
) -> PlanAnswerResponse:
    question = request.question.strip()
    if not 3 <= len(question) <= 4000:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Question must be between 3 and 4000 characters.",
        )
    try:
        result = await service.ask(question)
    except UpstreamServiceError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    return PlanAnswerResponse(
        answer=result.answer,
        citations=[
            CitationResponse(
                citation_id=citation.citation_id,
                filename=citation.filename,
                page_number=citation.page_number,
                excerpt=citation.excerpt,
            )
            for citation in result.citations
        ],
        retrieval_candidate_count=result.candidate_count,
        sub_questions=result.sub_questions,
        sub_answers=result.sub_answers,
    )


@router.get(
    "/evaluations/latest",
    response_model=EvaluationSummaryResponse,
    summary="Read the newest Ragas evaluation summary",
)
async def latest_evaluation() -> EvaluationSummaryResponse:
    """Expose aggregate scores only; detailed answers stay in report files."""
    settings = get_settings()
    results_directory = Path(settings.evaluation_results_directory)
    reports = sorted(results_directory.glob("ragas-*.json"), key=lambda path: path.stat().st_mtime)
    if not reports:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No evaluation report exists yet. Run: python -m app.evaluation.runner",
        )
    try:
        # Prefer the cached report for the current ground-truth file.  A report
        # from a different dataset may be newer, but must not be shown as this
        # dataset's quality result on the dashboard.
        dataset_path = Path("data/evaluations/ground_truth.json")
        cached = (
            load_cached_report(results_directory, ground_truth_sha256(dataset_path))
            if dataset_path.exists()
            else None
        )
        report = (
            json.loads(cached[1].read_text(encoding="utf-8"))
            if cached is not None
            else json.loads(reports[-1].read_text(encoding="utf-8"))
        )
        return EvaluationSummaryResponse(
            created_at=report["created_at"],
            dataset_size=report["dataset_size"],
            averages=report["averages"],
            targets=report["targets"],
            passed=report["passed"],
        )
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The latest evaluation report is invalid.",
        ) from exc
