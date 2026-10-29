import json
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status
from pydantic import BaseModel

from app.agent.plan import GeminiResearchPlanner, PlanAndExecuteAgentService
from app.agent.service import ResearchAgentService
from app.answering.adapters import CohereReranker, DeepSeekChatClient
from app.answering.service import ResearchAnswerService
from app.answering.web_rag import WebSearchRagService
from app.core.config import get_settings
from app.core.errors import UpstreamServiceError
from app.core.tracking import (
    approve_user_access,
    check_user_query_quota,
    delete_user_document,
    get_recent_queries,
    get_tracked_users,
    get_user_documents,
    get_user_quota_info,
    identify_or_register_user,
    is_document_owned_by_other_users,
    log_user_query,
    record_user_document,
    reject_user_access,
    request_user_access,
)
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
    source_mode: str = "knowledge_base"


class AskRequest(BaseModel):
    question: str
    search_mode: str = "auto"
    user_email: str | None = None


class IdentifyRequest(BaseModel):
    email: str


class IdentifyResponse(BaseModel):
    email: str
    first_seen_at: str
    last_seen_at: str
    query_count: int
    query_limit: int = 2
    approval_status: str = "none"
    is_new: bool
    is_admin: bool = False


class QuotaStatusResponse(BaseModel):
    email: str
    is_admin: bool
    query_count: int
    query_limit: int
    approval_status: str
    approval_requested_at: str | None
    can_query: bool
    admin_email: str


class RequestApprovalRequest(BaseModel):
    email: str


class ApproveUserRequest(BaseModel):
    admin_email: str
    target_email: str
    additional_queries: int = 10
    unlimited: bool = False


class RejectUserRequest(BaseModel):
    admin_email: str
    target_email: str


class AdminUsersResponse(BaseModel):
    total_users: int
    users: list[dict]
    recent_queries: list[dict]


class PlanAnswerResponse(AnswerResponse):
    sub_questions: list[str]
    sub_answers: list[str]


class EvaluationSummaryResponse(BaseModel):
    created_at: str
    dataset_size: int
    averages: dict[str, float]
    targets: dict[str, float]
    passed: bool


class DocumentItem(BaseModel):
    filename: str
    chunk_count: int
    page_count: int


class DeleteDocumentResponse(BaseModel):
    filename: str
    deleted_chunks: int


class AgentToolInfo(BaseModel):
    name: str
    description: str



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
    llm = DeepSeekChatClient(
        api_key=settings.deepseek_api_key.get_secret_value(),
        model=settings.deepseek_model,
        max_output_tokens=settings.answer_max_completion_tokens,
    )
    web_rag = WebSearchRagService(
        llm=llm,
        gemini_api_key=(
            settings.gemini_api_key.get_secret_value()
            if settings.gemini_api_key is not None
            else None
        ),
        gemini_model=settings.gemini_model,
    )
    return ResearchAnswerService(
        retriever=get_hybrid_retriever(),
        reranker=CohereReranker(
            api_key=settings.cohere_api_key.get_secret_value(),
            model=settings.cohere_rerank_model,
        ),
        llm=llm,
        final_count=settings.retrieval_final_count,
        web_rag=web_rag,
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
        lexical_index=SqliteBm25Index(settings.lexical_index_path),
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
    user_email: Annotated[str | None, Query()] = None,
) -> IngestionResponse:
    try:
        result = await service.ingest(filename=file.filename or "", content=await file.read())
        if user_email:
            record_user_document(user_email, result.filename, result.chunk_count)
    except IngestionValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    return IngestionResponse(**result.__dict__)


@router.get(
    "/documents",
    response_model=list[DocumentItem],
    summary="List all indexed research documents in the library",
)
async def list_documents(user_email: Annotated[str | None, Query()] = None) -> list[DocumentItem]:
    settings = get_settings()
    index = SqliteBm25Index(settings.lexical_index_path)
    all_docs = index.list_indexed_documents()
    all_docs_map = {str(doc["filename"]): doc for doc in all_docs}

    if user_email:
        cleaned_email = user_email.strip().lower()
        user_docs = get_user_documents(cleaned_email)
        
        # If user is mdashraf228144@gmail.com and has no records yet, link all current docs
        if not user_docs and cleaned_email == "mdashraf228144@gmail.com":
            for d in all_docs:
                record_user_document(cleaned_email, str(d["filename"]), int(d["chunk_count"]))
            user_docs = get_user_documents(cleaned_email)

        if user_docs:
            result = []
            for ud in user_docs:
                fn = ud["filename"]
                if fn in all_docs_map:
                    doc = all_docs_map[fn]
                    result.append(
                        DocumentItem(
                            filename=fn,
                            chunk_count=int(doc["chunk_count"]),
                            page_count=int(doc["page_count"]),
                        )
                    )
            return result
        # Any other user with no documents uploaded yet has an isolated, empty library
        return []

    return [
        DocumentItem(
            filename=str(doc["filename"]),
            chunk_count=int(doc["chunk_count"]),
            page_count=int(doc["page_count"]),
        )
        for doc in all_docs
    ]


@router.delete(
    "/documents/{filename}",
    response_model=DeleteDocumentResponse,
    summary="Delete an indexed document from semantic and lexical stores",
)
async def delete_document(
    filename: str,
    service: Annotated[IngestionService, Depends(get_ingestion_service)],
    user_email: Annotated[str | None, Query()] = None,
) -> DeleteDocumentResponse:
    chunk_store = service._chunk_store
    deleted = 0
    if user_email:
        delete_user_document(user_email, filename)
        # If no other user has this document, delete chunks from physical stores
        if not is_document_owned_by_other_users(filename, user_email):
            if hasattr(chunk_store, "delete_document"):
                deleted = chunk_store.delete_document(filename)
    else:
        if hasattr(chunk_store, "delete_document"):
            deleted = chunk_store.delete_document(filename)
    return DeleteDocumentResponse(filename=filename, deleted_chunks=deleted)



@router.get("/search", response_model=SearchResponse, summary="Hybrid document search")
async def hybrid_search(
    query: Annotated[str, Query(min_length=1, max_length=1000)],
    retriever: Annotated[HybridRetriever, Depends(get_hybrid_retriever)],
    user_email: Annotated[str | None, Query()] = None,
) -> SearchResponse:
    results = await retriever.search(query)
    if user_email:
        cleaned_email = user_email.strip().lower()
        user_docs = get_user_documents(cleaned_email)
        allowed_filenames = {d["filename"] for d in user_docs}
        results = [hit for hit in results if str(hit.metadata.get("filename")) in allowed_filenames]
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


@router.post(
    "/auth/identify",
    response_model=IdentifyResponse,
    summary="Register or verify user email for access tracking",
)
async def identify_user(request: IdentifyRequest) -> IdentifyResponse:
    try:
        user_info = identify_or_register_user(request.email)
        return IdentifyResponse(**user_info)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc


@router.get(
    "/admin/users",
    response_model=AdminUsersResponse,
    summary="List all tracked users and query activity for admin oversight",
)
async def admin_list_users() -> AdminUsersResponse:
    users = get_tracked_users()
    recent = get_recent_queries(limit=100)
    return AdminUsersResponse(
        total_users=len(users),
        users=users,
        recent_queries=recent,
    )


@router.get(
    "/auth/quota-status",
    response_model=QuotaStatusResponse,
    summary="Get current user query quota and access status",
)
async def get_quota_status(email: Annotated[str, Query()]) -> QuotaStatusResponse:
    settings = get_settings()
    info = get_user_quota_info(email, settings.admin_email)
    return QuotaStatusResponse(**info, admin_email=settings.admin_email)


@router.post(
    "/auth/request-approval",
    response_model=QuotaStatusResponse,
    summary="Request administrator approval for additional queries",
)
async def request_approval(request: RequestApprovalRequest) -> QuotaStatusResponse:
    settings = get_settings()
    info = request_user_access(request.email)
    return QuotaStatusResponse(**info, admin_email=settings.admin_email)


@router.post(
    "/admin/approve",
    response_model=QuotaStatusResponse,
    summary="Admin approval to grant additional queries",
)
async def approve_user(request: ApproveUserRequest) -> QuotaStatusResponse:
    settings = get_settings()
    if request.admin_email.strip().lower() != settings.admin_email.strip().lower():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Only the administrator can approve query limits."
        )
    info = approve_user_access(
        request.target_email,
        additional_queries=request.additional_queries,
        unlimited=request.unlimited,
    )
    return QuotaStatusResponse(**info, admin_email=settings.admin_email)


@router.post(
    "/admin/reject",
    response_model=QuotaStatusResponse,
    summary="Admin rejection of query limit request",
)
async def reject_user(request: RejectUserRequest) -> QuotaStatusResponse:
    settings = get_settings()
    if request.admin_email.strip().lower() != settings.admin_email.strip().lower():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Only the administrator can modify access status."
        )
    info = reject_user_access(request.target_email)
    return QuotaStatusResponse(**info, admin_email=settings.admin_email)


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
    settings = get_settings()
    user_email = (request.user_email or "").strip().lower()
    if not user_email:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User email is required to submit research queries.",
        )
    allowed, reason, quota_info = check_user_query_quota(user_email, settings.admin_email)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Query limit reached ({quota_info['query_count']}/{quota_info['query_limit']}). You have used your complimentary queries. Please request administrator approval to continue.",
        )
    log_user_query(user_email, question, f"direct:{request.search_mode}")
    try:
        result = await service.answer(question, search_mode=request.search_mode)
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
        source_mode=result.source_mode,
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
    settings = get_settings()
    user_email = (request.user_email or "").strip().lower()
    if not user_email:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User email is required to submit research queries.",
        )
    allowed, reason, quota_info = check_user_query_quota(user_email, settings.admin_email)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Query limit reached ({quota_info['query_count']}/{quota_info['query_limit']}). You have used your complimentary queries. Please request administrator approval to continue.",
        )
    log_user_query(user_email, question, "agent")
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
    settings = get_settings()
    user_email = (request.user_email or "").strip().lower()
    if not user_email:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User email is required to submit research queries.",
        )
    allowed, reason, quota_info = check_user_query_quota(user_email, settings.admin_email)
    if not allowed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Query limit reached ({quota_info['query_count']}/{quota_info['query_limit']}). You have used your complimentary queries. Please request administrator approval to continue.",
        )
    log_user_query(user_email, question, "plan")
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
    "/agent/tools",
    response_model=list[AgentToolInfo],
    summary="List available agent tools and capabilities",
)
async def list_agent_tools() -> list[AgentToolInfo]:
    return [
        AgentToolInfo(
            name="research_documents",
            description="Answer research questions with rigorous citation-backed evidence.",
        ),
        AgentToolInfo(
            name="search_corpus",
            description="Raw multi-source snippet retrieval without LLM synthesis.",
        ),
        AgentToolInfo(
            name="read_chunk",
            description="Inspect full passages or pages from any document by filename and page.",
        ),
        AgentToolInfo(
            name="summarize_document",
            description="Comprehensive structured summary of a research paper.",
        ),
        AgentToolInfo(
            name="compare_documents",
            description="Side-by-side comparative analysis of methodologies and performance.",
        ),
        AgentToolInfo(
            name="extract_metrics_table",
            description="Extract quantitative metrics into a cited Markdown comparison table.",
        ),
        AgentToolInfo(
            name="plan_and_execute",
            description="Decompose complex questions into sub-questions and synthesize cited findings.",
        ),
    ]


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
