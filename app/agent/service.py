import json
import re
from dataclasses import dataclass
from typing import Any

from app.agent.rag_tool import LlamaIndexRagTool
from app.answering.models import Citation
from app.answering.service import ResearchAnswerService
from app.retrieval.lexical import SqliteBm25Index

AGENT_SYSTEM_PROMPT = """You are an expert autonomous scientific research coordinator.
Select the most appropriate tool to answer the user's research query:
- Use `extract_metrics_table` when asked to aggregate, extract numbers, or compare metrics across papers into a markdown table.
- Use `compare_documents` when asked to compare findings, architectures, or models across papers.
- Use `summarize_document` when asked to summarize a specific paper.
- Use `read_chunk` when you need full page or passage details from a specific document.
- Use `search_corpus` for raw multi-source snippet retrieval.
- Use `research_documents` for targeted research questions requiring cited answers.

Always call the most relevant tool. Do not invent sources, browse the web, or use external ungrounded facts. After the tool returns, preserve its answer and citations accurately."""


@dataclass(frozen=True)
class AgentAnswer:
    answer: str
    citations: list[Citation]
    retrieval_candidate_count: int


class ResearchAgentService:
    def __init__(
        self,
        answer_service: ResearchAnswerService,
        gemini_api_key: str,
        model: str,
        recursion_limit: int,
        lexical_index: SqliteBm25Index | None = None,
    ) -> None:
        self._answer_service = answer_service
        self._gemini_api_key = gemini_api_key
        self._model = model
        self._recursion_limit = recursion_limit
        self._lexical_index = lexical_index or getattr(
            getattr(answer_service, "_retriever", None), "_lexical_index", None
        )

    def _get_rag_facade(self) -> LlamaIndexRagTool:
        return LlamaIndexRagTool(
            answer_service=self._answer_service,
            lexical_index=self._lexical_index,
            llm=getattr(self._answer_service, "_llm", None),
        )

    async def ask(self, question: str) -> AgentAnswer:
        try:
            from langchain.agents import create_agent
            from langchain.tools import tool
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ModuleNotFoundError as exc:
            if exc.name not in {"langchain", "langchain_google_genai"}:
                raise
            return await self._call_fallback_route(question)

        rag_facade = self._get_rag_facade()

        @tool("research_documents")
        async def research_documents(tool_question: str) -> str:
            """Query the indexed research corpus for a cited answer."""
            return await rag_facade.query(tool_question)

        @tool("search_corpus")
        async def search_corpus(query: str, limit: int = 5) -> str:
            """Retrieve raw matching passages from indexed documents without LLM synthesis."""
            return await rag_facade.search_corpus(query, limit)

        @tool("read_chunk")
        async def read_chunk(filename: str, page_number: int = 1, chunk_id: str = "") -> str:
            """Read specific passages or full pages from a paper by filename and page_number."""
            return await rag_facade.read_chunk(filename, page_number, chunk_id)

        @tool("summarize_document")
        async def summarize_document(filename: str) -> str:
            """Generate a comprehensive structured summary of a specific research paper."""
            return await rag_facade.summarize_document(filename)

        @tool("compare_documents")
        async def compare_documents(topic: str, paper_a: str = "", paper_b: str = "") -> str:
            """Compare methodologies, models, and findings across multiple research papers."""
            return await rag_facade.compare_documents(topic, paper_a, paper_b)

        @tool("extract_metrics_table")
        async def extract_metrics_table(
            focus: str = "performance metrics, accuracy, precision, recall, F1, FID, dataset size"
        ) -> str:
            """Extract quantitative performance metrics across all papers into a cited Markdown table."""
            return await rag_facade.extract_metrics_table(focus)

        tools = [
            research_documents,
            search_corpus,
            read_chunk,
            summarize_document,
            compare_documents,
            extract_metrics_table,
        ]

        agent = create_agent(
            model=ChatGoogleGenerativeAI(
                model=self._model,
                google_api_key=self._gemini_api_key,
                temperature=0,
            ),
            tools=tools,
            system_prompt=AGENT_SYSTEM_PROMPT,
            name="research_agent",
        )
        response = await agent.ainvoke(
            {"messages": [{"role": "user", "content": question}]},
            config={"recursion_limit": self._recursion_limit},
        )
        tool_payload = self._find_cited_payload(response.get("messages", []))
        return self._parse_agent_response(response, tool_payload)

    async def _call_fallback_route(self, question: str) -> AgentAnswer:
        """Heuristic routing when LangChain runtime is unavailable."""
        rag_facade = self._get_rag_facade()
        lowered = question.lower()
        if any(term in lowered for term in ["table", "metric", "accurac", "fid", "benchmark"]):
            payload = await rag_facade.extract_metrics_table(focus=question)
        elif "summar" in lowered and any(ext in lowered for ext in [".pdf", "paper", "document"]):
            # Extract document name if present
            found = re.search(r"[\w-]+\.pdf", question, re.IGNORECASE)
            doc_name = found.group(0) if found else "hybrid_dl_model.pdf"
            payload = await rag_facade.summarize_document(filename=doc_name)
        elif "compare" in lowered:
            payload = await rag_facade.compare_documents(topic=question)
        else:
            payload = await rag_facade.query(question)
        return self._parse_tool_payload(payload)

    @staticmethod
    def _find_cited_payload(messages: list[Any]) -> str | None:
        """Find the most complete tool response containing answers and citations."""
        for message in reversed(messages):
            if getattr(message, "type", None) == "tool":
                content = getattr(message, "content", None)
                if isinstance(content, str):
                    try:
                        data = json.loads(content)
                        if isinstance(data, dict) and "answer" in data and "citations" in data:
                            return content
                    except Exception:
                        pass
        for message in reversed(messages):
            if getattr(message, "type", None) == "tool":
                content = getattr(message, "content", None)
                if isinstance(content, str):
                    return content
        return None

    def _parse_agent_response(
        self, response: dict[str, Any], tool_payload: str | None
    ) -> AgentAnswer:
        if tool_payload:
            try:
                return self._parse_tool_payload(tool_payload)
            except Exception:
                pass
        messages = response.get("messages", [])
        last_ai = next(
            (
                m
                for m in reversed(messages)
                if getattr(m, "type", None) == "ai" and getattr(m, "content", None)
            ),
            None,
        )
        content = getattr(last_ai, "content", "") if last_ai else (tool_payload or "")
        if isinstance(content, list):
            content = " ".join(
                str(p.get("text", p)) if isinstance(p, dict) else str(p) for p in content
            )
        all_citations: list[Citation] = []
        seen_ids: set[str] = set()
        for m in messages:
            if getattr(m, "type", None) == "tool":
                t_content = getattr(m, "content", None)
                if isinstance(t_content, str):
                    try:
                        data = json.loads(t_content)
                        for item in data.get("citations", []):
                            cid = item.get("id", "")
                            if cid and cid not in seen_ids:
                                seen_ids.add(cid)
                                all_citations.append(
                                    Citation(
                                        citation_id=cid,
                                        chunk_id="",
                                        filename=item.get("filename", ""),
                                        page_number=item.get("page_number", 1),
                                        excerpt=item.get("excerpt", ""),
                                    )
                                )
                    except Exception:
                        pass
        return AgentAnswer(
            answer=str(content),
            citations=all_citations,
            retrieval_candidate_count=len(all_citations),
        )

    @staticmethod
    def _parse_tool_payload(payload: str) -> AgentAnswer:
        try:
            data = json.loads(payload)
            citations = [
                Citation(
                    citation_id=item["id"],
                    chunk_id="",
                    filename=item["filename"],
                    page_number=item["page_number"],
                    excerpt=item["excerpt"],
                )
                for item in data.get("citations", [])
            ]
            return AgentAnswer(
                answer=data.get("answer", payload),
                citations=citations,
                retrieval_candidate_count=data.get("retrieval_candidate_count", len(citations)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("The research tool returned an invalid response.") from exc
