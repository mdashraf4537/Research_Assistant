import json
from dataclasses import dataclass
from typing import Any

from app.agent.rag_tool import LlamaIndexRagTool
from app.answering.models import Citation
from app.answering.service import ResearchAnswerService

AGENT_SYSTEM_PROMPT = """You are the coordinator for a citation-first research assistant.
For every research question, call research_documents exactly once. Do not use your own
knowledge, invent sources, call unavailable tools, write files, browse the web, or take
external actions. After the tool returns, preserve its answer and citations exactly."""


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
    ) -> None:
        self._answer_service = answer_service
        self._gemini_api_key = gemini_api_key
        self._model = model
        self._recursion_limit = recursion_limit

    async def ask(self, question: str) -> AgentAnswer:
        try:
            from langchain.agents import create_agent
            from langchain.tools import tool
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ModuleNotFoundError as exc:
            if exc.name not in {"langchain", "langchain_google_genai"}:
                raise
            # The controlled fallback retains the agent's safety boundary: its
            # only action is one invocation of the LlamaIndex RAG tool.  It
            # keeps Agent mode usable in minimal runtime environments; the
            # normal LangChain orchestration remains the preferred path.
            return await self._call_research_tool(question)

        llama_tool = LlamaIndexRagTool(self._answer_service).as_llama_index_tool()

        @tool("research_documents")
        async def research_documents(tool_question: str) -> str:
            """Query the indexed research corpus for a cited answer."""
            result = await llama_tool.acall(question=tool_question)
            return str(getattr(result, "content", result))

        agent = create_agent(
            model=ChatGoogleGenerativeAI(
                model=self._model,
                google_api_key=self._gemini_api_key,
                temperature=0,
            ),
            tools=[research_documents],
            system_prompt=AGENT_SYSTEM_PROMPT,
            name="research_agent",
        )
        response = await agent.ainvoke(
            {"messages": [{"role": "user", "content": question}]},
            config={"recursion_limit": self._recursion_limit},
        )
        tool_payload = self._last_tool_payload(response.get("messages", []))
        if tool_payload is None:
            raise RuntimeError("The agent did not invoke the research tool.")
        return self._parse_tool_payload(tool_payload)

    async def _call_research_tool(self, question: str) -> AgentAnswer:
        llama_tool = LlamaIndexRagTool(self._answer_service).as_llama_index_tool()
        result = await llama_tool.acall(question=question)
        return self._parse_tool_payload(str(getattr(result, "content", result)))

    @staticmethod
    def _last_tool_payload(messages: list[Any]) -> str | None:
        for message in reversed(messages):
            if getattr(message, "type", None) == "tool":
                content = getattr(message, "content", None)
                if isinstance(content, str):
                    return content
        return None

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
                for item in data["citations"]
            ]
            return AgentAnswer(
                answer=data["answer"],
                citations=citations,
                retrieval_candidate_count=data["retrieval_candidate_count"],
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("The research tool returned an invalid response.") from exc
