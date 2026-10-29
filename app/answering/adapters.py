import asyncio
import logging
import random
from collections.abc import Sequence

import httpx

from app.core.errors import UpstreamServiceError
from app.retrieval.models import RetrievedChunk

logger = logging.getLogger(__name__)


class CohereReranker:
    """Cohere v2 reranking adapter. Only retrieved candidate text is sent to Cohere."""

    def __init__(self, api_key: str, model: str) -> None:
        self._api_key = api_key
        self._model = model

    async def rerank(
        self, query: str, chunks: Sequence[RetrievedChunk], top_n: int
    ) -> list[RetrievedChunk]:
        if not chunks:
            return []

        max_retries = 5
        results = None

        for attempt in range(max_retries):
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    response = await client.post(
                        "https://api.cohere.com/v2/rerank",
                        headers={"Authorization": f"Bearer {self._api_key}"},
                        json={
                            "model": self._model,
                            "query": query,
                            "documents": [chunk.text for chunk in chunks],
                            "top_n": min(top_n, len(chunks)),
                        },
                    )
                    
                    # Handle Rate Limits (HTTP 429)
                    if response.status_code == 429 and attempt < max_retries - 1:
                        await asyncio.sleep(10 * (attempt + 1))
                        continue

                    response.raise_for_status()
                    results = response.json().get("results", [])
                    break

            except (httpx.ConnectError, httpx.TimeoutException):
                # If network connection fails, fall back to initial un-reranked ordering
                if attempt == max_retries - 1:
                    return list(chunks[:top_n])
                await asyncio.sleep(5 * (attempt + 1))

            except httpx.HTTPError as exc:
                if attempt == max_retries - 1:
                    raise UpstreamServiceError(
                        "cohere", "Cohere reranking service is unavailable."
                    ) from exc
                await asyncio.sleep(5 * (attempt + 1))

        if results is None:
            return list(chunks[:top_n])

        reranked: list[RetrievedChunk] = []
        for result in results:
            index = result.get("index")
            if not isinstance(index, int) or index < 0 or index >= len(chunks):
                raise UpstreamServiceError(
                    "cohere", "Cohere returned an invalid reranking response."
                )
            chunk = chunks[index]
            reranked.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    metadata=chunk.metadata,
                    source=chunk.source,
                    score=float(result.get("relevance_score", 0.0)),
                )
            )
        return reranked


class DeepSeekChatClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        max_output_tokens: int,
        thinking_enabled: bool = True,
        reasoning_effort: str = "low",
    ) -> None:
        self._api_key = api_key
        self._model = model
        self._max_output_tokens = max_output_tokens
        self._thinking_enabled = thinking_enabled
        self._reasoning_effort = reasoning_effort

    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        max_retries = 5
        current_thinking = self._thinking_enabled
        content = ""

        for attempt in range(max_retries):
            try:
                body: dict[str, object] = {
                    "model": self._model,
                    "temperature": 0.1,
                    "max_tokens": self._max_output_tokens,
                    "thinking": {"type": "enabled" if current_thinking else "disabled"},
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                }
                if current_thinking and self._reasoning_effort:
                    body["reasoning_effort"] = self._reasoning_effort

                async with httpx.AsyncClient(timeout=90.0) as client:
                    response = await client.post(
                        "https://api.deepseek.com/chat/completions",
                        headers={"Authorization": f"Bearer {self._api_key}"},
                        json=body,
                    )

                    # Handle DeepSeek's transient rate/capacity responses.
                    if response.status_code == 429:
                        if attempt < max_retries - 1:
                            await asyncio.sleep(15 * (attempt + 1))
                            continue

                    response.raise_for_status()

                payload = response.json()
                choice = payload["choices"][0]
                message = choice["message"]
                content = self._extract_content(message.get("content"))
                if content:
                    break

                finish_reason = choice.get("finish_reason", "unknown")

                # If tokens were exhausted during reasoning (no content generated)
                # or content was empty while thinking was enabled, fallback immediately to non-thinking mode.
                if current_thinking:
                    logger.warning(
                        "DeepSeek returned no answer content (finish_reason=%s) with thinking enabled. "
                        "Falling back to non-thinking mode (thinking disabled).",
                        finish_reason,
                    )
                    current_thinking = False
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1)
                        continue

                if attempt < max_retries - 1:
                    await asyncio.sleep(2 * (attempt + 1))
                    continue

                raise UpstreamServiceError(
                    "deepseek",
                    "DeepSeek returned no answer content "
                    f"(finish_reason={finish_reason}). Increase answer_max_completion_tokens "
                    "or use a non-reasoning chat model.",
                )

            except httpx.HTTPStatusError as exc:
                status_code = exc.response.status_code
                if status_code in {429, 500, 502, 503, 504} and attempt < max_retries - 1:
                    retry_after = exc.response.headers.get("Retry-After")
                    delay = (
                        float(retry_after)
                        if retry_after and retry_after.isdigit()
                        else 2**attempt
                    )
                    await asyncio.sleep(delay + random.uniform(0, 0.5))
                    continue

                if status_code in {429, 500, 502, 503, 504}:
                    raise UpstreamServiceError(
                        "deepseek",
                        "DeepSeek answer-generation service remained unavailable after retries.",
                    ) from exc
                raise UpstreamServiceError(
                    "deepseek",
                    f"DeepSeek rejected the configured model '{self._model}' (HTTP {status_code}). "
                    "Set DEEPSEEK_MODEL to an available model from DeepSeek.",
                ) from exc

            except (httpx.HTTPError, KeyError, IndexError, TypeError, ValueError) as exc:
                if attempt < max_retries - 1:
                    await asyncio.sleep(5 * (attempt + 1))
                    continue

                raise UpstreamServiceError(
                    "deepseek", "DeepSeek answer-generation service is unavailable."
                ) from exc

        return content.strip()

    @staticmethod
    def _extract_content(content: object) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = [
                item.get("text", "")
                for item in content
                if isinstance(item, dict) and isinstance(item.get("text"), str)
            ]
            return "".join(parts).strip()
        return ""
