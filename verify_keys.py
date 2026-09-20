"""Verify configured Cohere and Gemini credentials without storing secrets in source."""

import asyncio
import os

import httpx


async def main() -> None:
    async with httpx.AsyncClient(timeout=15.0) as client:
        cohere = await client.post(
            "https://api.cohere.com/v2/rerank",
            headers={"Authorization": f"Bearer {os.environ['COHERE_API_KEY']}"},
            json={"model": "rerank-v4.0-pro", "query": "test", "documents": ["hello"], "top_n": 1},
        )
        print("COHERE STATUS:", cohere.status_code)
        gemini = await client.post(
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
            headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]},
            json={"contents": [{"parts": [{"text": "Say hi."}]}]},
        )
        print("GEMINI STATUS:", gemini.status_code)


asyncio.run(main())
