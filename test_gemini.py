"""Minimal Gemini connectivity check; reads GEMINI_API_KEY from the environment."""

import os

import httpx


response = httpx.post(
    "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent",
    headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]},
    json={"contents": [{"parts": [{"text": "Say hi."}]}]},
    timeout=15,
)
print("GEMINI:", response.status_code, response.text[:200])
