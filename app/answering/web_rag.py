"""Scientific Literature & Web Search RAG Service.

When a researcher asks a question that is not covered by the locally
indexed document library, this service retrieves live research passages
from arXiv Scientific Papers, Web Search (DuckDuckGo), Google RSS, and
Academic Knowledge APIs, constructs numbered citations [S1]..[Sn],
and synthesizes an evidence-grounded answer.
"""

import asyncio
import html
import re
import urllib.parse
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import httpx

from app.answering.models import AnswerResult, Citation
from app.answering.protocols import LanguageModel

WEB_RAG_SYSTEM_PROMPT = """You are an evidence-based academic and scientific research assistant.
The user's research query was not fully covered in the local document repository, so live scientific literature and web research sources have been retrieved below.
Synthesize a clear, rigorous, and comprehensive answer using the supplied research evidence.
Cite every factual claim using the supplied bracketed citation IDs, such as [S1], [S2].
Preserve exact quantitative metrics, percentages, benchmarks, equations, and technical terms.
Do not invent citations outside the supplied [S#] list."""


@dataclass(frozen=True)
class WebSnippet:
    title: str
    url: str
    snippet: str
    source_label: str


class WebSearchRagService:
    """Retrieves live academic literature and web passages for RAG synthesis."""

    def __init__(
        self,
        llm: LanguageModel,
        gemini_api_key: str | None = None,
        gemini_model: str = "gemini-2.5-flash",
        max_results: int = 6,
    ) -> None:
        self._llm = llm
        self._gemini_api_key = gemini_api_key
        self._gemini_model = gemini_model
        self._max_results = max_results

    async def answer(self, question: str) -> AnswerResult:
        search_query = self._extract_clean_search_query(question)

        # Query academic and web search endpoints concurrently
        tasks = [
            self._search_arxiv(search_query),
            self._search_duckduckgo(search_query),
            self._search_google_rss(search_query),
            self._search_wikipedia(search_query),
        ]
        results_lists = await asyncio.gather(*tasks, return_exceptions=True)

        snippets: list[WebSnippet] = []
        seen_urls: set[str] = set()
        for item in results_lists:
            if isinstance(item, list):
                for s in item:
                    norm_url = s.url.strip().lower()
                    if s.snippet and norm_url not in seen_urls:
                        seen_urls.add(norm_url)
                        snippets.append(s)

        # Fallback to Gemini Google Search grounding if external requests yielded empty snippets
        if not snippets and self._gemini_api_key:
            grounded = await self._try_gemini_google_grounding(question)
            if grounded is not None:
                return grounded

        top_snippets = snippets[: self._max_results]
        if not top_snippets:
            return AnswerResult(
                answer="Could not find sufficient evidence in the local documents or through live scientific search.",
                citations=[],
                candidate_count=0,
                source_mode="google_search_rag",
            )

        citations = [
            Citation(
                citation_id=f"S{idx}",
                chunk_id=f"web-{idx}",
                filename=f"{s.source_label}: {s.title[:85]}",
                page_number=1,
                excerpt=f"{s.snippet}\nSource URL: {s.url}",
            )
            for idx, s in enumerate(top_snippets, start=1)
        ]

        evidence = "\n\n".join(
            f"[{c.citation_id}] {c.filename}\n{c.excerpt}"
            for c in citations
        )

        answer = await self._llm.complete(
            WEB_RAG_SYSTEM_PROMPT,
            f"Research Question:\n{question}\n\nRetrieved Scientific & Web Evidence:\n{evidence}",
        )
        allowed_ids = {c.citation_id for c in citations}
        answer = re.sub(
            r"\[S(\d+)\]",
            lambda m: m.group(0) if f"S{m.group(1)}" in allowed_ids else "",
            answer,
        )
        return AnswerResult(
            answer=answer,
            citations=citations,
            candidate_count=len(snippets),
            source_mode="google_search_rag",
        )

    @staticmethod
    def _extract_clean_search_query(question: str) -> str:
        """Strip conversational wrappers to produce a clean keyword retrieval query."""
        q = question.strip()
        if "Question:" in q:
            q = q.split("Question:")[-1].strip()
        q = re.sub(r"\s+", " ", q)
        return q[:250]

    async def _search_arxiv(self, query: str) -> list[WebSnippet]:
        """Search arXiv API for peer-reviewed academic preprints and papers."""
        try:
            clean_term = re.sub(r"[^a-zA-Z0-9\s]", " ", query).strip()
            clean_term = " ".join(clean_term.split()[:8])
            async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
                resp = await client.get(
                    "http://export.arxiv.org/api/query",
                    params={
                        "search_query": f"all:{clean_term}",
                        "start": 0,
                        "max_results": 3,
                    },
                )
                if resp.status_code != 200:
                    return []
            root = ET.fromstring(resp.text)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            snippets: list[WebSnippet] = []
            for entry in root.findall("atom:entry", ns):
                title = (entry.findtext("atom:title", namespaces=ns) or "").strip()
                title = re.sub(r"\s+", " ", title)
                summary = (entry.findtext("atom:summary", namespaces=ns) or "").strip()
                summary = re.sub(r"\s+", " ", summary)
                link_el = entry.find("atom:id", ns)
                link = (link_el.text or "").strip() if link_el is not None else ""
                if title and summary:
                    snippets.append(
                        WebSnippet(
                            title=title,
                            url=link or "https://arxiv.org",
                            snippet=summary[:800],
                            source_label="arXiv Scientific Paper",
                        )
                    )
            return snippets
        except Exception:
            return []

    async def _search_duckduckgo(self, query: str) -> list[WebSnippet]:
        """Search Web via DuckDuckGo HTML endpoint."""
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
            )
        }
        try:
            async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
                resp = await client.post(
                    "https://html.duckduckgo.com/html/",
                    data={"q": query},
                    headers=headers,
                )
                if resp.status_code != 200:
                    return []
                body = resp.text

            snippets: list[WebSnippet] = []
            blocks = re.findall(
                r'<div class="result__body">(.*?)</div>\s*</div>',
                body,
                flags=re.DOTALL,
            )
            for block in blocks[:5]:
                title_match = re.search(
                    r'class="result__a"[^>]* href="([^"]+)"[^>]*>(.*?)</a>',
                    block,
                    flags=re.DOTALL,
                )
                snip_match = re.search(
                    r'class="result__snippet"[^>]*>(.*?)</a>|class="result__snippet"[^>]*>(.*?)</div>',
                    block,
                    flags=re.DOTALL,
                )
                if not title_match:
                    continue
                raw_url = html.unescape(title_match.group(1))
                if "uddg=" in raw_url:
                    parsed_qs = urllib.parse.parse_qs(urllib.parse.urlparse(raw_url).query)
                    raw_url = parsed_qs.get("uddg", [raw_url])[0]
                title = self._strip_tags(title_match.group(2))
                raw_snip = (
                    (snip_match.group(1) or snip_match.group(2) or "")
                    if snip_match
                    else ""
                )
                snippet = self._strip_tags(raw_snip)
                if title and snippet:
                    domain = urllib.parse.urlparse(raw_url).netloc or "Web"
                    snippets.append(
                        WebSnippet(
                            title=title,
                            url=raw_url,
                            snippet=snippet,
                            source_label=f"Web ({domain})",
                        )
                    )
            return snippets
        except Exception:
            return []

    async def _search_google_rss(self, query: str) -> list[WebSnippet]:
        """Search Google Search / News RSS feed for recent publications and articles."""
        try:
            async with httpx.AsyncClient(timeout=8.0, follow_redirects=True) as client:
                resp = await client.get(
                    "https://news.google.com/rss/search",
                    params={"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"},
                )
                if resp.status_code != 200:
                    return []
            root = ET.fromstring(resp.text)
            snippets: list[WebSnippet] = []
            for item in root.findall(".//item")[:3]:
                title = (item.findtext("title") or "").strip()
                link = (item.findtext("link") or "").strip()
                desc = self._strip_tags(item.findtext("description") or "")
                source = (item.findtext("source") or "Search").strip()
                if title:
                    snippets.append(
                        WebSnippet(
                            title=title,
                            url=link,
                            snippet=desc if len(desc) > 25 else title,
                            source_label=f"Source ({source})",
                        )
                    )
            return snippets
        except Exception:
            return []

    async def _search_wikipedia(self, query: str) -> list[WebSnippet]:
        """Search Wikipedia API for scientific definitions and background context."""
        try:
            headers = {"User-Agent": "EnterpriseResearchAssistant/1.0"}
            async with httpx.AsyncClient(timeout=8.0, headers=headers) as client:
                resp = await client.get(
                    "https://en.wikipedia.org/w/api.php",
                    params={
                        "action": "query",
                        "list": "search",
                        "srsearch": query,
                        "srlimit": 2,
                        "format": "json",
                    },
                )
                if resp.status_code != 200:
                    return []
                items = resp.json().get("query", {}).get("search", [])
                snippets: list[WebSnippet] = []
                for item in items:
                    title = item.get("title", "")
                    snip = self._strip_tags(item.get("snippet", ""))
                    if title and snip:
                        url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"
                        snippets.append(
                            WebSnippet(
                                title=title,
                                url=url,
                                snippet=snip,
                                source_label="Encyclopedia",
                            )
                        )
                return snippets
        except Exception:
            return []

    async def _try_gemini_google_grounding(self, question: str) -> AnswerResult | None:
        """Native Google Search tool grounding fallback."""
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self._gemini_api_key)
            resp = await asyncio.to_thread(
                client.models.generate_content,
                model=self._gemini_model,
                contents=question,
                config=types.GenerateContentConfig(
                    tools=[types.Tool(google_search=types.GoogleSearch())],
                    temperature=0.1,
                ),
            )
            text = (resp.text or "").strip()
            if not text:
                return None
            citations: list[Citation] = []
            candidates = getattr(resp, "candidates", None) or []
            if candidates:
                gm = getattr(candidates[0], "grounding_metadata", None)
                chunks = getattr(gm, "grounding_chunks", None) or []
                for idx, ch in enumerate(chunks[:6], start=1):
                    web = getattr(ch, "web", None)
                    if web:
                        uri = getattr(web, "uri", "") or ""
                        title = getattr(web, "title", "") or "Google Search Reference"
                        citations.append(
                            Citation(
                                citation_id=f"S{idx}",
                                chunk_id=f"gsearch-{idx}",
                                filename=f"Google Search: {title}",
                                page_number=1,
                                excerpt=f"Grounded Source: {title}\nURL: {uri}",
                            )
                        )
            return AnswerResult(
                answer=text,
                citations=citations,
                candidate_count=len(citations),
                source_mode="google_search_rag",
            )
        except Exception:
            return None

    @staticmethod
    def _strip_tags(raw_html: str) -> str:
        cleaned = re.sub(r"<[^>]+>", " ", raw_html)
        cleaned = html.unescape(cleaned)
        return re.sub(r"\s+", " ", cleaned).strip()
