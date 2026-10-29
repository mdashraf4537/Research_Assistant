from dataclasses import dataclass


@dataclass(frozen=True)
class Citation:
    citation_id: str
    chunk_id: str
    filename: str
    page_number: int
    excerpt: str


@dataclass(frozen=True)
class AnswerResult:
    answer: str
    citations: list[Citation]
    candidate_count: int
    source_mode: str = "knowledge_base"
