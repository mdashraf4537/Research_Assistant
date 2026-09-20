from app.retrieval.models import RetrievedChunk
from app.retrieval.service import HybridRetriever


def _hit(chunk_id: str, source: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=f"text for {chunk_id}",
        metadata={"filename": "paper.pdf", "page_number": 1},
        source=source,
    )


def test_rrf_rewards_chunks_found_by_both_retrievers() -> None:
    retriever = HybridRetriever(
        None, None, None, candidate_count=5, final_count=5, rrf_k=60  # type: ignore[arg-type]
    )

    results = retriever._reciprocal_rank_fusion(
        (
            [_hit("lexical-only", "bm25"), _hit("shared", "bm25")],
            [_hit("shared", "semantic")],
        )
    )

    assert results[0].chunk_id == "shared"
    assert results[0].source == "bm25+semantic"
