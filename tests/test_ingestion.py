from collections.abc import Sequence

from fastapi.testclient import TestClient

from app.api.routes import get_ingestion_service
from app.ingestion.models import ChunkRecord
from app.ingestion.service import IngestionService
from app.main import create_app


class FakeEmbeddingClient:
    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[float(index), 1.0] for index, _ in enumerate(texts)]


class FakeChunkStore:
    def __init__(self) -> None:
        self.chunks: list[ChunkRecord] = []
        self.embeddings: list[list[float]] = []

    def upsert(self, chunks: Sequence[ChunkRecord], embeddings: Sequence[Sequence[float]]) -> None:
        self.chunks = list(chunks)
        self.embeddings = [list(vector) for vector in embeddings]


def test_ingest_text_stores_page_aware_chunks() -> None:
    store = FakeChunkStore()
    service = IngestionService(FakeEmbeddingClient(), store, 1024 * 1024, 64, 16)
    app = create_app()
    app.dependency_overrides[get_ingestion_service] = lambda: service

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/documents/ingest",
            files={"file": ("paper.txt", b"Retrieval augmented generation improves answers. " * 8)},
        )

    assert response.status_code == 201
    assert response.json()["chunk_count"] >= 1
    assert store.chunks[0].metadata["page_number"] == 1
    assert len(store.chunks) == len(store.embeddings)


def test_ingest_rejects_unsupported_files() -> None:
    service = IngestionService(FakeEmbeddingClient(), FakeChunkStore(), 1024 * 1024, 64, 16)
    app = create_app()
    app.dependency_overrides[get_ingestion_service] = lambda: service

    with TestClient(app) as client:
        response = client.post(
            "/api/v1/documents/ingest",
            files={"file": ("paper.exe", b"not a paper")},
        )

    assert response.status_code == 422
