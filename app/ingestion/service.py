from hashlib import sha256

from app.ingestion.chunking import create_chunks
from app.ingestion.extractors import IngestionValidationError, extract_pages
from app.ingestion.models import IngestionResult
from app.ingestion.protocols import ChunkStore, EmbeddingClient


class IngestionService:
    def __init__(
        self,
        embedding_client: EmbeddingClient,
        chunk_store: ChunkStore,
        max_file_size_bytes: int,
        chunk_size: int,
        chunk_overlap: int,
    ) -> None:
        self._embedding_client = embedding_client
        self._chunk_store = chunk_store
        self._max_file_size_bytes = max_file_size_bytes
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    async def ingest(self, filename: str, content: bytes) -> IngestionResult:
        if not filename:
            raise IngestionValidationError("A filename is required.")
        if not content:
            raise IngestionValidationError("The uploaded file is empty.")
        if len(content) > self._max_file_size_bytes:
            raise IngestionValidationError("The uploaded file exceeds the configured size limit.")

        content_sha256 = sha256(content).hexdigest()
        document_id = content_sha256[:32]
        pages = extract_pages(filename, content)
        chunks = create_chunks(
            document_id=document_id,
            filename=filename,
            pages=pages,
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )
        embeddings = await self._embedding_client.embed([chunk.text for chunk in chunks])
        self._chunk_store.upsert(chunks, embeddings)
        return IngestionResult(
            document_id=document_id,
            filename=filename,
            page_count=len(pages),
            chunk_count=len(chunks),
            content_sha256=content_sha256,
        )
