"""Batch-ingest a local folder of research documents.

Run with: python -m app.ingestion.batch data
"""

import argparse
import asyncio
from pathlib import Path

from app.core.config import get_settings
from app.ingestion.adapters import ChromaChunkStore, OllamaEmbeddingClient
from app.ingestion.extractors import SUPPORTED_EXTENSIONS
from app.ingestion.service import IngestionService
from app.retrieval.lexical import SqliteBm25Index
from app.retrieval.store import DualWriteChunkStore


def build_service() -> IngestionService:
    settings = get_settings()
    return IngestionService(
        embedding_client=OllamaEmbeddingClient(
            base_url=settings.ollama_base_url,
            model=settings.ollama_embedding_model,
        ),
        chunk_store=DualWriteChunkStore(
            vector_store=ChromaChunkStore(
                persist_directory=settings.chroma_persist_directory,
                collection_name=settings.chroma_collection_name,
            ),
            lexical_index=SqliteBm25Index(settings.lexical_index_path),
        ),
        max_file_size_bytes=settings.ingestion_max_file_size_bytes,
        chunk_size=settings.ingestion_chunk_size,
        chunk_overlap=settings.ingestion_chunk_overlap,
    )


async def ingest_folder(folder: Path) -> None:
    service = build_service()
    files = [
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if not files:
        raise SystemExit("No supported documents found.")
    for path in files:
        result = await service.ingest(path.name, path.read_bytes())
        print(f"Ingested {result.filename}: {result.chunk_count} chunks ({result.document_id})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest a folder of research documents.")
    parser.add_argument(
        "folder", type=Path, help="Folder containing PDF, TXT, or Markdown documents"
    )
    args = parser.parse_args()
    if not args.folder.is_dir():
        raise SystemExit(f"Folder does not exist: {args.folder}")
    asyncio.run(ingest_folder(args.folder))


if __name__ == "__main__":
    main()
