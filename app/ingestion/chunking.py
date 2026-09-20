from hashlib import sha256

from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter

from app.ingestion.models import ChunkRecord, ExtractedPage


def create_chunks(
    *,
    document_id: str,
    filename: str,
    pages: list[ExtractedPage],
    chunk_size: int,
    chunk_overlap: int,
) -> list[ChunkRecord]:
    """Create citation-safe chunks without allowing a chunk to cross a PDF page boundary."""

    splitter = SentenceSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: list[ChunkRecord] = []
    for page in pages:
        document = Document(text=page.text, metadata={"page_number": page.page_number})
        nodes = splitter.get_nodes_from_documents([document])
        for index, node in enumerate(nodes):
            text = node.get_content().strip()
            if not text:
                continue
            fingerprint = sha256(text.encode("utf-8")).hexdigest()[:16]
            chunk_id = f"{document_id}:{page.page_number}:{index}:{fingerprint}"
            chunks.append(
                ChunkRecord(
                    chunk_id=chunk_id,
                    text=text,
                    metadata={
                        "document_id": document_id,
                        "filename": filename,
                        "page_number": page.page_number,
                        "chunk_index": index,
                        "content_sha256": fingerprint,
                    },
                )
            )
    if not chunks:
        raise ValueError("No chunks could be created from the extracted text.")
    return chunks
