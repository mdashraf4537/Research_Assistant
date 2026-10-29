import json
import re
import sqlite3
from collections.abc import Sequence
from pathlib import Path

from app.ingestion.models import ChunkRecord
from app.retrieval.models import RetrievedChunk


class SqliteBm25Index:
    """Persistent BM25 index backed by SQLite FTS5.

    SQLite's bm25() returns lower values for stronger matches; callers consume rank,
    so raw scores never need to be compared with vector-space distances.
    """

    def __init__(self, database_path: str) -> None:
        path = Path(database_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._database_path = str(path)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._database_path)

    def _initialize(self) -> None:
        connection = self._connect()
        try:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS chunk_metadata (
                    chunk_id TEXT PRIMARY KEY,
                    metadata_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS chunk_fts
                USING fts5(chunk_id UNINDEXED, text, tokenize='unicode61')
                """
            )
            connection.commit()
        finally:
            connection.close()

    def upsert(self, chunks: Sequence[ChunkRecord]) -> None:
        connection = self._connect()
        try:
            for chunk in chunks:
                connection.execute("DELETE FROM chunk_fts WHERE chunk_id = ?", (chunk.chunk_id,))
                connection.execute(
                    "INSERT INTO chunk_fts(chunk_id, text) VALUES (?, ?)",
                    (chunk.chunk_id, chunk.text),
                )
                connection.execute(
                    """
                    INSERT INTO chunk_metadata(chunk_id, metadata_json) VALUES (?, ?)
                    ON CONFLICT(chunk_id) DO UPDATE SET metadata_json = excluded.metadata_json
                    """,
                    (chunk.chunk_id, json.dumps(chunk.metadata, sort_keys=True)),
                )
            connection.commit()
        finally:
            connection.close()

    def search(self, query: str, limit: int) -> list[RetrievedChunk]:
        terms = re.findall(r"[\w-]+", query.lower())
        if not terms:
            return []
        fts_query = " OR ".join(f'"{term}"' for term in terms)
        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT chunk_fts.chunk_id, chunk_fts.text, chunk_metadata.metadata_json,
                       bm25(chunk_fts) AS bm25_score
                FROM chunk_fts
                JOIN chunk_metadata ON chunk_metadata.chunk_id = chunk_fts.chunk_id
                WHERE chunk_fts MATCH ?
                ORDER BY bm25_score
                LIMIT ?
                """,
                (fts_query, limit),
            ).fetchall()
        finally:
            connection.close()
        return [
            RetrievedChunk(
                chunk_id=chunk_id,
                text=text,
                metadata=json.loads(metadata_json),
                source="bm25",
                score=float(score),
            )
            for chunk_id, text, metadata_json, score in rows
        ]

    def get_chunk_by_id(self, chunk_id: str) -> RetrievedChunk | None:
        connection = self._connect()
        try:
            row = connection.execute(
                """
                SELECT chunk_fts.chunk_id, chunk_fts.text, chunk_metadata.metadata_json
                FROM chunk_fts
                JOIN chunk_metadata ON chunk_metadata.chunk_id = chunk_fts.chunk_id
                WHERE chunk_fts.chunk_id = ?
                """,
                (chunk_id,),
            ).fetchone()
        finally:
            connection.close()
        if not row:
            return None
        c_id, text, metadata_json = row
        return RetrievedChunk(
            chunk_id=c_id,
            text=text,
            metadata=json.loads(metadata_json),
            source="bm25",
            score=1.0,
        )

    def get_chunks_by_document(
        self, filename: str, page_number: int | None = None, limit: int = 50
    ) -> list[RetrievedChunk]:
        connection = self._connect()
        try:
            if page_number is not None:
                rows = connection.execute(
                    """
                    SELECT chunk_fts.chunk_id, chunk_fts.text, chunk_metadata.metadata_json
                    FROM chunk_fts
                    JOIN chunk_metadata ON chunk_metadata.chunk_id = chunk_fts.chunk_id
                    WHERE json_extract(chunk_metadata.metadata_json, '$.filename') = ?
                      AND json_extract(chunk_metadata.metadata_json, '$.page_number') = ?
                    ORDER BY CAST(json_extract(chunk_metadata.metadata_json, '$.chunk_index') AS INTEGER)
                    LIMIT ?
                    """,
                    (filename, page_number, limit),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT chunk_fts.chunk_id, chunk_fts.text, chunk_metadata.metadata_json
                    FROM chunk_fts
                    JOIN chunk_metadata ON chunk_metadata.chunk_id = chunk_fts.chunk_id
                    WHERE json_extract(chunk_metadata.metadata_json, '$.filename') = ?
                    ORDER BY CAST(json_extract(chunk_metadata.metadata_json, '$.chunk_index') AS INTEGER)
                    LIMIT ?
                    """,
                    (filename, limit),
                ).fetchall()
        finally:
            connection.close()
        return [
            RetrievedChunk(
                chunk_id=c_id,
                text=text,
                metadata=json.loads(meta),
                source="bm25",
                score=1.0,
            )
            for c_id, text, meta in rows
        ]

    def list_indexed_documents(self) -> list[dict[str, object]]:
        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT json_extract(metadata_json, '$.filename') AS filename,
                       COUNT(*) AS chunk_count,
                       MAX(CAST(json_extract(metadata_json, '$.page_number') AS INTEGER)) AS page_count
                FROM chunk_metadata
                GROUP BY 1
                ORDER BY 1
                """
            ).fetchall()
        finally:
            connection.close()
        return [
            {"filename": str(filename), "chunk_count": int(chunk_count), "page_count": int(page_count or 1)}
            for filename, chunk_count, page_count in rows
            if filename is not None
        ]

    def delete_document(self, filename: str) -> int:
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT chunk_id FROM chunk_metadata WHERE json_extract(metadata_json, '$.filename') = ?",
                (filename,),
            ).fetchall()
            chunk_ids = [r[0] for r in rows]
            for c_id in chunk_ids:
                connection.execute("DELETE FROM chunk_fts WHERE chunk_id = ?", (c_id,))
                connection.execute("DELETE FROM chunk_metadata WHERE chunk_id = ?", (c_id,))
            connection.commit()
            return len(chunk_ids)
        finally:
            connection.close()

