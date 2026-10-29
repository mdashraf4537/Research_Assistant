import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import get_settings


def _get_db_path() -> Path:
    settings = get_settings()
    db_path = Path(settings.lexical_index_path).parent / "user_tracking.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return db_path


def init_tracking_db() -> None:
    db_path = _get_db_path()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                email TEXT PRIMARY KEY,
                first_seen_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                query_count INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL,
                query_text TEXT NOT NULL,
                mode TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS user_documents (
                email TEXT NOT NULL,
                filename TEXT NOT NULL,
                chunk_count INTEGER DEFAULT 0,
                added_at TEXT NOT NULL,
                PRIMARY KEY (email, filename)
            )
            """
        )
        conn.commit()


def record_user_document(email: str, filename: str, chunk_count: int = 0) -> None:
    cleaned_email = email.strip().lower()
    if not cleaned_email or not filename:
        return
    init_tracking_db()
    now_iso = datetime.now(timezone.utc).isoformat()
    db_path = _get_db_path()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO user_documents (email, filename, chunk_count, added_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(email, filename) DO UPDATE SET
                chunk_count = excluded.chunk_count,
                added_at = excluded.added_at
            """,
            (cleaned_email, filename, chunk_count, now_iso),
        )
        conn.commit()


def get_user_documents(email: str) -> list[dict[str, Any]]:
    cleaned_email = email.strip().lower()
    if not cleaned_email:
        return []
    init_tracking_db()
    db_path = _get_db_path()
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT filename, chunk_count, added_at FROM user_documents WHERE email = ? ORDER BY added_at DESC",
            (cleaned_email,),
        )
        return [dict(row) for row in cursor.fetchall()]


def delete_user_document(email: str, filename: str) -> None:
    cleaned_email = email.strip().lower()
    if not cleaned_email or not filename:
        return
    init_tracking_db()
    db_path = _get_db_path()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "DELETE FROM user_documents WHERE email = ? AND filename = ?",
            (cleaned_email, filename),
        )
        conn.commit()


def is_document_owned_by_other_users(filename: str, current_email: str) -> bool:
    cleaned_email = current_email.strip().lower()
    init_tracking_db()
    db_path = _get_db_path()
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM user_documents WHERE filename = ? AND email != ?",
            (filename, cleaned_email),
        )
        count = cursor.fetchone()[0]
        return count > 0


def identify_or_register_user(email: str) -> dict[str, Any]:
    cleaned_email = email.strip().lower()
    if not cleaned_email or "@" not in cleaned_email or "." not in cleaned_email:
        raise ValueError("Invalid email format.")

    init_tracking_db()
    now_iso = datetime.now(timezone.utc).isoformat()
    db_path = _get_db_path()

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email = ?", (cleaned_email,))
        row = cursor.fetchone()

        if row is None:
            cursor.execute(
                "INSERT INTO users (email, first_seen_at, last_seen_at, query_count) VALUES (?, ?, ?, 0)",
                (cleaned_email, now_iso, now_iso),
            )
            conn.commit()
            return {
                "email": cleaned_email,
                "first_seen_at": now_iso,
                "last_seen_at": now_iso,
                "query_count": 0,
                "is_new": True,
            }
        else:
            cursor.execute(
                "UPDATE users SET last_seen_at = ? WHERE email = ?",
                (now_iso, cleaned_email),
            )
            conn.commit()
            return {
                "email": cleaned_email,
                "first_seen_at": row["first_seen_at"],
                "last_seen_at": now_iso,
                "query_count": row["query_count"],
                "is_new": False,
            }


def log_user_query(email: str, query_text: str, mode: str) -> None:
    cleaned_email = email.strip().lower()
    if not cleaned_email:
        return

    init_tracking_db()
    now_iso = datetime.now(timezone.utc).isoformat()
    db_path = _get_db_path()

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO queries (email, query_text, mode, created_at) VALUES (?, ?, ?, ?)",
            (cleaned_email, query_text.strip(), mode, now_iso),
        )
        conn.execute(
            """
            INSERT INTO users (email, first_seen_at, last_seen_at, query_count)
            VALUES (?, ?, ?, 1)
            ON CONFLICT(email) DO UPDATE SET
                last_seen_at = excluded.last_seen_at,
                query_count = query_count + 1
            """,
            (cleaned_email, now_iso, now_iso),
        )
        conn.commit()


def get_tracked_users() -> list[dict[str, Any]]:
    init_tracking_db()
    db_path = _get_db_path()
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT email, first_seen_at, last_seen_at, query_count FROM users ORDER BY last_seen_at DESC"
        )
        return [dict(row) for row in cursor.fetchall()]


def get_recent_queries(limit: int = 100) -> list[dict[str, Any]]:
    init_tracking_db()
    db_path = _get_db_path()
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute(
            "SELECT id, email, query_text, mode, created_at FROM queries ORDER BY id DESC LIMIT ?",
            (limit,),
        )
        return [dict(row) for row in cursor.fetchall()]
