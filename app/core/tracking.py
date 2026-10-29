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
                query_count INTEGER NOT NULL DEFAULT 0,
                query_limit INTEGER NOT NULL DEFAULT 2,
                approval_status TEXT NOT NULL DEFAULT 'none',
                approval_requested_at TEXT
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
        
        # Safe migration for existing databases missing quota/approval columns
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(users)")
        existing_cols = {row[1] for row in cursor.fetchall()}
        if "query_limit" not in existing_cols:
            conn.execute("ALTER TABLE users ADD COLUMN query_limit INTEGER NOT NULL DEFAULT 2")
        if "approval_status" not in existing_cols:
            conn.execute("ALTER TABLE users ADD COLUMN approval_status TEXT NOT NULL DEFAULT 'none'")
        if "approval_requested_at" not in existing_cols:
            conn.execute("ALTER TABLE users ADD COLUMN approval_requested_at TEXT")
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
    settings = get_settings()
    is_admin = cleaned_email == settings.admin_email.strip().lower()
    default_limit = 999999 if is_admin else settings.default_query_limit
    default_status = "approved" if is_admin else "none"

    now_iso = datetime.now(timezone.utc).isoformat()
    db_path = _get_db_path()

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email = ?", (cleaned_email,))
        row = cursor.fetchone()

        if row is None:
            cursor.execute(
                """
                INSERT INTO users (email, first_seen_at, last_seen_at, query_count, query_limit, approval_status)
                VALUES (?, ?, ?, 0, ?, ?)
                """,
                (cleaned_email, now_iso, now_iso, default_limit, default_status),
            )
            conn.commit()
            return {
                "email": cleaned_email,
                "first_seen_at": now_iso,
                "last_seen_at": now_iso,
                "query_count": 0,
                "query_limit": default_limit,
                "approval_status": default_status,
                "is_new": True,
                "is_admin": is_admin,
            }
        else:
            # If admin, ensure unlimited limit
            if is_admin and row["query_limit"] < 999999:
                cursor.execute(
                    "UPDATE users SET last_seen_at = ?, query_limit = 999999, approval_status = 'approved' WHERE email = ?",
                    (now_iso, cleaned_email),
                )
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
                "query_limit": 999999 if is_admin else row["query_limit"],
                "approval_status": "approved" if is_admin else row["approval_status"],
                "approval_requested_at": row["approval_requested_at"],
                "is_new": False,
                "is_admin": is_admin,
            }


def get_user_quota_info(email: str, admin_email: str) -> dict[str, Any]:
    cleaned_email = email.strip().lower()
    is_admin = cleaned_email == admin_email.strip().lower()
    if is_admin:
        return {
            "email": cleaned_email,
            "is_admin": True,
            "query_count": 0,
            "query_limit": 999999,
            "approval_status": "approved",
            "approval_requested_at": None,
            "can_query": True,
        }

    init_tracking_db()
    db_path = _get_db_path()
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email = ?", (cleaned_email,))
        row = cursor.fetchone()

        if row is None:
            settings = get_settings()
            return {
                "email": cleaned_email,
                "is_admin": False,
                "query_count": 0,
                "query_limit": settings.default_query_limit,
                "approval_status": "none",
                "approval_requested_at": None,
                "can_query": True,
            }

        q_count = int(row["query_count"])
        q_limit = int(row["query_limit"])
        can_query = q_count < q_limit

        return {
            "email": cleaned_email,
            "is_admin": False,
            "query_count": q_count,
            "query_limit": q_limit,
            "approval_status": row["approval_status"] or "none",
            "approval_requested_at": row["approval_requested_at"],
            "can_query": can_query,
        }


def check_user_query_quota(email: str, admin_email: str) -> tuple[bool, str, dict[str, Any]]:
    info = get_user_quota_info(email, admin_email)
    if info["can_query"]:
        return True, "Quota within allowed limit", info
    return False, f"Query quota exceeded ({info['query_count']}/{info['query_limit']})", info


def request_user_access(email: str) -> dict[str, Any]:
    cleaned_email = email.strip().lower()
    if not cleaned_email:
        raise ValueError("Email required")
    init_tracking_db()
    now_iso = datetime.now(timezone.utc).isoformat()
    db_path = _get_db_path()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            UPDATE users SET approval_status = 'pending', approval_requested_at = ?
            WHERE email = ?
            """,
            (now_iso, cleaned_email),
        )
        conn.commit()
    settings = get_settings()
    return get_user_quota_info(cleaned_email, settings.admin_email)


def approve_user_access(email: str, additional_queries: int = 10, unlimited: bool = False) -> dict[str, Any]:
    cleaned_email = email.strip().lower()
    if not cleaned_email:
        raise ValueError("Email required")
    init_tracking_db()
    db_path = _get_db_path()
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email = ?", (cleaned_email,))
        row = cursor.fetchone()
        if not row:
            raise ValueError(f"User {cleaned_email} not found")

        current_count = int(row["query_count"])
        current_limit = int(row["query_limit"])

        if unlimited:
            new_limit = 999999
        else:
            new_limit = max(current_count, current_limit) + additional_queries

        conn.execute(
            """
            UPDATE users SET query_limit = ?, approval_status = 'approved'
            WHERE email = ?
            """,
            (new_limit, cleaned_email),
        )
        conn.commit()
    settings = get_settings()
    return get_user_quota_info(cleaned_email, settings.admin_email)


def reject_user_access(email: str) -> dict[str, Any]:
    cleaned_email = email.strip().lower()
    if not cleaned_email:
        raise ValueError("Email required")
    init_tracking_db()
    db_path = _get_db_path()
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE users SET approval_status = 'rejected' WHERE email = ?",
            (cleaned_email,),
        )
        conn.commit()
    settings = get_settings()
    return get_user_quota_info(cleaned_email, settings.admin_email)


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
            """
            SELECT email, first_seen_at, last_seen_at, query_count, query_limit, approval_status, approval_requested_at
            FROM users ORDER BY last_seen_at DESC
            """
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
