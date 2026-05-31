"""
db.py — SQLite database layer for the Learning Path Generator.

Designed for easy migration to PostgreSQL (uses standard SQL syntax only, no
SQLite-specific extensions).  All public functions are synchronous so they work
with Streamlit's execution model without requiring an async event loop.

Tables
------
users               — application-level user accounts
oauth_connections   — per-user Composio connected account IDs, one row per provider
learning_paths      — generated path metadata + output URLs
"""
from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Generator, Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DB_PATH = "app.db"
_LOCK = threading.Lock()  # serialize writes from multiple Streamlit threads


def set_db_path(path: str) -> None:
    """Override the database path (useful for tests or alternate envs)."""
    global _DB_PATH
    _DB_PATH = path


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

@contextmanager
def _get_conn() -> Generator[sqlite3.Connection, None, None]:
    """Yield a SQLite connection with row_factory and WAL mode enabled."""
    conn = sqlite3.connect(_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    email       TEXT    NOT NULL UNIQUE,
    name        TEXT,
    created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE IF NOT EXISTS oauth_connections (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id               INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    provider              TEXT    NOT NULL,
    connected_account_id  TEXT    NOT NULL,
    created_at            TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    UNIQUE (user_id, provider)
);

CREATE TABLE IF NOT EXISTS learning_paths (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    goal          TEXT    NOT NULL,
    playlist_url  TEXT,
    google_doc_url TEXT,
    notion_url    TEXT,
    created_at    TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);
"""


def init_db(db_path: Optional[str] = None) -> None:
    """Create all tables if they don't already exist."""
    if db_path:
        set_db_path(db_path)
    with _get_conn() as conn:
        conn.executescript(SCHEMA_SQL)


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

def get_or_create_user(email: str, name: Optional[str] = None) -> dict:
    """Return the user row for *email*, creating it if it doesn't exist.

    Returns a plain dict with keys: id, email, name, created_at.
    """
    email = email.strip().lower()
    with _LOCK:
        with _get_conn() as conn:
            row = conn.execute(
                "SELECT id, email, name, created_at FROM users WHERE email = ?", (email,)
            ).fetchone()
            if row:
                return dict(row)
            now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            conn.execute(
                "INSERT INTO users (email, name, created_at) VALUES (?, ?, ?)",
                (email, name or email.split("@")[0], now),
            )
            row = conn.execute(
                "SELECT id, email, name, created_at FROM users WHERE email = ?", (email,)
            ).fetchone()
            return dict(row)


def get_user_by_id(user_id: int) -> Optional[dict]:
    """Return user dict by ID or None if not found."""
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT id, email, name, created_at FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        return dict(row) if row else None


# ---------------------------------------------------------------------------
# OAuth connections
# ---------------------------------------------------------------------------

def save_oauth_connection(user_id: int, provider: str, connected_account_id: str) -> None:
    """Upsert a Composio connected account ID for a user+provider pair."""
    provider = provider.strip().lower()
    with _LOCK:
        with _get_conn() as conn:
            conn.execute(
                """
                INSERT INTO oauth_connections (user_id, provider, connected_account_id)
                VALUES (?, ?, ?)
                ON CONFLICT(user_id, provider) DO UPDATE SET
                    connected_account_id = excluded.connected_account_id,
                    created_at = strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
                """,
                (user_id, provider, connected_account_id),
            )


def get_connection(user_id: int, provider: str) -> Optional[str]:
    """Return the connectedAccountId for (user, provider) or None if not connected."""
    provider = provider.strip().lower()
    with _get_conn() as conn:
        row = conn.execute(
            "SELECT connected_account_id FROM oauth_connections WHERE user_id = ? AND provider = ?",
            (user_id, provider),
        ).fetchone()
        return row["connected_account_id"] if row else None


def get_connection_status(user_id: int) -> dict:
    """Return a dict of {provider: bool} for all known providers."""
    providers = ["youtube", "googledrive", "notion"]
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT provider FROM oauth_connections WHERE user_id = ?", (user_id,)
        ).fetchall()
    connected = {row["provider"] for row in rows}
    return {p: (p in connected) for p in providers}


def delete_connection(user_id: int, provider: str) -> None:
    """Remove a connection (e.g., on disconnect or token revocation)."""
    provider = provider.strip().lower()
    with _LOCK:
        with _get_conn() as conn:
            conn.execute(
                "DELETE FROM oauth_connections WHERE user_id = ? AND provider = ?",
                (user_id, provider),
            )


# ---------------------------------------------------------------------------
# Learning paths
# ---------------------------------------------------------------------------

def save_learning_path(
    user_id: int,
    goal: str,
    *,
    playlist_url: Optional[str] = None,
    google_doc_url: Optional[str] = None,
    notion_url: Optional[str] = None,
) -> int:
    """Persist a generated learning path and return its row ID."""
    with _LOCK:
        with _get_conn() as conn:
            cursor = conn.execute(
                """
                INSERT INTO learning_paths (user_id, goal, playlist_url, google_doc_url, notion_url)
                VALUES (?, ?, ?, ?, ?)
                """,
                (user_id, goal, playlist_url, google_doc_url, notion_url),
            )
            return cursor.lastrowid


def get_user_paths(user_id: int, limit: int = 20) -> list[dict]:
    """Return recent learning paths for a user, newest first."""
    with _get_conn() as conn:
        rows = conn.execute(
            """
            SELECT id, goal, playlist_url, google_doc_url, notion_url, created_at
            FROM learning_paths
            WHERE user_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        ).fetchall()
        return [dict(row) for row in rows]
