"""
db.py — Database layer for the Learning Path Generator using SQLAlchemy.

Supports both SQLite (for local dev) and PostgreSQL (for production).
Reads DATABASE_URL from the environment.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv
from sqlalchemy import (
    Column, Integer, String, DateTime, ForeignKey, create_engine, text, UniqueConstraint
)
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_DB_URL = os.getenv("DATABASE_URL", "sqlite:///./app.db").strip()

# Handle standard postgres URI vs psycopg2 URI
if _DB_URL.startswith("postgres://"):
    _DB_URL = _DB_URL.replace("postgres://", "postgresql://", 1)
if _DB_URL.startswith("postgresql://") and "psycopg2" not in _DB_URL:
    # Use standard psycopg2 driver if postgresql is specified without driver
    pass

# Engine configuration
connect_args = {}
if _DB_URL.startswith("sqlite"):
    connect_args["check_same_thread"] = False

engine = create_engine(_DB_URL, connect_args=connect_args, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ---------------------------------------------------------------------------
# Schema Definitions
# ---------------------------------------------------------------------------

def _now():
    return datetime.now(timezone.utc)

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, nullable=False, index=True)
    name = Column(String, nullable=True)
    created_at = Column(DateTime, default=_now, nullable=False)

    def to_dict(self):
        return {
            "id": self.id,
            "email": self.email,
            "name": self.name,
            "created_at": self.created_at.isoformat() + "Z" if self.created_at else None,
        }

class OAuthConnection(Base):
    __tablename__ = "oauth_connections"
    __table_args__ = (UniqueConstraint('user_id', 'provider', name='uix_user_provider'),)

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    provider = Column(String, nullable=False)
    connected_account_id = Column(String, nullable=False)
    # For Notion: UUID of the page/database to create pages inside.
    # The Notion API does not allow root-level page creation.
    notion_parent_page_id = Column(String, nullable=True)
    created_at = Column(DateTime, default=_now, nullable=False)

class LearningPathRecord(Base):
    __tablename__ = "learning_paths"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    goal = Column(String, nullable=False)
    playlist_url = Column(String, nullable=True)
    google_doc_url = Column(String, nullable=True)
    notion_url = Column(String, nullable=True)
    created_at = Column(DateTime, default=_now, nullable=False)

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "goal": self.goal,
            "playlist_url": self.playlist_url,
            "google_doc_url": self.google_doc_url,
            "notion_url": self.notion_url,
            "created_at": self.created_at.isoformat() + "Z" if self.created_at else None,
        }


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Create all tables if they don't already exist."""
    Base.metadata.create_all(bind=engine)
    # Migration: add notion_parent_page_id column to existing databases
    try:
        with engine.connect() as conn:
            conn.execute(text(
                "ALTER TABLE oauth_connections ADD COLUMN notion_parent_page_id VARCHAR"
            ))
            conn.commit()
    except Exception:
        pass  # Column already exists — safe to ignore


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

def get_or_create_user(email: str, name: Optional[str] = None) -> dict:
    email = email.strip().lower()
    with SessionLocal() as session:
        user = session.query(User).filter(User.email == email).first()
        if user:
            return user.to_dict()

        user = User(email=email, name=name or email.split("@")[0])
        session.add(user)
        session.commit()
        session.refresh(user)
        return user.to_dict()


def get_user_by_id(user_id: int) -> Optional[dict]:
    with SessionLocal() as session:
        user = session.query(User).filter(User.id == user_id).first()
        return user.to_dict() if user else None


# ---------------------------------------------------------------------------
# OAuth connections
# ---------------------------------------------------------------------------

def save_oauth_connection(user_id: int, provider: str, connected_account_id: str) -> None:
    provider = provider.strip().lower()
    with SessionLocal() as session:
        conn = session.query(OAuthConnection).filter(
            OAuthConnection.user_id == user_id,
            OAuthConnection.provider == provider
        ).first()

        if conn:
            conn.connected_account_id = connected_account_id
            conn.created_at = _now()
        else:
            conn = OAuthConnection(
                user_id=user_id,
                provider=provider,
                connected_account_id=connected_account_id,
            )
            session.add(conn)
        session.commit()


def get_connection(user_id: int, provider: str) -> Optional[str]:
    provider = provider.strip().lower()
    with SessionLocal() as session:
        conn = session.query(OAuthConnection).filter(
            OAuthConnection.user_id == user_id,
            OAuthConnection.provider == provider
        ).first()
        return conn.connected_account_id if conn else None


def get_connection_status(user_id: int) -> dict:
    providers = ["youtube", "googledrive", "notion"]
    with SessionLocal() as session:
        conns = session.query(OAuthConnection.provider).filter(
            OAuthConnection.user_id == user_id
        ).all()
        connected = {c[0] for c in conns}
        return {p: (p in connected) for p in providers}


def delete_connection(user_id: int, provider: str) -> None:
    provider = provider.strip().lower()
    with SessionLocal() as session:
        session.query(OAuthConnection).filter(
            OAuthConnection.user_id == user_id,
            OAuthConnection.provider == provider
        ).delete()
        session.commit()


def save_notion_parent_id(user_id: int, page_id: str) -> None:
    """Persist the Notion parent page/database UUID for a user."""
    page_id = page_id.strip()
    with SessionLocal() as session:
        conn = session.query(OAuthConnection).filter(
            OAuthConnection.user_id == user_id,
            OAuthConnection.provider == "notion"
        ).first()
        if conn:
            conn.notion_parent_page_id = page_id or None
            session.commit()


def get_notion_parent_id(user_id: int) -> Optional[str]:
    """Return the stored Notion parent page/database UUID for a user, or None."""
    with SessionLocal() as session:
        conn = session.query(OAuthConnection).filter(
            OAuthConnection.user_id == user_id,
            OAuthConnection.provider == "notion"
        ).first()
        return conn.notion_parent_page_id if conn else None


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
    with SessionLocal() as session:
        lp = LearningPathRecord(
            user_id=user_id,
            goal=goal,
            playlist_url=playlist_url,
            google_doc_url=google_doc_url,
            notion_url=notion_url,
        )
        session.add(lp)
        session.commit()
        session.refresh(lp)
        return lp.id


def get_user_paths(user_id: int, limit: int = 20) -> list[dict]:
    with SessionLocal() as session:
        paths = session.query(LearningPathRecord).filter(
            LearningPathRecord.user_id == user_id
        ).order_by(LearningPathRecord.created_at.desc()).limit(limit).all()
        return [p.to_dict() for p in paths]
