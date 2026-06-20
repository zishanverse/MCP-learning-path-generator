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

# Ensure SSL mode is enabled by default for PostgreSQL on cloud deployments.
# This prevents OperationalError when connecting to secure managed databases (e.g. Neon, Supabase).
scheme = _DB_URL.split("://")[0] if "://" in _DB_URL else ""
is_postgres = "postgres" in scheme or "postgresql" in scheme

if is_postgres and "sslmode" not in _DB_URL:
    if "?" in _DB_URL:
        _DB_URL += "&sslmode=require"
    else:
        _DB_URL += "?sslmode=require"

# Engine configuration
connect_args = {}
if _DB_URL.startswith("sqlite"):
    connect_args["check_same_thread"] = False

engine = create_engine(_DB_URL, connect_args=connect_args, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

using_fallback_db = False
fallback_error_message = None

# ---------------------------------------------------------------------------
# Schema Definitions
# ---------------------------------------------------------------------------

def _now():
    return datetime.now(timezone.utc)

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, nullable=False, index=True)
    hashed_password = Column(String, nullable=True)
    session_token = Column(String, nullable=True, unique=True, index=True)
    name = Column(String, nullable=True)
    created_at = Column(DateTime, default=_now, nullable=False)

    def to_dict(self):
        return {
            "id": self.id,
            "email": self.email,
            "hashed_password": self.hashed_password,
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
    markdown = Column(String, nullable=True)
    created_at = Column(DateTime, default=_now, nullable=False)

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "goal": self.goal,
            "playlist_url": self.playlist_url,
            "google_doc_url": self.google_doc_url,
            "notion_url": self.notion_url,
            "markdown": self.markdown,
            "created_at": self.created_at.isoformat() + "Z" if self.created_at else None,
        }


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

def init_db() -> None:
    """Create all tables if they don't already exist."""
    global engine, SessionLocal, using_fallback_db, fallback_error_message
    
    try:
        # Try to initialize with the configured engine
        Base.metadata.create_all(bind=engine)
    except Exception as e:
        # If it failed and we are already using sqlite, propagate error
        if _DB_URL.startswith("sqlite"):
            raise e
            
        # Fall back to SQLite
        using_fallback_db = True
        fallback_error_message = str(e)
        
        # Configure fallback sqlite engine
        fallback_url = "sqlite:///./app_fallback.db"
        connect_args_fallback = {"check_same_thread": False}
        engine = create_engine(fallback_url, connect_args=connect_args_fallback, pool_pre_ping=True)
        SessionLocal.configure(bind=engine)
        
        # Initialize the fallback database
        Base.metadata.create_all(bind=engine)
    
    # Migration: add hashed_password column to users
    try:
        with engine.connect() as conn:
            conn.execute(text(
                "ALTER TABLE users ADD COLUMN hashed_password VARCHAR"
            ))
            conn.commit()
    except Exception:
        pass  # Column already exists
        
    # Migration: add session_token column to users
    try:
        with engine.connect() as conn:
            conn.execute(text(
                "ALTER TABLE users ADD COLUMN session_token VARCHAR"
            ))
            conn.commit()
    except Exception:
        pass  # Column already exists
        
    # Migration: add notion_parent_page_id column to existing databases
    try:
        with engine.connect() as conn:
            conn.execute(text(
                "ALTER TABLE oauth_connections ADD COLUMN notion_parent_page_id VARCHAR"
            ))
            conn.commit()
    except Exception:
        pass  # Column already exists — safe to ignore

    # Migration: add google_doc_url and notion_url columns to learning_paths if they don't exist
    try:
        with engine.connect() as conn:
            conn.execute(text(
                "ALTER TABLE learning_paths ADD COLUMN google_doc_url VARCHAR"
            ))
            conn.commit()
    except Exception:
        pass

    try:
        with engine.connect() as conn:
            conn.execute(text(
                "ALTER TABLE learning_paths ADD COLUMN notion_url VARCHAR"
            ))
            conn.commit()
    except Exception:
        pass

    # Migration: add markdown column to learning_paths if it doesn't exist
    try:
        with engine.connect() as conn:
            conn.execute(text(
                "ALTER TABLE learning_paths ADD COLUMN markdown TEXT"
            ))
            conn.commit()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Users
# ---------------------------------------------------------------------------

def get_or_create_user(email: str, name: Optional[str] = None, hashed_password: Optional[str] = None) -> dict:
    email = email.strip().lower()
    with SessionLocal() as session:
        user = session.query(User).filter(User.email == email).first()
        if user:
            # If an existing user (e.g. from passwordless preview) sets a password, save it
            if hashed_password and not user.hashed_password:
                user.hashed_password = hashed_password
                session.commit()
                session.refresh(user)
            return user.to_dict()

        user = User(email=email, name=name or email.split("@")[0], hashed_password=hashed_password)
        session.add(user)
        session.commit()
        session.refresh(user)
        return user.to_dict()

def get_user_by_email(email: str) -> Optional[dict]:
    email = email.strip().lower()
    with SessionLocal() as session:
        user = session.query(User).filter(User.email == email).first()
        return user.to_dict() if user else None

import uuid

def update_session_token(user_id: int, clear: bool = False) -> Optional[str]:
    """Generate a new UUID session token and save it to the DB. Or clear it if clear=True."""
    with SessionLocal() as session:
        user = session.query(User).filter(User.id == user_id).first()
        if not user:
            return None
        token = None if clear else str(uuid.uuid4())
        user.session_token = token
        session.commit()
        return token

def get_user_by_session_token(token: str) -> Optional[dict]:
    """Securely look up a user by their unguessable UUID session token."""
    if not token:
        return None
    with SessionLocal() as session:
        user = session.query(User).filter(User.session_token == token).first()
        return user.to_dict() if user else None


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
    markdown: Optional[str] = None,
) -> int:
    with SessionLocal() as session:
        lp = LearningPathRecord(
            user_id=user_id,
            goal=goal,
            playlist_url=playlist_url,
            google_doc_url=google_doc_url,
            notion_url=notion_url,
            markdown=markdown,
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


# Auto-initialize database on import so connection/fallback checks are ran immediately
try:
    init_db()
except Exception:
    pass
