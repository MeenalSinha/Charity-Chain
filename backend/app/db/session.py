"""
session.py - Database Engine, Session Factory & Connectivity Check
==================================================================
Provides reliable connection pooling, session management, and health checks.
"""

from typing import Generator
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from backend.app.core.config import settings
from backend.app.core.logging import logger

# Build database connection URL
db_url = settings.DATABASE_URL.replace("+asyncpg", "").replace("+aiosqlite", "")

engine_kwargs = {"echo": False}
if "sqlite" in db_url:
    engine_kwargs["connect_args"] = {"check_same_thread": False}
else:
    engine_kwargs.update({
        "pool_size": settings.DATABASE_POOL_SIZE,
        "max_overflow": settings.DATABASE_MAX_OVERFLOW,
        "pool_timeout": settings.DATABASE_POOL_TIMEOUT_SECONDS,
        "pool_pre_ping": True,
    })

engine = create_engine(db_url, **engine_kwargs)
SessionFactory = sessionmaker(bind=engine, autocommit=False, autoflush=False)
SessionLocal = SessionFactory


def get_db_session() -> Generator[Session, None, None]:
    """Dependency injection helper yielding an active transactional session."""
    session = SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


get_db = get_db_session



def check_db_connection() -> bool:
    """Verifies that the database engine can execute queries and ping the server."""
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            return result.scalar() == 1
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False
