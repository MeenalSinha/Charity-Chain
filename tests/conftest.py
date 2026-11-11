"""
conftest.py - Pytest Configuration & Test Fixtures
=================================================
Provides isolated environments and synthetic image fixtures.
Ensures tests NEVER modify real user data or production databases.
"""

import os
import sys
from pathlib import Path
import pytest
from io import BytesIO
from PIL import Image, ImageDraw

# Add workspace root to sys.path
WORKSPACE_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(WORKSPACE_ROOT))

from charity_tracker import DatabaseManager, EnhancedAIVerifier, Config


@pytest.fixture
def temp_db_path(tmp_path):
    """Provides a fresh isolated database path for every test."""
    db_file = tmp_path / "test_charity.db"
    return db_file


@pytest.fixture
def isolated_db(temp_db_path):
    """Provides a fresh DatabaseManager instance on a temporary database."""
    db = DatabaseManager(db_file=temp_db_path)
    yield db
    db.close()


@pytest.fixture
def sample_jpeg_bytes():
    """Generates a valid 100x100 synthetic RGB JPEG in memory."""
    img = Image.new('RGB', (100, 100), color=(73, 109, 137))
    d = ImageDraw.Draw(img)
    d.rectangle([(20, 20), (80, 80)], fill=(255, 255, 0))
    buffer = BytesIO()
    img.save(buffer, format='JPEG', quality=95)
    return buffer.getvalue()


@pytest.fixture
def sample_modified_jpeg_bytes():
    """Generates a modified version of sample image to simulate real visual change."""
    img = Image.new('RGB', (100, 100), color=(200, 50, 50))
    d = ImageDraw.Draw(img)
    d.ellipse([(10, 10), (90, 90)], fill=(0, 255, 255))
    buffer = BytesIO()
    img.save(buffer, format='JPEG', quality=95)
    return buffer.getvalue()


@pytest.fixture
def sample_png_bytes():
    """Generates a valid PNG in memory."""
    img = Image.new('RGB', (50, 50), color=(120, 200, 80))
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    return buffer.getvalue()


@pytest.fixture
def ai_verifier():
    """Provides an instance of EnhancedAIVerifier."""
    return EnhancedAIVerifier()



@pytest.fixture
def db_session(tmp_path):
    """Provides a fresh isolated transactional SQLAlchemy session for testing."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from backend.app.db.base import Base
    from backend.app.db.session import get_db_session, get_db
    from backend.app.main import app
    import backend.app.db.models  # ensure models registered

    test_db_file = tmp_path / "test_charity_sql.db"
    test_engine = create_engine(f"sqlite:///{test_db_file}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=test_engine)
    TestSessionFactory = sessionmaker(bind=test_engine, autocommit=False, autoflush=False)
    session = TestSessionFactory()

    # Override FastAPI dependency for test execution
    app.dependency_overrides[get_db_session] = lambda: session
    app.dependency_overrides[get_db] = lambda: session

    try:
        yield session
    finally:
        session.close()
        test_engine.dispose()
        app.dependency_overrides.pop(get_db_session, None)
        app.dependency_overrides.pop(get_db, None)
