"""
base.py - SQLAlchemy Declarative Base & Base Entity Mixin
=========================================================
Provides common columns and table conventions across all domain models.
"""

import uuid
from datetime import datetime
from sqlalchemy import Column, String, DateTime, Boolean
from sqlalchemy.orm import declarative_base, declared_attr


class CustomBase:
    """Base mixin providing auto-generated table names and primary audit timestamps."""
    @declared_attr
    def __tablename__(cls) -> str:
        # Converts CamelCase to snake_case table name
        name = cls.__name__
        import re
        s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()), index=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)

    def __init__(self, **kwargs):
        if "id" not in kwargs:
            self.id = str(uuid.uuid4())
        if "created_at" not in kwargs:
            self.created_at = datetime.utcnow()
        if "updated_at" not in kwargs:
            self.updated_at = datetime.utcnow()
        if "is_active" not in kwargs:
            self.is_active = True
        for key, value in kwargs.items():
            setattr(self, key, value)


Base = declarative_base(cls=CustomBase, constructor=None)
