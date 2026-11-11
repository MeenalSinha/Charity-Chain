"""
schemas.py - Common Standardized API Request & Response Schemas
==============================================================
Defines consistent envelopes, pagination, and error payloads across all endpoints.
"""

from typing import Generic, TypeVar, Optional, List, Any, Dict
from datetime import datetime
from pydantic import BaseModel, Field

DataT = TypeVar("DataT")


class StandardResponse(BaseModel, Generic[DataT]):
    """Standard success response wrapper."""
    success: bool = True
    data: DataT
    message: Optional[str] = None
    correlation_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Optional[Dict[str, Any]] = None


class StandardErrorResponse(BaseModel):
    """Standard error response wrapper."""
    success: bool = False
    error: ErrorDetail
    correlation_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class PaginationMeta(BaseModel):
    page: int
    page_size: int
    total_items: int
    total_pages: int


class PaginatedResponse(BaseModel, Generic[DataT]):
    """Paginated list response wrapper."""
    success: bool = True
    data: List[DataT]
    pagination: PaginationMeta
    correlation_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    database_connected: bool
    timestamp: datetime = Field(default_factory=datetime.utcnow)

