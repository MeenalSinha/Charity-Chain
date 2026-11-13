"""
security_audit.py - Centralized Tamper-Evident Security Audit Trail
===================================================================
Maintains a cryptographically hash-chained append-only record of security events,
distinct from the Proof-of-Impact ledger.
Provides non-repudiation, tamper detection, and compliance auditing.
"""

import uuid
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional
from sqlalchemy import Column, Integer, String, DateTime, JSON, Index, Text

from backend.app.db.base import Base
from backend.app.core.canonical_json import canonicalize_json

GENESIS_SECURITY_AUDIT_HASH = "0" * 64


def compute_security_audit_hash(
    previous_hash: str,
    event_id: str,
    event_type: str,
    action: str,
    created_at_iso: str,
    details: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Computes a deterministic SHA-256 hash for a security audit event.
    Ensures cryptographic tamper-evidence and chain integrity.
    """
    normalized_details = canonicalize_json(details or {})
    payload = f"{previous_hash}|{event_id}|{event_type}|{action}|{created_at_iso}|{normalized_details}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class SecurityAuditEvent(Base):
    """
    Append-only security audit log with cryptographic SHA-256 hash chaining.
    Tracks identity events, privilege escalations, rate-limit violations, and administrative actions.
    Inherits id (String(36)), created_at, updated_at, is_active from Base.
    """
    __tablename__ = "security_audit_events"

    event_id = Column(String(36), unique=True, nullable=False, default=lambda: str(uuid.uuid4()), index=True)
    sequence_number = Column(Integer, nullable=True, index=True)
    event_type = Column(String(64), nullable=False, index=True)
    severity = Column(String(16), nullable=False, default="INFO", index=True)
    user_id = Column(String(36), nullable=True, index=True)
    actor_email = Column(String(255), nullable=True)
    ip_address = Column(String(64), nullable=True)
    user_agent = Column(String(255), nullable=True)
    resource_type = Column(String(64), nullable=True)
    resource_id = Column(String(64), nullable=True)
    action = Column(String(128), nullable=False)
    details = Column(JSON, nullable=False, default=dict)
    previous_hash = Column(String(64), nullable=False)
    event_hash = Column(String(64), nullable=False, index=True)

    __table_args__ = (
        Index("ix_sec_audit_type_created", "event_type", "created_at"),
        Index("ix_sec_audit_user_created", "user_id", "created_at"),
        Index("ix_sec_audit_severity_created", "severity", "created_at"),
    )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "event_id": self.event_id,
            "sequence_number": self.sequence_number,
            "event_type": self.event_type,
            "severity": self.severity,
            "user_id": self.user_id,
            "actor_email": self.actor_email,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "resource_type": self.resource_type,
            "resource_id": self.resource_id,
            "action": self.action,
            "details": self.details,
            "previous_hash": self.previous_hash,
            "event_hash": self.event_hash,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
