"""
security_audit_service.py - Centralized Tamper-Evident Security Audit Service
=============================================================================
Provides append-only persistence with SHA-256 hash chaining, periodic integrity
audits, and compliance reporting for identity, authorization, and administrative events.
Designed for synchronous SQLAlchemy Session consistency across the platform.
"""

import uuid
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List
from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from backend.app.db.models.security_audit import (
    SecurityAuditEvent,
    compute_security_audit_hash,
    GENESIS_SECURITY_AUDIT_HASH,
)

logger = logging.getLogger("charitychain.security_audit")


class SecurityAuditService:
    """Manages append-only security audit events and cryptographic chain integrity."""

    @staticmethod
    def record_event(
        db: Session,
        event_type: str,
        action: str,
        severity: str = "INFO",
        user_id: Optional[str] = None,
        actor_email: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> SecurityAuditEvent:
        """
        Appends a new security event to the tamper-evident hash chain.
        """
        # 1. Fetch latest event to get previous_hash
        latest_event = (
            db.query(SecurityAuditEvent)
            .order_by(desc(SecurityAuditEvent.created_at), desc(SecurityAuditEvent.id))
            .first()
        )

        previous_hash = latest_event.event_hash if latest_event else GENESIS_SECURITY_AUDIT_HASH

        event_id = str(uuid.uuid4())
        created_at = datetime.utcnow()
        created_at_iso = created_at.isoformat()
        event_details = details or {}

        # 2. Compute cryptographic SHA-256 hash incorporating previous_hash
        event_hash = compute_security_audit_hash(
            previous_hash=previous_hash,
            event_id=event_id,
            event_type=event_type,
            action=action,
            created_at_iso=created_at_iso,
            details=event_details,
        )

        # 3. Instantiate and persist model
        audit_entry = SecurityAuditEvent(
            id=event_id,
            event_id=event_id,
            event_type=event_type,
            severity=severity.upper(),
            user_id=user_id,
            actor_email=actor_email,
            ip_address=ip_address,
            user_agent=user_agent,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            details=event_details,
            previous_hash=previous_hash,
            event_hash=event_hash,
            created_at=created_at,
        )

        db.add(audit_entry)
        db.flush()

        logger.info(
            f"Security audit event recorded: type={event_type} action={action} "
            f"severity={severity} user={user_id} hash={event_hash[:16]}..."
        )
        return audit_entry

    @staticmethod
    def verify_chain_integrity(
        db: Session,
        limit: int = 1000,
    ) -> Dict[str, Any]:
        """
        Audits the hash chain sequentially from genesis to verify no entries
        have been altered, deleted, or injected.
        """
        events = (
            db.query(SecurityAuditEvent)
            .order_by(SecurityAuditEvent.created_at.asc(), SecurityAuditEvent.id.asc())
            .limit(limit)
            .all()
        )

        if not events:
            return {
                "is_valid": True,
                "audited_count": 0,
                "message": "Security audit chain is empty (genesis ready)",
            }

        prev_hash = GENESIS_SECURITY_AUDIT_HASH

        for idx, event in enumerate(events):
            # Check 1: Previous hash link
            if idx == 0:
                if event.previous_hash != GENESIS_SECURITY_AUDIT_HASH:
                    prev_hash = event.previous_hash
            else:
                if event.previous_hash != prev_hash:
                    logger.error(
                        f"Audit chain broken at id={event.id} event_id={event.event_id}: "
                        f"expected prev_hash={prev_hash}, found={event.previous_hash}"
                    )
                    return {
                        "is_valid": False,
                        "broken_at_event_id": event.event_id,
                        "broken_at_id": event.id,
                        "reason": "PREVIOUS_HASH_MISMATCH",
                        "audited_count": idx,
                    }

            # Check 2: Recompute event hash
            created_at_iso = event.created_at.isoformat() if event.created_at else ""
            expected_hash = compute_security_audit_hash(
                previous_hash=event.previous_hash,
                event_id=event.event_id,
                event_type=event.event_type,
                action=event.action,
                created_at_iso=created_at_iso,
                details=event.details,
            )

            if event.event_hash != expected_hash:
                logger.error(
                    f"Audit hash tampering detected at id={event.id} event_id={event.event_id}: "
                    f"stored={event.event_hash}, expected={expected_hash}"
                )
                return {
                    "is_valid": False,
                    "broken_at_event_id": event.event_id,
                    "broken_at_id": event.id,
                    "reason": "EVENT_HASH_MISMATCH",
                    "audited_count": idx,
                }

            prev_hash = event.event_hash

        return {
            "is_valid": True,
            "audited_count": len(events),
            "latest_event_id": events[-1].event_id,
            "latest_event_hash": events[-1].event_hash,
            "message": f"Successfully verified {len(events)} security audit records. Chain is intact.",
        }

    @staticmethod
    def get_audit_events(
        db: Session,
        event_type: Optional[str] = None,
        severity: Optional[str] = None,
        user_id: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        page: int = 1,
        page_size: int = 50,
    ) -> Dict[str, Any]:
        """
        Returns paginated and filtered security audit logs.
        """
        query = db.query(SecurityAuditEvent)

        if event_type:
            query = query.filter(SecurityAuditEvent.event_type == event_type)
        if severity:
            query = query.filter(SecurityAuditEvent.severity == severity.upper())
        if user_id:
            query = query.filter(SecurityAuditEvent.user_id == user_id)
        if start_time:
            query = query.filter(SecurityAuditEvent.created_at >= start_time)
        if end_time:
            query = query.filter(SecurityAuditEvent.created_at <= end_time)

        total_count = query.count()
        offset = (page - 1) * page_size
        events = (
            query.order_by(desc(SecurityAuditEvent.created_at), desc(SecurityAuditEvent.id))
            .offset(offset)
            .limit(page_size)
            .all()
        )

        return {
            "items": [e.to_dict() for e in events],
            "total": total_count,
            "page": page,
            "page_size": page_size,
            "total_pages": (total_count + page_size - 1) // page_size if page_size else 1,
        }
