"""
events.py - Centralized Domain Event Contracts & Event Catalog
==============================================================
Defines the canonical event schemas, serialization, and sanitization
for CharityChain's event-driven communication infrastructure.
"""

import uuid
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional

from backend.app.core.constants import DomainEventType
from backend.app.core.canonical_json import canonicalize_json


@dataclass
class DomainEvent:
    """Canonical domain event envelope emitted across CharityChain subsystems."""
    event_id: str
    event_type: str
    schema_version: str
    occurred_at: str
    entity_type: str
    entity_id: str
    payload: Dict[str, Any]
    organization_id: Optional[str] = None
    actor_user_id: Optional[str] = None
    correlation_id: Optional[str] = None
    idempotency_key: Optional[str] = None

    def __post_init__(self):
        if not self.event_id:
            self.event_id = f"evt_{uuid.uuid4().hex}"
        if not self.occurred_at:
            self.occurred_at = datetime.now(timezone.utc).isoformat()
        if not self.idempotency_key:
            self.idempotency_key = f"{self.event_type}:{self.entity_id}:{self.event_id}"

    @property
    def id(self) -> str:
        return self.event_id

    @property
    def timestamp(self) -> str:
        return self.occurred_at

    def to_dict(self) -> Dict[str, Any]:
        """Returns standard dictionary representation."""
        return {
            "id": self.event_id,
            "type": self.event_type,
            "version": self.schema_version,
            "created_at": self.occurred_at,
            "entity_type": self.entity_type,
            "entity_id": self.entity_id,
            "organization_id": self.organization_id,
            "actor_user_id": self.actor_user_id,
            "correlation_id": self.correlation_id,
            "idempotency_key": self.idempotency_key,
            "data": self.payload
        }

    def to_json(self) -> str:
        """Serializes event envelope using RFC 8785 canonical JSON."""
        return canonicalize_json(self.to_dict())

    def to_canonical_json(self) -> str:
        return self.to_json()

    def compute_sha256(self) -> str:
        import hashlib
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    @classmethod
    def create(
        cls,
        event_type: DomainEventType | str,
        entity_type: str,
        entity_id: str,
        payload: Dict[str, Any],
        organization_id: Optional[str] = None,
        actor_user_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        schema_version: str = "v1.0.0"
    ) -> "DomainEvent":
        """Factory helper creating a validated, sanitized domain event."""
        event_type_str = event_type.value if isinstance(event_type, DomainEventType) else str(event_type)

        # Sanitize payload: never leak sensitive credentials, tokens, or raw payment cards
        sanitized_payload = cls._sanitize_payload(payload)

        return cls(
            event_id=f"evt_{uuid.uuid4().hex}",
            event_type=event_type_str,
            schema_version=schema_version,
            occurred_at=datetime.now(timezone.utc).isoformat(),
            entity_type=entity_type,
            entity_id=str(entity_id),
            organization_id=str(organization_id) if organization_id else None,
            actor_user_id=str(actor_user_id) if actor_user_id else None,
            correlation_id=correlation_id,
            idempotency_key=idempotency_key or f"{event_type_str}:{entity_id}:{uuid.uuid4().hex[:12]}",
            payload=sanitized_payload
        )

    @staticmethod
    def _sanitize_payload(data: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively removes sensitive keys from event payloads."""
        if not isinstance(data, dict):
            return data

        prohibited_keys = {
            "password", "hashed_password", "token", "access_token",
            "refresh_token", "secret", "private_key", "client_secret",
            "card_number", "cvv", "payment_method_token", "signing_key",
            "api_key", "authorization"
        }

        sanitized = {}
        for k, v in data.items():
            if k.lower() in prohibited_keys:
                sanitized[k] = "[REDACTED]"
                continue
            if isinstance(v, dict):
                sanitized[k] = DomainEvent._sanitize_payload(v)
            elif isinstance(v, list):
                sanitized[k] = [
                    DomainEvent._sanitize_payload(item) if isinstance(item, dict) else item
                    for item in v
                ]
            else:
                sanitized[k] = v

        return sanitized


sanitize_payload_for_event = DomainEvent._sanitize_payload
