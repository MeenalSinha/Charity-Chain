"""
impact_ledger_service.py - Cryptographically Linked Proof-of-Impact Event Ledger
================================================================================
Implements an append-only, tamper-evident event stream recording all verified milestones,
evidence evaluations, human review decisions, fund disbursements, and certificate issuances.
Each event is cryptographically linked to its predecessor via SHA-256 hash chaining.
"""

import hashlib
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple

from sqlalchemy.orm import Session
from sqlalchemy import select

from backend.app.core.config import settings
from backend.app.core.canonical_json import canonicalize_json, compute_canonical_hash
from backend.app.core.constants import ImpactLedgerEventType
from backend.app.db.models.reviews_impact import (
    ImpactLedgerStream, ImpactLedgerEvent, ImpactLedgerCheckpoint
)
from backend.app.services.key_management_service import KeyManagementService

GENESIS_HASH = "0" * 64


class ImpactLedgerService:
    @staticmethod
    def compute_event_hash(
        previous_hash: str,
        stream_id: str,
        sequence_number: int,
        event_type: str,
        payload_hash: str
    ) -> str:
        """
        Deterministic domain-separated hash linking each event to its immediate predecessor.
        Formula: SHA-256(previous_hash | stream_id | sequence_number | event_type | payload_hash)
        """
        message = f"{previous_hash}|{stream_id}|{sequence_number}|{event_type}|{payload_hash}"
        return hashlib.sha256(message.encode("utf-8")).hexdigest()

    @classmethod
    def get_or_create_stream(
        cls,
        db: Session,
        stream_id: str,
        aggregate_type: str = "MILESTONE",
        aggregate_id: str = ""
    ) -> ImpactLedgerStream:
        """Retrieves or atomically creates an Impact Ledger stream."""
        stream = db.query(ImpactLedgerStream).filter(ImpactLedgerStream.stream_id == stream_id).first()
        if not stream:
            stream = ImpactLedgerStream(
                stream_id=stream_id,
                aggregate_type=aggregate_type,
                aggregate_id=aggregate_id or stream_id,
                current_sequence_number=0,
                head_event_hash=GENESIS_HASH,
                status="ACTIVE"
            )
            db.add(stream)
            db.commit()
            db.refresh(stream)
        return stream

    @classmethod
    def append_event(
        cls,
        db: Session,
        stream_id: str,
        event_type: str,
        source_event_type: str,
        source_event_id: str,
        payload: Dict[str, Any],
        actor_id: Optional[str] = None,
        aggregate_type: str = "MILESTONE",
        aggregate_id: str = ""
    ) -> ImpactLedgerEvent:
        """
        Appends an event to the append-only Impact Ledger stream.
        Guarantees strict monotonic sequence progression and cryptographic hash chaining.
        Idempotent: duplicate source events return existing ledger records.
        """
        # 1. Idempotency check
        existing_event = db.query(ImpactLedgerEvent).filter(
            ImpactLedgerEvent.stream_id == stream_id,
            ImpactLedgerEvent.source_event_type == source_event_type,
            ImpactLedgerEvent.source_event_id == source_event_id
        ).first()
        if existing_event:
            return existing_event

        # 2. Lock and retrieve stream
        stream = db.query(ImpactLedgerStream).filter(
            ImpactLedgerStream.stream_id == stream_id
        ).with_for_update().first()

        if not stream:
            stream = ImpactLedgerStream(
                stream_id=stream_id,
                aggregate_type=aggregate_type,
                aggregate_id=aggregate_id or stream_id,
                current_sequence_number=0,
                head_event_hash=GENESIS_HASH,
                status="ACTIVE"
            )
            db.add(stream)
            db.flush()

        next_seq = stream.current_sequence_number + 1
        prev_hash = stream.head_event_hash

        # 3. Canonicalize payload and compute hashes
        canonical_str = canonicalize_json(payload)
        payload_hash = hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()
        current_hash = cls.compute_event_hash(
            previous_hash=prev_hash,
            stream_id=stream_id,
            sequence_number=next_seq,
            event_type=event_type,
            payload_hash=payload_hash
        )

        # 4. Create and persist event
        event = ImpactLedgerEvent(
            stream_id=stream_id,
            sequence_number=next_seq,
            event_type=event_type,
            schema_version=getattr(settings, "IMPACT_LEDGER_SCHEMA_VERSION", "v9.0.0"),
            source_event_type=source_event_type,
            source_event_id=source_event_id,
            payload_json=canonical_str,
            payload_hash=payload_hash,
            previous_event_hash=prev_hash,
            current_event_hash=current_hash,
            actor_id=actor_id,
            timestamp=datetime.utcnow()
        )
        db.add(event)

        # 5. Update stream head
        stream.current_sequence_number = next_seq
        stream.head_event_hash = current_hash

        db.commit()
        db.refresh(event)

        # 6. Periodic automated checkpoint
        interval = getattr(settings, "IMPACT_LEDGER_AUTO_CHECKPOINT_INTERVAL", 50)
        if next_seq > 0 and next_seq % interval == 0:
            try:
                cls.create_checkpoint(db, stream_id)
            except Exception:
                pass

        return event

    @classmethod
    def verify_stream_chain(cls, db: Session, stream_id: str) -> Dict[str, Any]:
        """
        Mathematically verifies the integrity of an Impact Ledger event stream.
        Checks sequence continuity, previous hash chaining, payload hash matching,
        and current event hash recomputation.
        """
        stream = db.query(ImpactLedgerStream).filter(ImpactLedgerStream.stream_id == stream_id).first()
        if not stream:
            return {
                "stream_id": stream_id,
                "is_valid": False,
                "status": "STREAM_NOT_FOUND",
                "total_events": 0,
                "errors": [f"Stream '{stream_id}' does not exist."]
            }

        events = db.query(ImpactLedgerEvent).filter(
            ImpactLedgerEvent.stream_id == stream_id
        ).order_by(ImpactLedgerEvent.sequence_number.asc()).all()

        if not events:
            return {
                "stream_id": stream_id,
                "is_valid": True,
                "status": "EMPTY",
                "total_events": 0,
                "head_hash": stream.head_event_hash,
                "errors": []
            }

        expected_prev_hash = GENESIS_HASH
        errors: List[str] = []

        for idx, ev in enumerate(events, start=1):
            # Check sequence continuity
            if ev.sequence_number != idx:
                errors.append(
                    f"Sequence break at index {idx}: expected sequence {idx}, found {ev.sequence_number}"
                )

            # Check previous hash continuity
            if ev.previous_event_hash != expected_prev_hash:
                errors.append(
                    f"Hash chain broken at sequence {ev.sequence_number}: "
                    f"expected previous hash {expected_prev_hash}, found {ev.previous_event_hash}"
                )

            # Recompute payload hash
            recomputed_payload_hash = hashlib.sha256(ev.payload_json.encode("utf-8")).hexdigest()
            if ev.payload_hash != recomputed_payload_hash:
                errors.append(
                    f"Payload tampering detected at sequence {ev.sequence_number}: "
                    f"stored hash {ev.payload_hash} != recomputed {recomputed_payload_hash}"
                )

            # Recompute current event hash
            recomputed_current_hash = cls.compute_event_hash(
                previous_hash=ev.previous_event_hash,
                stream_id=stream_id,
                sequence_number=ev.sequence_number,
                event_type=ev.event_type,
                payload_hash=ev.payload_hash
            )
            if ev.current_event_hash != recomputed_current_hash:
                errors.append(
                    f"Event hash invalid at sequence {ev.sequence_number}: "
                    f"stored {ev.current_event_hash} != recomputed {recomputed_current_hash}"
                )

            expected_prev_hash = ev.current_event_hash

        # Verify head hash matches stream record
        if stream.head_event_hash != expected_prev_hash:
            errors.append(
                f"Stream head hash mismatch: stream record {stream.head_event_hash} != chain end {expected_prev_hash}"
            )

        is_valid = len(errors) == 0
        return {
            "stream_id": stream_id,
            "is_valid": is_valid,
            "status": "VALID" if is_valid else "INVALID",
            "total_events": len(events),
            "head_hash": stream.head_event_hash,
            "errors": errors
        }

    @classmethod
    def create_checkpoint(cls, db: Session, stream_id: str) -> ImpactLedgerCheckpoint:
        """
        Creates an Ed25519 digitally signed checkpoint of the stream's current head.
        """
        stream = db.query(ImpactLedgerStream).filter(ImpactLedgerStream.stream_id == stream_id).first()
        if not stream:
            raise ValueError(f"Stream '{stream_id}' not found.")

        # Ensure key is registered
        KeyManagementService.ensure_active_key_registered(db)
        active_kid, _ = KeyManagementService.get_active_private_key()

        checkpoint_payload = {
            "stream_id": stream.stream_id,
            "sequence_number": stream.current_sequence_number,
            "head_hash": stream.head_event_hash,
            "checkpoint_time": datetime.utcnow().isoformat()
        }
        canonical_str = canonicalize_json(checkpoint_payload)
        sig_b64, kid, _ = KeyManagementService.sign_payload(canonical_str, key_id=active_kid)

        checkpoint = ImpactLedgerCheckpoint(
            stream_id=stream.stream_id,
            sequence_number=stream.current_sequence_number,
            head_hash=stream.head_event_hash,
            signing_key_id=kid,
            signature_algorithm="Ed25519",
            signature=sig_b64,
            checkpoint_metadata_json=canonical_str
        )
        db.add(checkpoint)
        db.commit()
        db.refresh(checkpoint)
        return checkpoint

