"""
test_impact_ledger_chain.py - Unit Tests for Impact Ledger Chain Integrity & Verification
========================================================================================
Verifies monotonic sequence progression, cryptographic hash chaining, tamper detection,
idempotent deduplication, and periodic signed checkpoints.
"""

import uuid
import pytest
from backend.app.core.constants import ImpactLedgerEventType
from backend.app.db.models.reviews_impact import ImpactLedgerStream, ImpactLedgerEvent
from backend.app.services.impact_ledger_service import ImpactLedgerService, GENESIS_HASH


def test_append_events_monotonic_sequence_and_hash_chaining(db_session):
    """Verifies that appending events increments sequence number and links hashes sequentially."""
    stream_id = f"stream-{uuid.uuid4()}"

    e1 = ImpactLedgerService.append_event(
        db=db_session,
        stream_id=stream_id,
        event_type=ImpactLedgerEventType.EVIDENCE_SUBMITTED.value,
        source_event_type="evidence_submission",
        source_event_id=f"src-{uuid.uuid4()}",
        payload={"round": 1, "evidence_count": 3}
    )
    assert e1.sequence_number == 1
    assert e1.previous_event_hash == GENESIS_HASH

    e2 = ImpactLedgerService.append_event(
        db=db_session,
        stream_id=stream_id,
        event_type=ImpactLedgerEventType.DUAL_KEY_APPROVAL_FINALIZED.value,
        source_event_type="review_decision",
        source_event_id=f"src-{uuid.uuid4()}",
        payload={"decision": "APPROVE", "approvers": 2}
    )
    assert e2.sequence_number == 2
    assert e2.previous_event_hash == e1.current_event_hash

    # Verify stream chain report
    audit = ImpactLedgerService.verify_stream_chain(db=db_session, stream_id=stream_id)
    assert audit["is_valid"] is True
    assert audit["total_events"] == 2
    assert len(audit["errors"]) == 0


def test_idempotent_event_append(db_session):
    """Verifies that submitting the same source event id returns the existing record without appending."""
    stream_id = f"stream-{uuid.uuid4()}"
    src_id = f"src-{uuid.uuid4()}"

    e1 = ImpactLedgerService.append_event(
        db=db_session,
        stream_id=stream_id,
        event_type=ImpactLedgerEventType.EVIDENCE_SUBMITTED.value,
        source_event_type="evidence_submission",
        source_event_id=src_id,
        payload={"attempt": 1}
    )

    # Re-submit with same source_event_id
    e2 = ImpactLedgerService.append_event(
        db=db_session,
        stream_id=stream_id,
        event_type=ImpactLedgerEventType.EVIDENCE_SUBMITTED.value,
        source_event_type="evidence_submission",
        source_event_id=src_id,
        payload={"attempt": 2}
    )

    assert e1.id == e2.id
    assert e2.sequence_number == 1

    stream = db_session.query(ImpactLedgerStream).filter_by(stream_id=stream_id).first()
    assert stream.current_sequence_number == 1


def test_tamper_detection_on_mutated_payload(db_session):
    """Verifies that an attacker altering stored payload_json is detected by verify_stream_chain."""
    stream_id = f"stream-{uuid.uuid4()}"

    ImpactLedgerService.append_event(
        db=db_session,
        stream_id=stream_id,
        event_type=ImpactLedgerEventType.MILESTONE_COMPLETED.value,
        source_event_type="milestone",
        source_event_id=f"src-{uuid.uuid4()}",
        payload={"target_amount": 10000}
    )

    # Simulating unauthorized database row mutation
    event = db_session.query(ImpactLedgerEvent).filter_by(stream_id=stream_id, sequence_number=1).first()
    event.payload_json = '{"target_amount":9999999}'
    db_session.commit()

    audit = ImpactLedgerService.verify_stream_chain(db=db_session, stream_id=stream_id)
    assert audit["is_valid"] is False
    assert any("Payload tampering detected" in err for err in audit["errors"])


def test_tamper_detection_on_broken_hash_pointer(db_session):
    """Verifies that an attacker altering previous_event_hash or current_event_hash breaks verification."""
    stream_id = f"stream-{uuid.uuid4()}"

    ImpactLedgerService.append_event(
        db=db_session,
        stream_id=stream_id,
        event_type=ImpactLedgerEventType.MILESTONE_COMPLETED.value,
        source_event_type="milestone",
        source_event_id=f"src-1",
        payload={"data": 1}
    )
    ImpactLedgerService.append_event(
        db=db_session,
        stream_id=stream_id,
        event_type=ImpactLedgerEventType.DISBURSEMENT_CONFIRMED.value,
        source_event_type="disbursement",
        source_event_id=f"src-2",
        payload={"data": 2}
    )

    # Corrupt current_event_hash of event 1
    event1 = db_session.query(ImpactLedgerEvent).filter_by(stream_id=stream_id, sequence_number=1).first()
    event1.current_event_hash = "f" * 64
    db_session.commit()

    audit = ImpactLedgerService.verify_stream_chain(db=db_session, stream_id=stream_id)
    assert audit["is_valid"] is False
    assert len(audit["errors"]) > 0


def test_checkpoint_creation_and_integrity(db_session):
    """Verifies creation of signed cryptographic checkpoints over the event ledger stream."""
    stream_id = f"stream-{uuid.uuid4()}"

    for i in range(1, 4):
        ImpactLedgerService.append_event(
            db=db_session,
            stream_id=stream_id,
            event_type=ImpactLedgerEventType.EVIDENCE_SUBMITTED.value,
            source_event_type="test",
            source_event_id=f"src-{i}",
            payload={"index": i}
        )

    checkpoint = ImpactLedgerService.create_checkpoint(db=db_session, stream_id=stream_id)
    assert checkpoint.sequence_number == 3
    assert len(checkpoint.head_hash) == 64
    assert len(checkpoint.signature) > 30
