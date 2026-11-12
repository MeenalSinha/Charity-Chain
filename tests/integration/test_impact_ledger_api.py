"""
test_impact_ledger_api.py - Integration Tests for Impact Ledger REST APIs
=========================================================================
Tests milestone ledger event retrieval, public cryptographic verification reports,
and administrative checkpointing endpoints.
"""

import uuid
import pytest
from datetime import datetime, timezone

from backend.app.core.constants import PlatformRole, ImpactLedgerEventType
from backend.app.db.models.users import User, AccountStatus
from backend.app.services.impact_ledger_service import ImpactLedgerService
from backend.app.api.v1.impact_ledger import (
    get_milestone_impact_events,
    verify_milestone_ledger_chain,
    list_impact_ledger_streams,
    create_ledger_checkpoint
)


@pytest.fixture
def admin_user(db_session):
    admin = User(
        id=str(uuid.uuid4()),
        email=f"admin_{uuid.uuid4().hex[:8]}@charitychain.org",
        full_name="Ledger Admin",
        hashed_password="mock",
        role=PlatformRole.PLATFORM_ADMIN,
        account_status=AccountStatus.ACTIVE,
        is_verified=True
    )
    db_session.add(admin)
    db_session.commit()
    return admin


@pytest.fixture
def populated_milestone_stream(db_session):
    milestone_id = f"m-{uuid.uuid4()}"
    stream_id = f"milestone-{milestone_id}"

    ImpactLedgerService.append_event(
        db=db_session,
        stream_id=stream_id,
        event_type=ImpactLedgerEventType.EVIDENCE_SUBMITTED.value,
        source_event_type="evidence",
        source_event_id=f"ev-{uuid.uuid4()}",
        payload={"file_count": 2}
    )
    ImpactLedgerService.append_event(
        db=db_session,
        stream_id=stream_id,
        event_type=ImpactLedgerEventType.DUAL_KEY_APPROVAL_FINALIZED.value,
        source_event_type="review",
        source_event_id=f"rev-{uuid.uuid4()}",
        payload={"decision": "APPROVE"}
    )
    return milestone_id, stream_id


def test_get_milestone_impact_events(db_session, populated_milestone_stream):
    """Verifies retrieval of milestone event history in sequence order."""
    milestone_id, stream_id = populated_milestone_stream

    events = get_milestone_impact_events(
        milestone_id=milestone_id,
        db=db_session
    )

    assert len(events) == 2
    assert events[0].sequence_number == 1
    assert events[1].sequence_number == 2
    assert events[1].previous_event_hash == events[0].current_event_hash


def test_verify_milestone_ledger_chain(db_session, populated_milestone_stream):
    """Verifies that mathematical integrity check succeeds on intact stream."""
    milestone_id, stream_id = populated_milestone_stream

    report = verify_milestone_ledger_chain(
        milestone_id=milestone_id,
        db=db_session
    )

    assert report["is_valid"] is True
    assert report["total_events"] == 2
    assert len(report["errors"]) == 0


def test_admin_list_streams_and_create_checkpoint(db_session, populated_milestone_stream, admin_user):
    """Verifies admin listing of streams and manual signed checkpoint creation."""
    milestone_id, stream_id = populated_milestone_stream

    streams = list_impact_ledger_streams(
        current_user=admin_user,
        db=db_session
    )
    assert any(s.stream_id == stream_id for s in streams)

    checkpoint = create_ledger_checkpoint(
        stream_id=stream_id,
        current_user=admin_user,
        db=db_session
    )
    assert checkpoint.stream_id == stream_id
    assert checkpoint.sequence_number == 2
    assert checkpoint.signature is not None
