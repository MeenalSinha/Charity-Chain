"""
test_payout_batch.py - Multi-Item Outbound Payout Batch Integration Tests
========================================================================
Tests grouping multiple approved disbursements into a single provider batch,
handling partial item failures, fee attribution, and ledger integrity.
"""

import pytest
import uuid
from datetime import datetime, timezone
from backend.app.core.constants import (
    MilestoneStatus, ProjectStatus, OrganizationStatus,
    PayoutRecipientStatus, RecipientType, DisbursementStatus
)
from backend.app.db.models.projects import Project, Milestone
from backend.app.db.models.users import Organization
from backend.app.db.models.financial import (
    PayoutRecipient, Disbursement, PayoutBatch
)
from backend.app.services.disbursement_service import DisbursementService
from backend.app.services.ledger_service import LedgerService
from tests.mocks.payout_mock import MockPayoutProvider


@pytest.fixture
def multi_batch_setup(db_session):
    """Sets up an org with multiple approved disbursements."""
    org = Organization(id=str(uuid.uuid4()), name="Global Education", legal_name="Global Education Inc", status=OrganizationStatus.VERIFIED)
    recip = PayoutRecipient(
        id=str(uuid.uuid4()), organization_id=org.id, recipient_type=RecipientType.PAYPAL_EMAIL,
        encrypted_destination="mock_enc", destination_masked="p***t@education.org", destination_hash="h_edu",
        status=PayoutRecipientStatus.VERIFIED, is_primary=True
    )
    d1 = Disbursement(
        id=str(uuid.uuid4()), milestone_id="ms-b1", organization_id=org.id, recipient_id=recip.id,
        amount_minor_units=10000, currency="USD", status=DisbursementStatus.APPROVED
    )
    d2 = Disbursement(
        id=str(uuid.uuid4()), milestone_id="ms-b2", organization_id=org.id, recipient_id=recip.id,
        amount_minor_units=20000, currency="USD", status=DisbursementStatus.APPROVED
    )
    db_session.add_all([org, recip, d1, d2])
    db_session.commit()

    # Pre-post commitments for both
    LedgerService.record_payout_commitment(db_session, d1)
    LedgerService.record_payout_commitment(db_session, d2)
    db_session.commit()

    return org, recip, d1, d2


def test_multi_item_batch_partial_failure_handling(db_session, multi_batch_setup):
    """
    Tests submitting a 2-item batch where 1 item fails:
    - Item 1 succeeds -> settled in ledger.
    - Item 2 fails -> reversed back to designated funds.
    - Overall trial balance remains strictly balanced!
    """
    org, recip, d1, d2 = multi_batch_setup
    mock_provider = MockPayoutProvider()
    # Simulate d2 item failure
    mock_provider.simulate_item_failure_ids.add(d2.id)

    srv = DisbursementService(provider=mock_provider)

    batch = srv.create_and_submit_payout_batch(
        db=db_session,
        disbursement_ids=[d1.id, d2.id],
        submitter_id="approver-1"
    )

    db_session.refresh(d1)
    db_session.refresh(d2)

    assert d1.status == DisbursementStatus.SUCCEEDED
    assert d1.disbursed_at is not None

    assert d2.status == DisbursementStatus.FAILED
    assert d2.failure_reason is not None

    # Verify trial balance is still balanced after partial settlement + reversal
    tb = LedgerService.get_trial_balance(db_session)
    assert tb["is_balanced"] is True
    assert tb["discrepancy_minor_units"] == 0
