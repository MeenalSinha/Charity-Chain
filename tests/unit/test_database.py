"""
test_database.py - Database manager tests
"""

import pytest
from charity_tracker import DatabaseManager


def test_database_initialization(isolated_db):
    assert isolated_db.db is not None
    assert isolated_db.milestones is not None
    assert isolated_db.donations is not None
    assert isolated_db.verifications is not None
    assert isolated_db.ngo_profiles is not None
    assert isolated_db.nfts is not None
    assert isolated_db.analytics is not None
    assert isolated_db.get_all_milestones() == []


def test_create_and_retrieve_milestone(isolated_db):
    m_id = isolated_db.create_milestone(
        owner="Hope Foundation",
        goal=1000000000000000000,
        description="Build rural medical center",
        latitude=12.9716,
        longitude=77.5946
    )
    assert m_id == 1
    
    milestone = isolated_db.get_milestone(1)
    assert milestone is not None
    assert milestone['id'] == 1
    assert milestone['owner'] == "Hope Foundation"
    assert milestone['raised'] == 0
    assert milestone['status'] == 'PENDING'
    assert milestone['latitude'] == 12.9716
    assert milestone['longitude'] == 77.5946
    
    # Check NGO profile auto-created
    profile = isolated_db.get_ngo_profile("Hope Foundation")
    assert profile['address'] == "Hope Foundation"
    assert profile['reputation_score'] == 100


def test_add_donations_and_raised_update(isolated_db):
    m_id = isolated_db.create_milestone(
        owner="Clean Waters",
        goal=2000000000000000000,
        description="Well drilling",
        latitude=0.0,
        longitude=0.0
    )
    
    isolated_db.add_donation(m_id, "Donor A", 500000000000000000)
    isolated_db.add_donation(m_id, "Donor B", 300000000000000000)
    
    donations = isolated_db.get_donations_for_milestone(m_id)
    assert len(donations) == 2
    
    milestone = isolated_db.get_milestone(m_id)
    assert milestone['raised'] == 800000000000000000


def test_evidence_submission_and_analytics(isolated_db):
    m_id = isolated_db.create_milestone(
        owner="Trees for Tomorrow",
        goal=1000000000000000000,
        description="Planting trees",
        latitude=10.0,
        longitude=20.0
    )
    
    verification_result = {
        'timestamp': "2026-11-11T12:00:00",
        'checks': {
            'tamper': {'status': 'PASSED', 'passed': True, 'ela_score': 4.5},
            'gps': {'status': 'UNAVAILABLE', 'passed': False, 'message': 'No GPS in image'}
        },
        'pass': True,
        'confidence': 0.85
    }
    
    isolated_db.submit_evidence(m_id, "local-sha256-abc123", verification_result)
    
    milestone = isolated_db.get_milestone(m_id)
    assert milestone['status'] == 'SUBMITTED'
    assert milestone['evidence_cid'] == "local-sha256-abc123"
    assert milestone['verification_score'] == 0.85
    
    summary = isolated_db.get_analytics_summary()
    assert summary['total_verifications'] == 1
    assert summary['avg_confidence'] == 0.85


def test_milestone_release_and_certificate_issuance(isolated_db):
    m_id = isolated_db.create_milestone(
        owner="School Support",
        goal=1000000000000000000,
        description="School supplies",
        latitude=5.0,
        longitude=5.0
    )
    isolated_db.add_donation(m_id, "Supporter One", 500000000000000000)
    isolated_db.add_donation(m_id, "Supporter Two", 500000000000000000)
    
    # Update status to RELEASED
    isolated_db.update_milestone_status(m_id, 'RELEASED')
    
    updated = isolated_db.get_milestone(m_id)
    assert updated['status'] == 'RELEASED'
    
    # Check that certificates (Proof-of-Impact records) were issued
    certs_one = isolated_db.get_donor_nfts("Supporter One")
    certs_two = isolated_db.get_donor_nfts("Supporter Two")
    assert len(certs_one) == 1
    assert len(certs_two) == 1
    assert certs_one[0]['metadata']['record_type'] == 'LOCAL_DIGITAL_CERTIFICATE'

