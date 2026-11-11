"""
test_workflows.py - End-to-End Workflow Integration Tests
"""

import pytest
from charity_tracker import DatabaseManager, EnhancedAIVerifier, EnhancedIPFSManager


def test_full_project_milestone_lifecycle(isolated_db, ai_verifier, sample_jpeg_bytes, sample_modified_jpeg_bytes):
    # 1. Register NGO Project Milestone
    milestone_id = isolated_db.create_milestone(
        owner="Amazon Reforestation Alliance",
        goal=5000000000000000000,  # 5 ETH
        description="Planting 2500 native saplings in Acre, Brazil",
        latitude=-9.9754,
        longitude=-67.8249
    )
    assert milestone_id == 1
    
    # 2. Donor Contributes
    isolated_db.add_donation(milestone_id, "Donor 1", 3000000000000000000)
    isolated_db.add_donation(milestone_id, "Donor 2", 2000000000000000000)
    
    m = isolated_db.get_milestone(milestone_id)
    assert m['raised'] == 5000000000000000000
    assert m['status'] == 'PENDING'
    
    # 3. Store Evidence
    cid, url, storage_mode = EnhancedIPFSManager.upload_to_web3storage(sample_modified_jpeg_bytes, "saplings.jpg")
    assert cid is not None
    
    # 4. Run Forensic & AI Verification
    verification_result = ai_verifier.run_parallel_verification(
        evidence_bytes=sample_modified_jpeg_bytes,
        before_bytes=sample_jpeg_bytes,
        expected_lat=-9.9754,
        expected_lon=-67.8249
    )
    assert 'checks' in verification_result
    
    # 5. Submit Evidence to Database
    isolated_db.submit_evidence(milestone_id, cid, verification_result)
    m_after_submit = isolated_db.get_milestone(milestone_id)
    assert m_after_submit['status'] == 'SUBMITTED'
    assert m_after_submit['evidence_cid'] == cid
    
    # 6. Human Reviewer Approves and Releases Milestone Funds
    isolated_db.update_milestone_status(milestone_id, 'RELEASED')
    m_final = isolated_db.get_milestone(milestone_id)
    assert m_final['status'] == 'RELEASED'
    
    # 7. Donors receive verifiable Proof-of-Impact certificates
    certs1 = isolated_db.get_donor_nfts("Donor 1")
    certs2 = isolated_db.get_donor_nfts("Donor 2")
    assert len(certs1) == 1
    assert len(certs2) == 1
    assert certs1[0]['milestone_id'] == milestone_id
    assert certs2[0]['milestone_id'] == milestone_id

