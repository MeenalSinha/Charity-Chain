"""
test_ipfs_certificates.py - Storage & Proof-of-Impact Certificate Tests
"""

import pytest
from charity_tracker import EnhancedIPFSManager, AIAssistant, Config


def test_ipfs_manager_local_storage_fallback(sample_jpeg_bytes):
    cid, url, mode = EnhancedIPFSManager.upload_to_web3storage(sample_jpeg_bytes, "test.jpg")
    assert cid.startswith("local-sha256-")
    assert mode == "LOCAL_STORAGE"
    assert "local-evidence://" in url


def test_qr_code_generation():
    qr_bytes = EnhancedIPFSManager.generate_qr_code("sample-cid-12345")
    if qr_bytes is not None:
        assert len(qr_bytes) > 0
        assert qr_bytes[:8] == b'\x89PNG\r\n\x1a\n'


def test_ai_assistant_unconfigured_mode():
    assistant = AIAssistant(api_key="DEMO_MODE")
    assert assistant.available is False
    
    response = assistant.query("Is this project verified?", {"confidence": 80, "status": "PENDING"})
    assert "Rule-Based Status" in response
    assert "Live LLM inference provider not configured" in response

