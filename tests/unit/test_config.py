"""
test_config.py - Unit tests for Config
"""

from charity_tracker import Config


def test_config_defaults():
    assert Config.TAMPER_THRESHOLD > 0
    assert Config.MIN_DETECTIONS >= 1
    assert Config.CONFIDENCE_THRESHOLD > 0
    assert 'image/jpeg' in Config.ALLOWED_MIME_TYPES
    assert 'image/png' in Config.ALLOWED_MIME_TYPES


def test_mask_sensitive():
    assert Config.mask_sensitive("DEMO_MODE") == "UNSET_OR_DEMO"
    assert Config.mask_sensitive("optional") == "UNSET_OR_DEMO"
    assert Config.mask_sensitive("") == "UNSET_OR_DEMO"
    assert Config.mask_sensitive(None) == "UNSET_OR_DEMO"
    masked = Config.mask_sensitive("sk-1234567890abcdef")
    assert masked.startswith("sk-1")
    assert masked.endswith("cdef")
    assert "..." in masked

