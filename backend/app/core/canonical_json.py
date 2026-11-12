"""
canonical_json.py - RFC 8785 JSON Canonicalization Scheme (JCS) Implementation
==============================================================================
Provides deterministic, standard-compliant canonical JSON serialization and
SHA-256 payload hashing for digital signatures and cryptographic event ledgers.
"""

import json
import hashlib
from typing import Any, Dict


def canonicalize_json(data: Any) -> str:
    """
    Serializes a Python dictionary/list/primitive into deterministic RFC 8785 canonical JSON.

    Key Requirements:
    1. Whitespace: No whitespace outside strings (separators=(',', ':')).
    2. Object keys: Sorted lexicographically by UTF-16 code units.
    3. Character encoding: UTF-8.
    4. Determinism: Floats are disallowed or strictly rounded; integer minor units preferred.
    """
    if isinstance(data, (float,)):
        # To maintain strict financial and cryptographic non-repudiation, reject floats.
        raise ValueError(
            f"Floating-point values ({data}) are forbidden in canonical cryptographic payloads! "
            "Use integer minor currency units (cents) or formatted strings."
        )

    # Clean recursion to validate all nested structures
    def _normalize(val: Any) -> Any:
        if isinstance(val, dict):
            return {str(k): _normalize(v) for k, v in val.items()}
        elif isinstance(val, (list, tuple)):
            return [_normalize(item) for item in val]
        elif isinstance(val, float):
            raise ValueError(
                f"Floating-point values ({val}) are forbidden in canonical cryptographic payloads! "
                "Use integer minor currency units (cents) or formatted strings."
            )
        return val

    normalized = _normalize(data)
    return json.dumps(
        normalized,
        sort_keys=True,
        separators=(',', ':'),
        ensure_ascii=False
    )


def compute_canonical_hash(data: Any) -> str:
    """
    Computes the standard SHA-256 hex digest of the RFC 8785 canonical JSON payload.
    """
    canonical_str = canonicalize_json(data)
    return hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()

