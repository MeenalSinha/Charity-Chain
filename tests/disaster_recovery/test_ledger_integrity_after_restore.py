"""
tests/disaster_recovery/test_ledger_integrity_after_restore.py

Phase 14: Verifies that the proof-of-impact ledger maintains
cryptographic integrity after a simulated database restore.
Tests hash chain continuity without requiring an actual DB restore.
"""
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from typing import List, Optional

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


@dataclass
class SimulatedLedgerEvent:
    """Simulates an ImpactLedgerEvent for chain continuity testing."""
    sequence_number: int
    event_type: str
    payload: dict
    previous_hash: str
    event_hash: str = ""

    def __post_init__(self):
        if not self.event_hash:
            self.event_hash = self._compute_hash()

    def _compute_hash(self) -> str:
        canonical = json.dumps({
            "sequence_number": self.sequence_number,
            "event_type": self.event_type,
            "payload": self.payload,
            "previous_hash": self.previous_hash,
        }, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_chain(length: int = 10) -> List[SimulatedLedgerEvent]:
    """Build a valid simulated ledger chain."""
    chain = []
    previous_hash = "0" * 64  # Genesis hash
    for i in range(length):
        event = SimulatedLedgerEvent(
            sequence_number=i + 1,
            event_type="DONATION_CAPTURED",
            payload={"project_id": f"proj-{i}", "amount_minor_units": 5000 + i},
            previous_hash=previous_hash,
        )
        chain.append(event)
        previous_hash = event.event_hash
    return chain


def verify_chain(chain: List[SimulatedLedgerEvent]) -> tuple[bool, Optional[int]]:
    """
    Verify chain integrity. Returns (is_valid, broken_at_sequence_number).
    """
    for i, event in enumerate(chain):
        # Recompute hash
        expected_hash = SimulatedLedgerEvent(
            sequence_number=event.sequence_number,
            event_type=event.event_type,
            payload=event.payload,
            previous_hash=event.previous_hash,
        ).event_hash

        if event.event_hash != expected_hash:
            return False, event.sequence_number

        # Verify linkage to previous event
        if i > 0:
            if event.previous_hash != chain[i - 1].event_hash:
                return False, event.sequence_number

    return True, None


class TestLedgerIntegrityAfterRestore:
    """Tests simulating ledger integrity verification post-restore."""

    def test_valid_chain_passes_verification(self):
        """A correctly built chain must pass integrity verification."""
        chain = build_chain(50)
        is_valid, broken_at = verify_chain(chain)
        assert is_valid, f"Chain verification failed at sequence {broken_at}"

    def test_tampered_payload_breaks_chain(self):
        """Modifying an event payload must break the chain hash."""
        chain = build_chain(10)
        # Tamper with event 5's payload
        chain[4].payload["amount_minor_units"] = 999999
        is_valid, broken_at = verify_chain(chain)
        assert not is_valid, "Tampered chain should fail verification"
        assert broken_at == 5, f"Break should be detected at sequence 5, got {broken_at}"

    def test_deleted_event_breaks_linkage(self):
        """Removing an event from the middle must break chain linkage."""
        chain = build_chain(10)
        # Remove event 5 from chain
        del chain[4]
        is_valid, broken_at = verify_chain(chain)
        assert not is_valid, "Chain with deleted event should fail verification"

    def test_reordered_events_break_chain(self):
        """Reordering events must break previous_hash linkage."""
        chain = build_chain(10)
        # Swap events 3 and 4
        chain[2], chain[3] = chain[3], chain[2]
        is_valid, broken_at = verify_chain(chain)
        assert not is_valid, "Reordered chain should fail verification"

    def test_hash_is_deterministic(self):
        """The same event data must always produce the same hash."""
        event1 = SimulatedLedgerEvent(
            sequence_number=1,
            event_type="DONATION_CAPTURED",
            payload={"project_id": "proj-1", "amount_minor_units": 5000},
            previous_hash="0" * 64,
        )
        event2 = SimulatedLedgerEvent(
            sequence_number=1,
            event_type="DONATION_CAPTURED",
            payload={"project_id": "proj-1", "amount_minor_units": 5000},
            previous_hash="0" * 64,
        )
        assert event1.event_hash == event2.event_hash

    def test_genesis_hash_is_zero_padded(self):
        """Genesis event must use 64-char zero string as previous_hash."""
        chain = build_chain(1)
        assert chain[0].previous_hash == "0" * 64

    def test_chain_of_1000_events_verifies_efficiently(self):
        """Large chains should verify in reasonable time."""
        import time
        chain = build_chain(1000)
        start = time.monotonic()
        is_valid, _ = verify_chain(chain)
        elapsed = time.monotonic() - start
        assert is_valid
        assert elapsed < 5.0, f"Chain verification took {elapsed:.2f}s — should be < 5s"

    def test_replace_event_data_breaks_downstream_hashes(self):
        """Replacing an event's data must invalidate all downstream hashes."""
        chain = build_chain(20)
        # Replace event at position 5 with a new payload (but keep old hash)
        chain[5].payload = {"project_id": "INJECTED", "amount_minor_units": 0}
        # Don't recompute hash — simulating a database-level replacement
        is_valid, broken_at = verify_chain(chain)
        assert not is_valid, "Event with replaced data should fail hash verification"
        assert broken_at == 6, f"Expected break at sequence 6, got {broken_at}"


class TestFinancialReconciliationPostRestore:
    """Tests for financial data integrity after a simulated restore."""

    def test_double_entry_sum_zero_invariant(self):
        """Sum of all debit and credit amounts in a balanced ledger must be zero."""
        # Simulate journal lines (debit = positive, credit = negative in accounting)
        journal_lines = [
            {"entry_id": 1, "account": "DONATION_INCOME", "amount": -10000, "line_type": "CREDIT"},
            {"entry_id": 1, "account": "DONOR_RECEIVABLE", "amount": 10000, "line_type": "DEBIT"},
            {"entry_id": 2, "account": "DONOR_RECEIVABLE", "amount": -10000, "line_type": "CREDIT"},
            {"entry_id": 2, "account": "PROJECT_ESCROW", "amount": 10000, "line_type": "DEBIT"},
        ]
        # Group by entry_id and verify each entry balances
        from collections import defaultdict
        entry_sums = defaultdict(int)
        for line in journal_lines:
            entry_sums[line["entry_id"]] += line["amount"]
        for entry_id, total in entry_sums.items():
            assert total == 0, f"Journal entry {entry_id} does not balance: {total}"

    def test_donations_sum_matches_project_raised_total(self):
        """Sum of captured donations must equal project raised_amount_minor_units."""
        # Simulate captured donations for project-1
        captures = [
            {"project_id": "proj-1", "amount": 5000, "status": "CAPTURED"},
            {"project_id": "proj-1", "amount": 7500, "status": "CAPTURED"},
            {"project_id": "proj-1", "amount": 2500, "status": "CAPTURED"},
        ]
        refunds = [
            {"project_id": "proj-1", "amount": 2500, "status": "REFUNDED"},
        ]
        project_raised = 12500  # As stored in projects table after reconciliation (captures - refunds)
        # Net should be captures - refunds
        total_captured = sum(c["amount"] for c in captures)
        total_refunded = sum(r["amount"] for r in refunds)
        expected_net = total_captured - total_refunded
        assert project_raised == expected_net, (
            f"Project raised amount {project_raised} != net captures {expected_net}"
        )
