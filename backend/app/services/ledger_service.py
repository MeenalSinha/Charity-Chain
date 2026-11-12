"""
ledger_service.py - Double-Entry Accounting Ledger Service
==========================================================
Enforces double-entry financial accounting across CharityChain:
- Strict invariant: Sum(Debits) == Sum(Credits) for every posted journal entry.
- Immutable postings: entries cannot be updated or deleted; corrections only via reversals.
- Automated balance tracking and trial balance generation.
- Standardized Chart of Accounts for donations, refunds, processor fees, and payouts.
"""

import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from sqlalchemy import select, func, and_
from sqlalchemy.orm import Session

from backend.app.core.constants import (
    LedgerAccountType, LedgerEntryType
)
from backend.app.core.exceptions import (
    ValidationException, NotFoundException, ConflictException
)
from backend.app.core.logging import get_logger
from backend.app.db.models.ledger import (
    LedgerAccount, JournalEntry, JournalLine
)
from backend.app.db.models.financial import (
    Donation, PaymentCapture, PaymentRefund, Disbursement
)

logger = get_logger("services.ledger")

# Standard Chart of Accounts
SYSTEM_ACCOUNTS = [
    {
        "account_number": "1000",
        "name": "Gateway Clearing - PayPal",
        "account_type": LedgerAccountType.ASSET,
        "currency": "USD",
        "description": "Cash and unsettled balances held at payment processor gateways."
    },
    {
        "account_number": "2000",
        "name": "Designated Project Funds Liability",
        "account_type": LedgerAccountType.LIABILITY,
        "currency": "USD",
        "description": "Fiduciary liability for donor contributions restricted to active milestones."
    },
    {
        "account_number": "2100",
        "name": "Committed Milestone Payouts Payable",
        "account_type": LedgerAccountType.LIABILITY,
        "currency": "USD",
        "description": "Authorized milestone disbursements queued or processing for NGO payout."
    },
    {
        "account_number": "4000",
        "name": "Platform Operations Revenue",
        "account_type": LedgerAccountType.REVENUE,
        "currency": "USD",
        "description": "General operating contributions and unrestricted platform revenues."
    },
    {
        "account_number": "5000",
        "name": "Payment Processing Fees Expense",
        "account_type": LedgerAccountType.EXPENSE,
        "currency": "USD",
        "description": "Direct processor charges incurred for payment captures and payout batches."
    }
]


class LedgerService:
    """Core domain service for double-entry financial accounting."""

    @classmethod
    def ensure_system_accounts(cls, db: Session) -> Dict[str, LedgerAccount]:
        """Initializes system Chart of Accounts idempotently."""
        account_map = {}
        for acc_spec in SYSTEM_ACCOUNTS:
            acc = db.execute(
                select(LedgerAccount).where(LedgerAccount.account_number == acc_spec["account_number"])
            ).scalar_one_or_none()

            if not acc:
                acc = LedgerAccount(
                    id=str(uuid.uuid4()),
                    account_number=acc_spec["account_number"],
                    name=acc_spec["name"],
                    account_type=acc_spec["account_type"],
                    currency=acc_spec["currency"],
                    description=acc_spec["description"],
                    is_active=True
                )
                db.add(acc)
                db.flush()
            account_map[acc.account_number] = acc
        db.commit()
        return account_map

    @classmethod
    def get_account_by_number(cls, db: Session, account_number: str) -> LedgerAccount:
        """Retrieves ledger account by standard account number, creating if missing."""
        acc = db.execute(
            select(LedgerAccount).where(LedgerAccount.account_number == account_number)
        ).scalar_one_or_none()
        if not acc:
            cls.ensure_system_accounts(db)
            acc = db.execute(
                select(LedgerAccount).where(LedgerAccount.account_number == account_number)
            ).scalar_one_or_none()
            if not acc:
                raise NotFoundException(f"Ledger account '{account_number}' not found.")
        return acc

    @classmethod
    def post_journal_entry(
        cls,
        db: Session,
        entry_type: LedgerEntryType,
        description: str,
        lines: List[Dict[str, Any]],
        reference_id: Optional[str] = None,
        reference_type: Optional[str] = None,
        actor_id: Optional[str] = None
    ) -> JournalEntry:
        """
        Creates and persists an immutable double-entry journal transaction.
        Enforces:
        1. At least two lines.
        2. Sum(Debits) == Sum(Credits).
        3. All amounts > 0.
        """
        if len(lines) < 2:
            raise ValidationException("Double-entry journal entries must contain at least two lines.")

        total_debits = 0
        total_credits = 0

        for idx, line in enumerate(lines):
            d = line.get("debit_minor_units", 0)
            c = line.get("credit_minor_units", 0)

            if d < 0 or c < 0:
                raise ValidationException(f"Line {idx}: Negative monetary amounts are not permitted in ledger.")
            if d == 0 and c == 0:
                raise ValidationException(f"Line {idx}: Journal line must specify non-zero debit or credit.")
            if d > 0 and c > 0:
                raise ValidationException(f"Line {idx}: Journal line cannot have both debit and credit.")

            total_debits += d
            total_credits += c

        if total_debits != total_credits:
            raise ValidationException(
                f"Double-entry ledger invariant violation: Total Debits ({total_debits}) != Total Credits ({total_credits})."
            )

        now = datetime.now(timezone.utc)
        entry_number = f"JE-{now.strftime('%Y%m%d%H%M%S')}-{str(uuid.uuid4())[:6].upper()}"

        entry = JournalEntry(
            id=str(uuid.uuid4()),
            entry_number=entry_number,
            entry_type=entry_type,
            reference_id=reference_id,
            reference_type=reference_type,
            description=description,
            posted_at=now,
            created_by_id=actor_id,
            is_reversed=False
        )
        db.add(entry)
        db.flush()

        for line in lines:
            account_id = line.get("account_id")
            if not account_id and line.get("account_number"):
                acc = cls.get_account_by_number(db, line["account_number"])
                account_id = acc.id

            if not account_id:
                raise ValidationException("Each journal line must specify account_id or account_number.")

            j_line = JournalLine(
                id=str(uuid.uuid4()),
                journal_entry_id=entry.id,
                account_id=account_id,
                debit_minor_units=line.get("debit_minor_units", 0),
                credit_minor_units=line.get("credit_minor_units", 0),
                currency=line.get("currency", "USD"),
                memo=line.get("memo")
            )
            db.add(j_line)

        db.commit()
        db.refresh(entry)

        logger.info(
            f"Posted balanced journal entry {entry.entry_number} ({entry_type.value}): {total_debits} minor units",
            extra={"entry_number": entry.entry_number, "total_minor_units": total_debits}
        )
        return entry

    # -------------------------------------------------------------------------
    # Specialized Event Postings
    # -------------------------------------------------------------------------

    @classmethod
    def record_donation_capture(
        cls,
        db: Session,
        donation: Donation,
        capture: PaymentCapture,
        processor_fee_minor_units: int = 0,
        actor_id: Optional[str] = None
    ) -> JournalEntry:
        """
        Posts journal entry for settled donation:
        Debit 1000 (Gateway Clearing) with capture amount
        Credit 2000 (Designated Project Funds Liability) with capture amount
        """
        lines = [
            {
                "account_number": "1000",
                "debit_minor_units": capture.amount_minor_units,
                "credit_minor_units": 0,
                "currency": capture.currency,
                "memo": f"Donation capture {capture.provider_capture_id}"
            },
            {
                "account_number": "2000",
                "debit_minor_units": 0,
                "credit_minor_units": capture.amount_minor_units,
                "currency": capture.currency,
                "memo": f"Designated funds for milestone {donation.milestone_id}"
            }
        ]

        if processor_fee_minor_units > 0:
            lines.extend([
                {
                    "account_number": "5000",
                    "debit_minor_units": processor_fee_minor_units,
                    "credit_minor_units": 0,
                    "currency": capture.currency,
                    "memo": f"Processing fee on capture {capture.provider_capture_id}"
                },
                {
                    "account_number": "1000",
                    "debit_minor_units": 0,
                    "credit_minor_units": processor_fee_minor_units,
                    "currency": capture.currency,
                    "memo": f"Gateway fee retention on capture {capture.provider_capture_id}"
                }
            ])

        return cls.post_journal_entry(
            db=db,
            entry_type=LedgerEntryType.DONATION_CAPTURE,
            description=f"Donation capture for milestone {donation.milestone_id}",
            lines=lines,
            reference_id=capture.id,
            reference_type="PaymentCapture",
            actor_id=actor_id
        )

    @classmethod
    def record_donation_refund(
        cls,
        db: Session,
        refund: PaymentRefund,
        actor_id: Optional[str] = None
    ) -> JournalEntry:
        """
        Posts journal entry for completed refund:
        Debit 2000 (Designated Project Funds Liability) with refund amount
        Credit 1000 (Gateway Clearing) with refund amount
        """
        lines = [
            {
                "account_number": "2000",
                "debit_minor_units": refund.amount_minor_units,
                "credit_minor_units": 0,
                "currency": refund.currency,
                "memo": f"Refund deduction: {refund.provider_refund_id}"
            },
            {
                "account_number": "1000",
                "debit_minor_units": 0,
                "credit_minor_units": refund.amount_minor_units,
                "currency": refund.currency,
                "memo": f"Gateway payout refund: {refund.provider_refund_id}"
            }
        ]

        return cls.post_journal_entry(
            db=db,
            entry_type=LedgerEntryType.DONATION_REFUND,
            description=f"Completed donation refund {refund.provider_refund_id}",
            lines=lines,
            reference_id=refund.id,
            reference_type="PaymentRefund",
            actor_id=actor_id
        )

    @classmethod
    def record_payout_commitment(
        cls,
        db: Session,
        disbursement: Disbursement,
        actor_id: Optional[str] = None
    ) -> JournalEntry:
        """
        Posts journal entry upon Maker-Checker approval of milestone disbursement:
        Debit 2000 (Designated Project Funds Liability) with disbursement amount
        Credit 2100 (Committed Milestone Payouts Payable) with disbursement amount
        """
        lines = [
            {
                "account_number": "2000",
                "debit_minor_units": disbursement.amount_minor_units,
                "credit_minor_units": 0,
                "currency": disbursement.currency,
                "memo": f"Disbursement approved for milestone {disbursement.milestone_id}"
            },
            {
                "account_number": "2100",
                "debit_minor_units": 0,
                "credit_minor_units": disbursement.amount_minor_units,
                "currency": disbursement.currency,
                "memo": f"Committed payout payable for NGO {disbursement.organization_id}"
            }
        ]

        return cls.post_journal_entry(
            db=db,
            entry_type=LedgerEntryType.PAYOUT_COMMITMENT,
            description=f"Authorized milestone payout commitment: {disbursement.id}",
            lines=lines,
            reference_id=disbursement.id,
            reference_type="Disbursement",
            actor_id=actor_id
        )

    @classmethod
    def record_payout_execution(
        cls,
        db: Session,
        disbursement: Disbursement,
        provider_fee_minor_units: int = 0,
        actor_id: Optional[str] = None
    ) -> JournalEntry:
        """
        Posts journal entry when outbound payout has succeeded:
        Debit 2100 (Committed Milestone Payouts Payable) with gross amount
        Credit 1000 (Gateway Clearing) with gross amount
        """
        lines = [
            {
                "account_number": "2100",
                "debit_minor_units": disbursement.amount_minor_units,
                "credit_minor_units": 0,
                "currency": disbursement.currency,
                "memo": f"Extinguish payout commitment: {disbursement.id}"
            },
            {
                "account_number": "1000",
                "debit_minor_units": 0,
                "credit_minor_units": disbursement.amount_minor_units,
                "currency": disbursement.currency,
                "memo": f"Disbursement executed via PayPal: {disbursement.provider_item_id or disbursement.id}"
            }
        ]

        if provider_fee_minor_units > 0:
            lines.extend([
                {
                    "account_number": "5000",
                    "debit_minor_units": provider_fee_minor_units,
                    "credit_minor_units": 0,
                    "currency": disbursement.currency,
                    "memo": f"Payout transaction fee on {disbursement.id}"
                },
                {
                    "account_number": "1000",
                    "debit_minor_units": 0,
                    "credit_minor_units": provider_fee_minor_units,
                    "currency": disbursement.currency,
                    "memo": f"Gateway fee retention for payout {disbursement.id}"
                }
            ])

        return cls.post_journal_entry(
            db=db,
            entry_type=LedgerEntryType.PAYOUT_EXECUTION,
            description=f"Outbound disbursement settled: {disbursement.id}",
            lines=lines,
            reference_id=disbursement.id,
            reference_type="Disbursement",
            actor_id=actor_id
        )

    @classmethod
    def record_payout_failure_reversal(
        cls,
        db: Session,
        disbursement: Disbursement,
        actor_id: Optional[str] = None
    ) -> JournalEntry:
        """
        Reverses payout commitment if outbound batch or item fails or is returned:
        Debit 2100 (Committed Milestone Payouts Payable) with disbursement amount
        Credit 2000 (Designated Project Funds Liability) with disbursement amount
        """
        lines = [
            {
                "account_number": "2100",
                "debit_minor_units": disbursement.amount_minor_units,
                "credit_minor_units": 0,
                "currency": disbursement.currency,
                "memo": f"Release commitment on failed payout: {disbursement.id}"
            },
            {
                "account_number": "2000",
                "debit_minor_units": 0,
                "credit_minor_units": disbursement.amount_minor_units,
                "currency": disbursement.currency,
                "memo": f"Restore designated funds for milestone {disbursement.milestone_id}"
            }
        ]

        return cls.post_journal_entry(
            db=db,
            entry_type=LedgerEntryType.PAYOUT_FAILURE_REVERSAL,
            description=f"Reversed failed payout commitment: {disbursement.id}",
            lines=lines,
            reference_id=disbursement.id,
            reference_type="Disbursement",
            actor_id=actor_id
        )

    @classmethod
    def reverse_journal_entry(
        cls,
        db: Session,
        entry_id: str,
        reason: str,
        actor_id: Optional[str] = None
    ) -> JournalEntry:
        """
        Creates an exact offsetting reversal entry for an existing journal entry.
        Marks original entry as reversed and links reversed_by_entry_id.
        """
        orig = db.execute(select(JournalEntry).where(JournalEntry.id == entry_id)).scalar_one_or_none()
        if not orig:
            raise NotFoundException(f"Journal entry '{entry_id}' not found.")
        if orig.is_reversed:
            raise ConflictException(f"Journal entry '{orig.entry_number}' has already been reversed.")

        # Invert debits and credits
        reversed_lines = []
        for line in orig.lines:
            reversed_lines.append({
                "account_id": line.account_id,
                "debit_minor_units": line.credit_minor_units,
                "credit_minor_units": line.debit_minor_units,
                "currency": line.currency,
                "memo": f"Reversal of {orig.entry_number}: {reason}"
            })

        reversal_entry = cls.post_journal_entry(
            db=db,
            entry_type=LedgerEntryType.MANUAL_ADJUSTMENT,
            description=f"Reversal of {orig.entry_number}: {reason}",
            lines=reversed_lines,
            reference_id=orig.id,
            reference_type="JournalEntry",
            actor_id=actor_id
        )

        orig.is_reversed = True
        orig.reversed_by_entry_id = reversal_entry.id
        db.commit()

        return reversal_entry

    # -------------------------------------------------------------------------
    # Reports & Balance Queries
    # -------------------------------------------------------------------------

    @classmethod
    def get_trial_balance(cls, db: Session) -> Dict[str, Any]:
        """
        Computes the complete Trial Balance across all active ledger accounts.
        Verifies total debits == total credits.
        """
        cls.ensure_system_accounts(db)
        accounts = db.execute(select(LedgerAccount).order_by(LedgerAccount.account_number)).scalars().all()

        trial_balance_rows = []
        sum_debits = 0
        sum_credits = 0

        for acc in accounts:
            totals = db.execute(
                select(
                    func.coalesce(func.sum(JournalLine.debit_minor_units), 0),
                    func.coalesce(func.sum(JournalLine.credit_minor_units), 0)
                ).where(JournalLine.account_id == acc.id)
            ).one()

            total_d, total_c = totals[0], totals[1]
            sum_debits += total_d
            sum_credits += total_c

            # Normal balance computation
            if acc.account_type in (LedgerAccountType.ASSET, LedgerAccountType.EXPENSE):
                net_balance = total_d - total_c
            else:
                net_balance = total_c - total_d

            trial_balance_rows.append({
                "account_id": acc.id,
                "account_number": acc.account_number,
                "account_name": acc.name,
                "account_type": acc.account_type.value,
                "currency": acc.currency,
                "total_debit_minor_units": total_d,
                "total_credit_minor_units": total_c,
                "net_balance_minor_units": net_balance
            })

        is_balanced = (sum_debits == sum_credits)

        return {
            "is_balanced": is_balanced,
            "total_debit_minor_units": sum_debits,
            "total_credit_minor_units": sum_credits,
            "discrepancy_minor_units": abs(sum_debits - sum_credits),
            "accounts": trial_balance_rows,
            "generated_at": datetime.now(timezone.utc).isoformat()
        }

    @classmethod
    def get_account_statement(
        cls,
        db: Session,
        account_number: str
    ) -> Dict[str, Any]:
        """Returns transactional ledger statement and running balance for a specific account."""
        acc = cls.get_account_by_number(db, account_number)
        lines = db.execute(
            select(JournalLine, JournalEntry)
            .join(JournalEntry, JournalLine.journal_entry_id == JournalEntry.id)
            .where(JournalLine.account_id == acc.id)
            .order_by(JournalEntry.posted_at.asc())
        ).all()

        statement_entries = []
        running_balance = 0

        for line, entry in lines:
            if acc.account_type in (LedgerAccountType.ASSET, LedgerAccountType.EXPENSE):
                running_balance += (line.debit_minor_units - line.credit_minor_units)
            else:
                running_balance += (line.credit_minor_units - line.debit_minor_units)

            statement_entries.append({
                "entry_number": entry.entry_number,
                "posted_at": entry.posted_at.isoformat() if entry.posted_at else None,
                "entry_type": entry.entry_type.value,
                "description": entry.description,
                "debit_minor_units": line.debit_minor_units,
                "credit_minor_units": line.credit_minor_units,
                "memo": line.memo,
                "running_balance_minor_units": running_balance
            })

        return {
            "account_number": acc.account_number,
            "account_name": acc.name,
            "account_type": acc.account_type.value,
            "currency": acc.currency,
            "current_balance_minor_units": running_balance,
            "entries_count": len(statement_entries),
            "entries": statement_entries
        }

