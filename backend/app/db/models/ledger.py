"""
ledger.py - Double-Entry Accounting Ledger SQLAlchemy Models
============================================================
Enforces double-entry bookkeeping for all financial movements across CharityChain:
- Invariant: Sum(Debits) == Sum(Credits) for every committed journal entry.
- Immutable postings: entries are never deleted or modified; adjustments only via reversals.
- Chart of Accounts covering Assets, Liabilities, Equity, Revenue, and Expenses.
"""

from datetime import datetime, timezone
from sqlalchemy import (
    Column, String, Integer, Text, Boolean, DateTime, ForeignKey,
    Enum as SQLEnum
)
from sqlalchemy.orm import relationship

from backend.app.db.base import Base
from backend.app.core.constants import LedgerAccountType, LedgerEntryType


class LedgerAccount(Base):
    """
    Chart of Accounts definition.
    Standardized account numbers:
    1000 - Assets (Gateway Clearing / Cash)
    2000 - Liabilities (Designated Project Funds)
    2100 - Liabilities (Committed Milestone Payouts Payable)
    4000 - Revenue (Platform Operations / Unrestricted)
    5000 - Expenses (Payment Processing Fees)
    """
    __tablename__ = "ledger_accounts"

    account_number = Column(String(32), unique=True, nullable=False, index=True)
    name = Column(String(128), nullable=False)
    account_type = Column(SQLEnum(LedgerAccountType), nullable=False, index=True)
    currency = Column(String(3), nullable=False, default="USD")
    description = Column(Text, nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)

    lines = relationship("JournalLine", back_populates="account")


class JournalEntry(Base):
    """
    Header for an atomic double-entry transaction.
    Must have at least two journal lines and balanced debits/credits.
    """
    __tablename__ = "journal_entries"

    entry_number = Column(String(64), unique=True, nullable=False, index=True)
    entry_type = Column(SQLEnum(LedgerEntryType), nullable=False, index=True)
    reference_id = Column(String(255), nullable=True, index=True)
    reference_type = Column(String(64), nullable=True)
    description = Column(Text, nullable=False)
    posted_at = Column(DateTime, nullable=False, default=lambda: datetime.now(timezone.utc))
    created_by_id = Column(String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    is_reversed = Column(Boolean, default=False, nullable=False)
    reversed_by_entry_id = Column(String(36), ForeignKey("journal_entries.id", ondelete="SET NULL"), nullable=True)

    lines = relationship("JournalLine", back_populates="journal_entry", cascade="all, delete-orphan")


class JournalLine(Base):
    """
    Individual debit or credit line within a journal entry.
    All monetary values represented in integer minor units (cents).
    """
    __tablename__ = "journal_lines"

    journal_entry_id = Column(String(36), ForeignKey("journal_entries.id", ondelete="CASCADE"), nullable=False, index=True)
    account_id = Column(String(36), ForeignKey("ledger_accounts.id", ondelete="CASCADE"), nullable=False, index=True)
    debit_minor_units = Column(Integer, default=0, nullable=False)
    credit_minor_units = Column(Integer, default=0, nullable=False)
    currency = Column(String(3), default="USD", nullable=False)
    memo = Column(String(255), nullable=True)

    journal_entry = relationship("JournalEntry", back_populates="lines")
    account = relationship("LedgerAccount", back_populates="lines")

