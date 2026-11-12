"""008_phase8_disbursement_and_financial_ledger

Revision ID: 008_phase8_disbursement_and_financial_ledger
Revises: 007_phase7_human_review_workflows
Create Date: 2026-11-12 12:35:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '008_phase8_disbursement_and_financial_ledger'
down_revision: Union[str, None] = '007_phase7_human_review_workflows'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Create payout_recipients table
    op.create_table(
        'payout_recipients',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('organization_id', sa.String(length=36), sa.ForeignKey('organizations.id', ondelete='CASCADE'), nullable=False),
        sa.Column('recipient_type', sa.String(length=50), nullable=False, server_default='PAYPAL_EMAIL'),
        sa.Column('payout_method', sa.String(length=50), nullable=False, server_default='PAYPAL_PAYOUTS'),
        sa.Column('encrypted_destination', sa.Text(), nullable=False),
        sa.Column('destination_masked', sa.String(length=128), nullable=False),
        sa.Column('destination_hash', sa.String(length=64), nullable=False),
        sa.Column('status', sa.String(length=50), nullable=False, server_default='PENDING_VERIFICATION'),
        sa.Column('verification_notes', sa.Text(), nullable=True),
        sa.Column('verified_by_id', sa.String(length=36), sa.ForeignKey('users.id', ondelete='SET NULL'), nullable=True),
        sa.Column('verified_at', sa.DateTime(), nullable=True),
        sa.Column('is_primary', sa.Boolean(), nullable=False, server_default='0'),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=True)
    )
    op.create_index('ix_payout_recipients_org_id', 'payout_recipients', ['organization_id'])
    op.create_index('ix_payout_recipients_dest_hash', 'payout_recipients', ['destination_hash'])
    op.create_index('ix_payout_recipients_status', 'payout_recipients', ['status'])

    # 2. Create payout_batches table
    op.create_table(
        'payout_batches',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('batch_reference_id', sa.String(length=255), unique=True, nullable=False),
        sa.Column('provider', sa.String(length=50), nullable=False, server_default='PAYPAL_PAYOUTS'),
        sa.Column('provider_batch_id', sa.String(length=255), unique=True, nullable=True),
        sa.Column('status', sa.String(length=50), nullable=False, server_default='REQUESTED'),
        sa.Column('currency', sa.String(length=3), nullable=False, server_default='USD'),
        sa.Column('total_items_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('total_amount_minor_units', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('total_fees_minor_units', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('submitted_by_id', sa.String(length=36), sa.ForeignKey('users.id', ondelete='SET NULL'), nullable=True),
        sa.Column('submitted_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('raw_response', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=True)
    )
    op.create_index('ix_payout_batches_ref_id', 'payout_batches', ['batch_reference_id'])
    op.create_index('ix_payout_batches_prov_batch_id', 'payout_batches', ['provider_batch_id'])
    op.create_index('ix_payout_batches_status', 'payout_batches', ['status'])

    # 3. Create ledger_accounts table
    op.create_table(
        'ledger_accounts',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('account_number', sa.String(length=32), unique=True, nullable=False),
        sa.Column('name', sa.String(length=128), nullable=False),
        sa.Column('account_type', sa.String(length=50), nullable=False),
        sa.Column('currency', sa.String(length=3), nullable=False, server_default='USD'),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='1'),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=True)
    )
    op.create_index('ix_ledger_accounts_acc_num', 'ledger_accounts', ['account_number'])
    op.create_index('ix_ledger_accounts_type', 'ledger_accounts', ['account_type'])

    # 4. Create journal_entries table
    op.create_table(
        'journal_entries',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('entry_number', sa.String(length=64), unique=True, nullable=False),
        sa.Column('entry_type', sa.String(length=50), nullable=False),
        sa.Column('reference_id', sa.String(length=255), nullable=True),
        sa.Column('reference_type', sa.String(length=64), nullable=True),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('posted_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('created_by_id', sa.String(length=36), sa.ForeignKey('users.id', ondelete='SET NULL'), nullable=True),
        sa.Column('is_reversed', sa.Boolean(), nullable=False, server_default='0'),
        sa.Column('reversed_by_entry_id', sa.String(length=36), sa.ForeignKey('journal_entries.id', ondelete='SET NULL'), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=True)
    )
    op.create_index('ix_journal_entries_num', 'journal_entries', ['entry_number'])
    op.create_index('ix_journal_entries_type', 'journal_entries', ['entry_type'])
    op.create_index('ix_journal_entries_ref_id', 'journal_entries', ['reference_id'])

    # 5. Create journal_lines table
    op.create_table(
        'journal_lines',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('journal_entry_id', sa.String(length=36), sa.ForeignKey('journal_entries.id', ondelete='CASCADE'), nullable=False),
        sa.Column('account_id', sa.String(length=36), sa.ForeignKey('ledger_accounts.id', ondelete='CASCADE'), nullable=False),
        sa.Column('debit_minor_units', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('credit_minor_units', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('currency', sa.String(length=3), nullable=False, server_default='USD'),
        sa.Column('memo', sa.String(length=255), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=True)
    )
    op.create_index('ix_journal_lines_entry_id', 'journal_lines', ['journal_entry_id'])
    op.create_index('ix_journal_lines_acc_id', 'journal_lines', ['account_id'])

    # 6. Alter disbursements table
    with op.batch_alter_table('disbursements') as batch_op:
        batch_op.add_column(sa.Column('review_case_id', sa.String(length=36), nullable=True))
        batch_op.add_column(sa.Column('recipient_id', sa.String(length=36), nullable=True))
        batch_op.add_column(sa.Column('provider', sa.String(length=50), nullable=False, server_default='PAYPAL_PAYOUTS'))
        batch_op.add_column(sa.Column('provider_batch_id', sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column('provider_item_id', sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column('idempotency_key', sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column('gross_amount_minor_units', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('provider_fee_minor_units', sa.Integer(), nullable=False, server_default='0'))
        batch_op.add_column(sa.Column('net_amount_minor_units', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('requested_by_id', sa.String(length=36), nullable=True))
        batch_op.add_column(sa.Column('approved_by_id', sa.String(length=36), nullable=True))
        batch_op.add_column(sa.Column('requested_at', sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column('approved_at', sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column('submitted_at', sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column('failure_reason', sa.Text(), nullable=True))
        batch_op.create_index('ix_disbursements_review_case_id', ['review_case_id'])
        batch_op.create_index('ix_disbursements_recipient_id', ['recipient_id'])
        batch_op.create_index('ix_disbursements_idempotency_key', ['idempotency_key'], unique=True)
        batch_op.create_index('ix_disbursements_prov_item_id', ['provider_item_id'])

    # 7. Create payout_approvals table
    op.create_table(
        'payout_approvals',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('disbursement_id', sa.String(length=36), sa.ForeignKey('disbursements.id', ondelete='CASCADE'), nullable=False),
        sa.Column('action', sa.String(length=32), nullable=False),
        sa.Column('actor_id', sa.String(length=36), sa.ForeignKey('users.id', ondelete='SET NULL'), nullable=False),
        sa.Column('actor_role', sa.String(length=32), nullable=False),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('decided_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=True)
    )
    op.create_index('ix_payout_approvals_disbursement_id', 'payout_approvals', ['disbursement_id'])

    # 8. Create reconciliation_runs table
    op.create_table(
        'reconciliation_runs',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('run_number', sa.String(length=64), unique=True, nullable=False),
        sa.Column('provider', sa.String(length=50), nullable=False, server_default='PAYPAL'),
        sa.Column('period_start', sa.DateTime(), nullable=False),
        sa.Column('period_end', sa.DateTime(), nullable=False),
        sa.Column('status', sa.String(length=50), nullable=False, server_default='PENDING'),
        sa.Column('total_records_analyzed', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('matched_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('discrepancy_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('total_local_minor_units', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('total_gateway_minor_units', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('executed_by_id', sa.String(length=36), sa.ForeignKey('users.id', ondelete='SET NULL'), nullable=True),
        sa.Column('notes', sa.Text(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=True)
    )
    op.create_index('ix_reconciliation_runs_num', 'reconciliation_runs', ['run_number'])
    op.create_index('ix_reconciliation_runs_status', 'reconciliation_runs', ['status'])

    # 9. Create reconciliation_items table
    op.create_table(
        'reconciliation_items',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('reconciliation_run_id', sa.String(length=36), sa.ForeignKey('reconciliation_runs.id', ondelete='CASCADE'), nullable=False),
        sa.Column('item_type', sa.String(length=32), nullable=False),
        sa.Column('reference_id', sa.String(length=255), nullable=True),
        sa.Column('provider_transaction_id', sa.String(length=255), nullable=True),
        sa.Column('local_amount_minor_units', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('gateway_amount_minor_units', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('local_status', sa.String(length=50), nullable=True),
        sa.Column('gateway_status', sa.String(length=50), nullable=True),
        sa.Column('reconciliation_status', sa.String(length=50), nullable=False, server_default='MATCHED'),
        sa.Column('discrepancy_type', sa.String(length=50), nullable=True),
        sa.Column('discrepancy_resolution', sa.String(length=50), nullable=True),
        sa.Column('resolution_notes', sa.Text(), nullable=True),
        sa.Column('resolved_by_id', sa.String(length=36), sa.ForeignKey('users.id', ondelete='SET NULL'), nullable=True),
        sa.Column('resolved_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=True)
    )
    op.create_index('ix_reconciliation_items_run_id', 'reconciliation_items', ['reconciliation_run_id'])
    op.create_index('ix_reconciliation_items_prov_tx', 'reconciliation_items', ['provider_transaction_id'])
    op.create_index('ix_reconciliation_items_status', 'reconciliation_items', ['reconciliation_status'])

    # 10. Create settlement_records table
    op.create_table(
        'settlement_records',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('provider', sa.String(length=50), nullable=False, server_default='PAYPAL'),
        sa.Column('settlement_batch_id', sa.String(length=255), nullable=True),
        sa.Column('transaction_id', sa.String(length=255), unique=True, nullable=False),
        sa.Column('transaction_type', sa.String(length=50), nullable=False),
        sa.Column('gross_amount_minor_units', sa.Integer(), nullable=False),
        sa.Column('fee_amount_minor_units', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('net_amount_minor_units', sa.Integer(), nullable=False),
        sa.Column('currency', sa.String(length=3), nullable=False, server_default='USD'),
        sa.Column('settled_at', sa.DateTime(), nullable=False),
        sa.Column('raw_data', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=True)
    )
    op.create_index('ix_settlement_records_tx_id', 'settlement_records', ['transaction_id'])

    # 11. Create processor_fee_records table
    op.create_table(
        'processor_fee_records',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('provider', sa.String(length=50), nullable=False, server_default='PAYPAL'),
        sa.Column('transaction_reference_id', sa.String(length=255), unique=True, nullable=False),
        sa.Column('fee_type', sa.String(length=50), nullable=False, server_default='TRANSACTION'),
        sa.Column('fee_minor_units', sa.Integer(), nullable=False),
        sa.Column('currency', sa.String(length=3), nullable=False, server_default='USD'),
        sa.Column('assessed_at', sa.DateTime(), nullable=False),
        sa.Column('journal_entry_id', sa.String(length=36), sa.ForeignKey('journal_entries.id', ondelete='SET NULL'), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=True)
    )
    op.create_index('ix_processor_fee_ref_id', 'processor_fee_records', ['transaction_reference_id'])


def downgrade() -> None:
    op.drop_table('processor_fee_records')
    op.drop_table('settlement_records')
    op.drop_table('reconciliation_items')
    op.drop_table('reconciliation_runs')
    op.drop_table('payout_approvals')

    with op.batch_alter_table('disbursements') as batch_op:
        batch_op.drop_index('ix_disbursements_prov_item_id')
        batch_op.drop_index('ix_disbursements_idempotency_key')
        batch_op.drop_index('ix_disbursements_recipient_id')
        batch_op.drop_index('ix_disbursements_review_case_id')
        batch_op.drop_column('failure_reason')
        batch_op.drop_column('submitted_at')
        batch_op.drop_column('approved_at')
        batch_op.drop_column('requested_at')
        batch_op.drop_column('approved_by_id')
        batch_op.drop_column('requested_by_id')
        batch_op.drop_column('net_amount_minor_units')
        batch_op.drop_column('provider_fee_minor_units')
        batch_op.drop_column('gross_amount_minor_units')
        batch_op.drop_column('idempotency_key')
        batch_op.drop_column('provider_item_id')
        batch_op.drop_column('provider_batch_id')
        batch_op.drop_column('provider')
        batch_op.drop_column('recipient_id')
        batch_op.drop_column('review_case_id')

    op.drop_table('journal_lines')
    op.drop_table('journal_entries')
    op.drop_table('ledger_accounts')
    op.drop_table('payout_batches')
    op.drop_table('payout_recipients')

