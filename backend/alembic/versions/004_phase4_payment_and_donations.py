"""004_phase4_payment_and_donations

Revision ID: 004_phase4_payment_and_donations
Revises: 003_phase3_org_and_projects
Create Date: 2026-11-11 19:10:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '004_phase4_payment_and_donations'
down_revision: Union[str, None] = '003_phase3_org_and_projects'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Update donations table with project association, donor attributes, and lifecycle fields
    with op.batch_alter_table('donations') as batch_op:
        batch_op.add_column(sa.Column('project_id', sa.String(length=36), nullable=True))
        batch_op.add_column(sa.Column('status', sa.String(length=50), nullable=False, server_default='CREATED'))
        batch_op.add_column(sa.Column('donor_name', sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column('donor_email', sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column('is_anonymous', sa.Boolean(), nullable=False, server_default='0'))
        batch_op.add_column(sa.Column('completed_at', sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column('cancelled_at', sa.DateTime(), nullable=True))
        batch_op.create_index('ix_donations_project_id', ['project_id'])
        batch_op.create_index('ix_donations_status', ['status'])

    # 2. Update payment_orders table with idempotency, approval URL, and failure details
    with op.batch_alter_table('payment_orders') as batch_op:
        batch_op.add_column(sa.Column('idempotency_key', sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column('approval_url', sa.String(length=512), nullable=True))
        batch_op.add_column(sa.Column('failure_code', sa.String(length=100), nullable=True))
        batch_op.add_column(sa.Column('failure_reason', sa.Text(), nullable=True))
        batch_op.create_index('ix_payment_orders_idempotency_key', ['idempotency_key'], unique=True)

    # 3. Update payment_captures table with direct order link and idempotency
    with op.batch_alter_table('payment_captures') as batch_op:
        batch_op.add_column(sa.Column('order_id', sa.String(length=36), nullable=True))
        batch_op.add_column(sa.Column('idempotency_key', sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column('captured_at', sa.DateTime(), nullable=True))
        batch_op.alter_column('authorization_id', existing_type=sa.String(length=36), nullable=True)
        batch_op.create_index('ix_payment_captures_order_id', ['order_id'])
        batch_op.create_index('ix_payment_captures_idempotency_key', ['idempotency_key'], unique=True)

    # 4. Update payment_refunds table with idempotency and timestamps
    with op.batch_alter_table('payment_refunds') as batch_op:
        batch_op.add_column(sa.Column('idempotency_key', sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column('refunded_at', sa.DateTime(), nullable=True))
        batch_op.create_index('ix_payment_refunds_idempotency_key', ['idempotency_key'], unique=True)

    # 5. Create webhook_events table for deduplication and audit trail
    op.create_table(
        'webhook_events',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('provider', sa.String(length=50), nullable=False, server_default='PAYPAL'),
        sa.Column('event_id', sa.String(length=255), nullable=False),
        sa.Column('event_type', sa.String(length=100), nullable=False),
        sa.Column('resource_id', sa.String(length=255), nullable=True),
        sa.Column('resource_type', sa.String(length=50), nullable=True),
        sa.Column('signature_verification_status', sa.String(length=50), nullable=False, server_default='PENDING'),
        sa.Column('processing_status', sa.String(length=50), nullable=False, server_default='PENDING'),
        sa.Column('payload', sa.Text(), nullable=True),
        sa.Column('retry_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('failure_reason', sa.Text(), nullable=True),
        sa.Column('processed_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
    )
    op.create_index('ix_webhook_events_event_id', 'webhook_events', ['event_id'], unique=True)
    op.create_index('ix_webhook_events_event_type', 'webhook_events', ['event_type'])
    op.create_index('ix_webhook_events_resource_id', 'webhook_events', ['resource_id'])
    op.create_index('ix_webhook_events_processing_status', 'webhook_events', ['processing_status'])


def downgrade() -> None:
    # 1. Drop webhook_events table
    op.drop_index('ix_webhook_events_processing_status', table_name='webhook_events')
    op.drop_index('ix_webhook_events_resource_id', table_name='webhook_events')
    op.drop_index('ix_webhook_events_event_type', table_name='webhook_events')
    op.drop_index('ix_webhook_events_event_id', table_name='webhook_events')
    op.drop_table('webhook_events')

    # 2. Revert payment_refunds table
    with op.batch_alter_table('payment_refunds') as batch_op:
        batch_op.drop_index('ix_payment_refunds_idempotency_key')
        batch_op.drop_column('refunded_at')
        batch_op.drop_column('idempotency_key')

    # 3. Revert payment_captures table
    with op.batch_alter_table('payment_captures') as batch_op:
        batch_op.drop_index('ix_payment_captures_idempotency_key')
        batch_op.drop_index('ix_payment_captures_order_id')
        batch_op.alter_column('authorization_id', existing_type=sa.String(length=36), nullable=False)
        batch_op.drop_column('captured_at')
        batch_op.drop_column('idempotency_key')
        batch_op.drop_column('order_id')

    # 4. Revert payment_orders table
    with op.batch_alter_table('payment_orders') as batch_op:
        batch_op.drop_index('ix_payment_orders_idempotency_key')
        batch_op.drop_column('failure_reason')
        batch_op.drop_column('failure_code')
        batch_op.drop_column('approval_url')
        batch_op.drop_column('idempotency_key')

    # 5. Revert donations table
    with op.batch_alter_table('donations') as batch_op:
        batch_op.drop_index('ix_donations_status')
        batch_op.drop_index('ix_donations_project_id')
        batch_op.drop_column('cancelled_at')
        batch_op.drop_column('completed_at')
        batch_op.drop_column('is_anonymous')
        batch_op.drop_column('donor_email')
        batch_op.drop_column('donor_name')
        batch_op.drop_column('status')
        batch_op.drop_column('project_id')
