"""005_phase5_evidence_management

Revision ID: 005_phase5_evidence_management
Revises: 004_phase4_payment_and_donations
Create Date: 2026-11-11 21:30:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '005_phase5_evidence_management'
down_revision: Union[str, None] = '004_phase4_payment_and_donations'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Update evidence_files table with storage and multi-versioning columns
    with op.batch_alter_table('evidence_files') as batch_op:
        batch_op.add_column(sa.Column('original_filename', sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column('storage_key', sa.String(length=500), nullable=True))
        batch_op.add_column(sa.Column('storage_provider', sa.String(length=50), nullable=False, server_default='local'))
        batch_op.add_column(sa.Column('storage_version_id', sa.String(length=100), nullable=True))
        batch_op.add_column(sa.Column('detected_mime_type', sa.String(length=100), nullable=True))
        batch_op.add_column(sa.Column('version_number', sa.Integer(), nullable=False, server_default='1'))
        batch_op.add_column(sa.Column('parent_evidence_id', sa.String(length=36), nullable=True))
        batch_op.add_column(sa.Column('is_archived', sa.Boolean(), nullable=False, server_default='0'))
        batch_op.add_column(sa.Column('archived_at', sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column('quarantine_reason', sa.String(length=255), nullable=True))
        
        batch_op.create_index('ix_evidence_files_milestone_id', ['milestone_id'])
        batch_op.create_index('ix_evidence_files_parent_evidence_id', ['parent_evidence_id'])
        batch_op.create_index('ix_evidence_files_is_archived', ['is_archived'])

    # 2. Create evidence_audit_events table for cryptographic hash-chaining
    op.create_table(
        'evidence_audit_events',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('evidence_id', sa.String(length=36), sa.ForeignKey('evidence_files.id', ondelete='CASCADE'), nullable=False),
        sa.Column('actor_id', sa.String(length=36), sa.ForeignKey('users.id', ondelete='SET NULL'), nullable=True),
        sa.Column('event_type', sa.String(length=50), nullable=False),
        sa.Column('details_json', sa.JSON(), nullable=True),
        sa.Column('previous_event_hash', sa.String(length=64), nullable=False),
        sa.Column('current_event_hash', sa.String(length=64), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='1')
    )
    op.create_index('ix_evidence_audit_events_evidence_id', 'evidence_audit_events', ['evidence_id'])
    op.create_index('ix_evidence_audit_events_current_event_hash', 'evidence_audit_events', ['current_event_hash'])
    op.create_index('ix_evidence_audit_events_event_type', 'evidence_audit_events', ['event_type'])


def downgrade() -> None:
    # 1. Drop evidence_audit_events
    op.drop_index('ix_evidence_audit_events_event_type', table_name='evidence_audit_events')
    op.drop_index('ix_evidence_audit_events_current_event_hash', table_name='evidence_audit_events')
    op.drop_index('ix_evidence_audit_events_evidence_id', table_name='evidence_audit_events')
    op.drop_table('evidence_audit_events')

    # 2. Drop columns from evidence_files
    with op.batch_alter_table('evidence_files') as batch_op:
        batch_op.drop_index('ix_evidence_files_is_archived')
        batch_op.drop_index('ix_evidence_files_parent_evidence_id')
        batch_op.drop_index('ix_evidence_files_milestone_id')
        batch_op.drop_column('quarantine_reason')
        batch_op.drop_column('archived_at')
        batch_op.drop_column('is_archived')
        batch_op.drop_column('parent_evidence_id')
        batch_op.drop_column('version_number')
        batch_op.drop_column('detected_mime_type')
        batch_op.drop_column('storage_version_id')
        batch_op.drop_column('storage_provider')
        batch_op.drop_column('storage_key')
        batch_op.drop_column('original_filename')

