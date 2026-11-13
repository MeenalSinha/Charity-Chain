"""012_phase13_security_audit_and_observability

Revision ID: 012_phase13_security_audit_and_observability
Revises: 011_phase11_analytics_and_transparency
Create Date: 2026-11-13 18:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '012_phase13_security_audit_and_observability'
down_revision: Union[str, None] = '011_phase11_analytics_and_transparency'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create security_audit_events table
    op.create_table(
        'security_audit_events',
        sa.Column('id', sa.String(length=36), primary_key=True, nullable=False),
        sa.Column('event_id', sa.String(length=36), nullable=False),
        sa.Column('sequence_number', sa.Integer(), nullable=True),
        sa.Column('event_type', sa.String(length=64), nullable=False),
        sa.Column('severity', sa.String(length=16), nullable=False, server_default='INFO'),
        sa.Column('user_id', sa.String(length=36), nullable=True),
        sa.Column('actor_email', sa.String(length=255), nullable=True),
        sa.Column('ip_address', sa.String(length=64), nullable=True),
        sa.Column('user_agent', sa.String(length=255), nullable=True),
        sa.Column('resource_type', sa.String(length=64), nullable=True),
        sa.Column('resource_id', sa.String(length=64), nullable=True),
        sa.Column('action', sa.String(length=128), nullable=False),
        sa.Column('details', sa.JSON(), nullable=False),
        sa.Column('previous_hash', sa.String(length=64), nullable=False),
        sa.Column('event_hash', sa.String(length=64), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.UniqueConstraint('event_id')
    )

    op.create_index('ix_security_audit_events_event_id', 'security_audit_events', ['event_id'])
    op.create_index('ix_security_audit_events_event_type', 'security_audit_events', ['event_type'])
    op.create_index('ix_security_audit_events_severity', 'security_audit_events', ['severity'])
    op.create_index('ix_security_audit_events_user_id', 'security_audit_events', ['user_id'])
    op.create_index('ix_security_audit_events_event_hash', 'security_audit_events', ['event_hash'])
    op.create_index('ix_security_audit_events_created_at', 'security_audit_events', ['created_at'])
    op.create_index('ix_sec_audit_type_created', 'security_audit_events', ['event_type', 'created_at'])
    op.create_index('ix_sec_audit_user_created', 'security_audit_events', ['user_id', 'created_at'])
    op.create_index('ix_sec_audit_severity_created', 'security_audit_events', ['severity', 'created_at'])


def downgrade() -> None:
    op.drop_index('ix_sec_audit_severity_created', table_name='security_audit_events')
    op.drop_index('ix_sec_audit_user_created', table_name='security_audit_events')
    op.drop_index('ix_sec_audit_type_created', table_name='security_audit_events')
    op.drop_index('ix_security_audit_events_created_at', table_name='security_audit_events')
    op.drop_index('ix_security_audit_events_event_hash', table_name='security_audit_events')
    op.drop_index('ix_security_audit_events_user_id', table_name='security_audit_events')
    op.drop_index('ix_security_audit_events_severity', table_name='security_audit_events')
    op.drop_index('ix_security_audit_events_event_type', table_name='security_audit_events')
    op.drop_index('ix_security_audit_events_event_id', table_name='security_audit_events')
    op.drop_table('security_audit_events')
