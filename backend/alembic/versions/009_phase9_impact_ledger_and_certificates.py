"""009_phase9_impact_ledger_and_certificates

Revision ID: 009_phase9_impact_ledger_and_certificates
Revises: 008_phase8_disbursement_and_financial_ledger
Create Date: 2026-11-12 15:45:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '009_phase9_impact_ledger_and_certificates'
down_revision: Union[str, None] = '008_phase8_disbursement_and_financial_ledger'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Create impact_ledger_streams table
    op.create_table(
        'impact_ledger_streams',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('stream_id', sa.String(length=128), unique=True, nullable=False),
        sa.Column('aggregate_type', sa.String(length=50), nullable=False),
        sa.Column('aggregate_id', sa.String(length=64), nullable=False),
        sa.Column('current_sequence_number', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('head_event_hash', sa.String(length=64), nullable=False, server_default='0000000000000000000000000000000000000000000000000000000000000000'),
        sa.Column('status', sa.String(length=32), nullable=False, server_default='ACTIVE'),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=True)
    )
    op.create_index('ix_impact_ledger_streams_stream_id', 'impact_ledger_streams', ['stream_id'])
    op.create_index('ix_impact_ledger_streams_agg_type', 'impact_ledger_streams', ['aggregate_type'])
    op.create_index('ix_impact_ledger_streams_agg_id', 'impact_ledger_streams', ['aggregate_id'])

    # 2. Create impact_ledger_events table
    op.create_table(
        'impact_ledger_events',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('stream_id', sa.String(length=128), sa.ForeignKey('impact_ledger_streams.stream_id', ondelete='RESTRICT'), nullable=False),
        sa.Column('sequence_number', sa.Integer(), nullable=False),
        sa.Column('event_type', sa.String(length=64), nullable=False),
        sa.Column('schema_version', sa.String(length=20), nullable=False, server_default='v9.0.0'),
        sa.Column('source_event_type', sa.String(length=64), nullable=False),
        sa.Column('source_event_id', sa.String(length=128), nullable=False),
        sa.Column('payload_json', sa.Text(), nullable=False),
        sa.Column('payload_hash', sa.String(length=64), nullable=False),
        sa.Column('previous_event_hash', sa.String(length=64), nullable=False),
        sa.Column('current_event_hash', sa.String(length=64), nullable=False),
        sa.Column('actor_id', sa.String(length=36), sa.ForeignKey('users.id', ondelete='SET NULL'), nullable=True),
        sa.Column('timestamp', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=True)
    )
    op.create_index('ix_impact_ledger_events_stream_id', 'impact_ledger_events', ['stream_id'])
    op.create_index('ix_impact_ledger_events_seq', 'impact_ledger_events', ['stream_id', 'sequence_number'], unique=True)
    op.create_index('ix_impact_ledger_events_type', 'impact_ledger_events', ['event_type'])
    op.create_index('ix_impact_ledger_events_payload_hash', 'impact_ledger_events', ['payload_hash'])
    op.create_index('ix_impact_ledger_events_curr_hash', 'impact_ledger_events', ['current_event_hash'])

    # 3. Create impact_ledger_checkpoints table
    op.create_table(
        'impact_ledger_checkpoints',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('stream_id', sa.String(length=128), nullable=False),
        sa.Column('sequence_number', sa.Integer(), nullable=False),
        sa.Column('head_hash', sa.String(length=64), nullable=False),
        sa.Column('signing_key_id', sa.String(length=64), nullable=False),
        sa.Column('signature_algorithm', sa.String(length=32), nullable=False, server_default='Ed25519'),
        sa.Column('signature', sa.Text(), nullable=False),
        sa.Column('checkpoint_metadata_json', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=True)
    )
    op.create_index('ix_impact_ledger_checkpoints_stream_id', 'impact_ledger_checkpoints', ['stream_id'])

    # 4. Create issuer_signing_keys table
    op.create_table(
        'issuer_signing_keys',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('key_id', sa.String(length=64), unique=True, nullable=False),
        sa.Column('algorithm', sa.String(length=32), nullable=False, server_default='Ed25519'),
        sa.Column('public_key_pem', sa.Text(), nullable=False),
        sa.Column('public_key_jwk', sa.Text(), nullable=False),
        sa.Column('status', sa.String(length=32), nullable=False, server_default='ACTIVE'),
        sa.Column('usage', sa.String(length=32), nullable=False, server_default='SIG'),
        sa.Column('activated_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('retired_at', sa.DateTime(), nullable=True),
        sa.Column('revoked_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=True)
    )
    op.create_index('ix_issuer_signing_keys_key_id', 'issuer_signing_keys', ['key_id'])
    op.create_index('ix_issuer_signing_keys_status', 'issuer_signing_keys', ['status'])

    # 5. Extend impact_certificates table
    with op.batch_alter_table('impact_certificates') as batch_op:
        batch_op.add_column(sa.Column('project_id', sa.String(length=36), nullable=True))
        batch_op.add_column(sa.Column('organization_id', sa.String(length=36), nullable=True))
        batch_op.add_column(sa.Column('review_case_id', sa.String(length=36), nullable=True))
        batch_op.add_column(sa.Column('disbursement_id', sa.String(length=36), nullable=True))
        batch_op.add_column(sa.Column('verification_id', sa.String(length=64), nullable=True))
        batch_op.add_column(sa.Column('idempotency_key', sa.String(length=128), nullable=True))
        batch_op.add_column(sa.Column('certificate_type', sa.String(length=50), nullable=False, server_default='CONTRIBUTION_IMPACT'))
        batch_op.add_column(sa.Column('schema_version', sa.String(length=20), nullable=False, server_default='v9.0.0'))
        batch_op.add_column(sa.Column('public_attribution', sa.Boolean(), nullable=False, server_default='0'))
        batch_op.add_column(sa.Column('public_display_name', sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column('canonical_payload', sa.Text(), nullable=False, server_default=''))
        batch_op.add_column(sa.Column('payload_hash', sa.String(length=64), nullable=False, server_default=''))
        batch_op.add_column(sa.Column('signature', sa.Text(), nullable=False, server_default=''))
        batch_op.add_column(sa.Column('signature_algorithm', sa.String(length=32), nullable=False, server_default='Ed25519'))
        batch_op.add_column(sa.Column('signing_key_id', sa.String(length=64), nullable=False, server_default='key-charitychain-2026-v1'))
        batch_op.add_column(sa.Column('issuer_id', sa.String(length=128), nullable=False, server_default='charitychain-core-v1'))
        batch_op.add_column(sa.Column('status', sa.String(length=32), nullable=False, server_default='ISSUED'))
        batch_op.add_column(sa.Column('revocation_reason', sa.String(length=100), nullable=True))
        batch_op.add_column(sa.Column('revoked_at', sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column('superseded_by_id', sa.String(length=36), nullable=True))
        batch_op.add_column(sa.Column('pdf_storage_key', sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column('pdf_hash', sa.String(length=64), nullable=True))
        batch_op.add_column(sa.Column('qr_code_url', sa.String(length=512), nullable=True))

        batch_op.create_unique_constraint('uq_impact_certificates_verification_id', ['verification_id'])
        batch_op.create_unique_constraint('uq_impact_certificates_idempotency_key', ['idempotency_key'])
        batch_op.create_foreign_key('fk_impact_certificates_project', 'projects', ['project_id'], ['id'], ondelete='RESTRICT')
        batch_op.create_foreign_key('fk_impact_certificates_organization', 'organizations', ['organization_id'], ['id'], ondelete='RESTRICT')
        batch_op.create_foreign_key('fk_impact_certificates_review_case', 'review_cases', ['review_case_id'], ['id'], ondelete='SET NULL')
        batch_op.create_foreign_key('fk_impact_certificates_disbursement', 'disbursements', ['disbursement_id'], ['id'], ondelete='SET NULL')
        batch_op.create_foreign_key('fk_impact_certificates_superseded_by', 'impact_certificates', ['superseded_by_id'], ['id'], ondelete='SET NULL')

        batch_op.create_index('ix_impact_certificates_verification_id', ['verification_id'])
        batch_op.create_index('ix_impact_certificates_idempotency_key', ['idempotency_key'])
        batch_op.create_index('ix_impact_certificates_type', ['certificate_type'])
        batch_op.create_index('ix_impact_certificates_payload_hash', ['payload_hash'])
        batch_op.create_index('ix_impact_certificates_status', ['status'])

    # 6. Create certificate_status_events table
    op.create_table(
        'certificate_status_events',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('certificate_id', sa.String(length=36), sa.ForeignKey('impact_certificates.id', ondelete='CASCADE'), nullable=False),
        sa.Column('previous_status', sa.String(length=32), nullable=False),
        sa.Column('new_status', sa.String(length=32), nullable=False),
        sa.Column('reason_code', sa.String(length=100), nullable=False),
        sa.Column('reason_description', sa.Text(), nullable=True),
        sa.Column('actor_id', sa.String(length=36), sa.ForeignKey('users.id', ondelete='SET NULL'), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=True)
    )
    op.create_index('ix_cert_status_events_cert_id', 'certificate_status_events', ['certificate_id'])


def downgrade() -> None:
    op.drop_table('certificate_status_events')

    with op.batch_alter_table('impact_certificates') as batch_op:
        batch_op.drop_index('ix_impact_certificates_status')
        batch_op.drop_index('ix_impact_certificates_payload_hash')
        batch_op.drop_index('ix_impact_certificates_type')
        batch_op.drop_index('ix_impact_certificates_idempotency_key')
        batch_op.drop_index('ix_impact_certificates_verification_id')

        batch_op.drop_constraint('fk_impact_certificates_superseded_by', type_='foreignkey')
        batch_op.drop_constraint('fk_impact_certificates_disbursement', type_='foreignkey')
        batch_op.drop_constraint('fk_impact_certificates_review_case', type_='foreignkey')
        batch_op.drop_constraint('fk_impact_certificates_organization', type_='foreignkey')
        batch_op.drop_constraint('fk_impact_certificates_project', type_='foreignkey')
        batch_op.drop_constraint('uq_impact_certificates_idempotency_key', type_='unique')
        batch_op.drop_constraint('uq_impact_certificates_verification_id', type_='unique')

        batch_op.drop_column('qr_code_url')
        batch_op.drop_column('pdf_hash')
        batch_op.drop_column('pdf_storage_key')
        batch_op.drop_column('superseded_by_id')
        batch_op.drop_column('revoked_at')
        batch_op.drop_column('revocation_reason')
        batch_op.drop_column('status')
        batch_op.drop_column('issuer_id')
        batch_op.drop_column('signing_key_id')
        batch_op.drop_column('signature_algorithm')
        batch_op.drop_column('signature')
        batch_op.drop_column('payload_hash')
        batch_op.drop_column('canonical_payload')
        batch_op.drop_column('public_display_name')
        batch_op.drop_column('public_attribution')
        batch_op.drop_column('schema_version')
        batch_op.drop_column('certificate_type')
        batch_op.drop_column('idempotency_key')
        batch_op.drop_column('verification_id')
        batch_op.drop_column('disbursement_id')
        batch_op.drop_column('review_case_id')
        batch_op.drop_column('organization_id')
        batch_op.drop_column('project_id')

    op.drop_table('issuer_signing_keys')
    op.drop_table('impact_ledger_checkpoints')
    op.drop_table('impact_ledger_events')
    op.drop_table('impact_ledger_streams')
