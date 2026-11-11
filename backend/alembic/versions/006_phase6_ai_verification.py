"""006_phase6_ai_verification

Revision ID: 006_phase6_ai_verification
Revises: 005_phase5_evidence_management
Create Date: 2026-11-11 23:15:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '006_phase6_ai_verification'
down_revision: Union[str, None] = '005_phase5_evidence_management'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Update verification_reports table with multi-indicator, calibration, and versioning metadata
    with op.batch_alter_table('verification_reports') as batch_op:
        batch_op.add_column(sa.Column('evidence_version', sa.Integer(), nullable=False, server_default='1'))
        batch_op.add_column(sa.Column('evidence_hash', sa.String(length=64), nullable=False, server_default=''))
        batch_op.add_column(sa.Column('project_id', sa.String(length=36), nullable=True))
        batch_op.add_column(sa.Column('milestone_id', sa.String(length=36), nullable=True))
        batch_op.add_column(sa.Column('pipeline_version', sa.String(length=50), nullable=False, server_default='v6.0.0'))
        batch_op.add_column(sa.Column('overall_assessment', sa.String(length=50), nullable=False, server_default='NEEDS_FURTHER_REVIEW'))
        batch_op.add_column(sa.Column('confidence_score', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('calibration_status', sa.String(length=50), nullable=False, server_default='UNVALIDATED'))
        batch_op.add_column(sa.Column('processing_started_at', sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column('processing_completed_at', sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column('provider_name', sa.String(length=100), nullable=False, server_default='yolo_forensics'))
        batch_op.add_column(sa.Column('model_metadata_json', sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column('limitations_json', sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column('config_snapshot_json', sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column('error_details', sa.Text(), nullable=True))

        batch_op.create_index('ix_verification_reports_evidence_file_id', ['evidence_file_id'])
        batch_op.create_index('ix_verification_reports_project_id', ['project_id'])
        batch_op.create_index('ix_verification_reports_milestone_id', ['milestone_id'])
        batch_op.create_index('ix_verification_reports_overall_assessment', ['overall_assessment'])

    # 2. Update verification_check_records with metrics, model details, and artifact paths
    with op.batch_alter_table('verification_check_records') as batch_op:
        batch_op.add_column(sa.Column('technical_result_json', sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column('raw_metrics_json', sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column('model_name', sa.String(length=100), nullable=True))
        batch_op.add_column(sa.Column('model_version', sa.String(length=50), nullable=True))
        batch_op.add_column(sa.Column('duration_ms', sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column('diagnostic_artifact_path', sa.String(length=512), nullable=True))
        batch_op.add_column(sa.Column('error_message', sa.Text(), nullable=True))

        batch_op.create_index('ix_verification_check_records_report_id', ['report_id'])
        batch_op.create_index('ix_verification_check_records_check_type', ['check_type'])

    # 3. Create verification_jobs table for asynchronous background verification tasks
    op.create_table(
        'verification_jobs',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('evidence_file_id', sa.String(length=36), sa.ForeignKey('evidence_files.id', ondelete='CASCADE'), nullable=False),
        sa.Column('evidence_hash', sa.String(length=64), nullable=False),
        sa.Column('evidence_version', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('requested_by_id', sa.String(length=36), sa.ForeignKey('users.id', ondelete='SET NULL'), nullable=True),
        sa.Column('status', sa.String(length=50), nullable=False, server_default='PENDING'),
        sa.Column('attempt_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('max_attempts', sa.Integer(), nullable=False, server_default='3'),
        sa.Column('pipeline_version', sa.String(length=50), nullable=False, server_default='v6.0.0'),
        sa.Column('requested_checks_json', sa.JSON(), nullable=True),
        sa.Column('baseline_evidence_id', sa.String(length=36), sa.ForeignKey('evidence_files.id', ondelete='SET NULL'), nullable=True),
        sa.Column('report_id', sa.String(length=36), sa.ForeignKey('verification_reports.id', ondelete='SET NULL'), nullable=True),
        sa.Column('idempotency_key', sa.String(length=128), nullable=True, unique=True),
        sa.Column('error_category', sa.String(length=100), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('queued_at', sa.DateTime(), nullable=False),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='1')
    )
    op.create_index('ix_verification_jobs_evidence_file_id', 'verification_jobs', ['evidence_file_id'])
    op.create_index('ix_verification_jobs_status', 'verification_jobs', ['status'])
    op.create_index('ix_verification_jobs_idempotency_key', 'verification_jobs', ['idempotency_key'])


def downgrade() -> None:
    # 1. Drop verification_jobs table
    op.drop_index('ix_verification_jobs_idempotency_key', table_name='verification_jobs')
    op.drop_index('ix_verification_jobs_status', table_name='verification_jobs')
    op.drop_index('ix_verification_jobs_evidence_file_id', table_name='verification_jobs')
    op.drop_table('verification_jobs')

    # 2. Revert verification_check_records
    with op.batch_alter_table('verification_check_records') as batch_op:
        batch_op.drop_index('ix_verification_check_records_check_type')
        batch_op.drop_index('ix_verification_check_records_report_id')
        batch_op.drop_column('error_message')
        batch_op.drop_column('diagnostic_artifact_path')
        batch_op.drop_column('duration_ms')
        batch_op.drop_column('model_version')
        batch_op.drop_column('model_name')
        batch_op.drop_column('raw_metrics_json')
        batch_op.drop_column('technical_result_json')

    # 3. Revert verification_reports
    with op.batch_alter_table('verification_reports') as batch_op:
        batch_op.drop_index('ix_verification_reports_overall_assessment')
        batch_op.drop_index('ix_verification_reports_milestone_id')
        batch_op.drop_index('ix_verification_reports_project_id')
        batch_op.drop_index('ix_verification_reports_evidence_file_id')
        batch_op.drop_column('error_details')
        batch_op.drop_column('config_snapshot_json')
        batch_op.drop_column('limitations_json')
        batch_op.drop_column('model_metadata_json')
        batch_op.drop_column('provider_name')
        batch_op.drop_column('processing_completed_at')
        batch_op.drop_column('processing_started_at')
        batch_op.drop_column('calibration_status')
        batch_op.drop_column('confidence_score')
        batch_op.drop_column('overall_assessment')
        batch_op.drop_column('pipeline_version')
        batch_op.drop_column('milestone_id')
        batch_op.drop_column('project_id')
        batch_op.drop_column('evidence_hash')
        batch_op.drop_column('evidence_version')

