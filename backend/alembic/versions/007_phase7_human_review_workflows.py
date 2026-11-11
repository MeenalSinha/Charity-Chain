"""007_phase7_human_review_workflows

Revision ID: 007_phase7_human_review_workflows
Revises: 006_phase6_ai_verification
Create Date: 2026-11-12 01:15:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '007_phase7_human_review_workflows'
down_revision: Union[str, None] = '006_phase6_ai_verification'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Create review_cases table
    op.create_table(
        'review_cases',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('milestone_id', sa.String(length=36), sa.ForeignKey('milestones.id', ondelete='CASCADE'), nullable=False),
        sa.Column('project_id', sa.String(length=36), sa.ForeignKey('projects.id', ondelete='CASCADE'), nullable=False),
        sa.Column('organization_id', sa.String(length=36), sa.ForeignKey('organizations.id', ondelete='CASCADE'), nullable=False),
        sa.Column('submission_round', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('submitted_by_id', sa.String(length=36), sa.ForeignKey('users.id', ondelete='SET NULL'), nullable=True),
        sa.Column('status', sa.String(length=50), nullable=False, server_default='SUBMITTED'),
        sa.Column('final_decision', sa.String(length=50), nullable=True),
        sa.Column('reconciliation_result', sa.String(length=50), nullable=True),
        sa.Column('submission_notes', sa.Text(), nullable=True),
        sa.Column('snapshot_digest', sa.String(length=64), nullable=False, server_default=''),
        sa.Column('snapshot_data_json', sa.JSON(), nullable=True),
        sa.Column('review_policy_version', sa.String(length=32), nullable=False, server_default='v7.0.0'),
        sa.Column('revision_deadline', sa.DateTime(), nullable=True),
        sa.Column('appeal_deadline', sa.DateTime(), nullable=True),
        sa.Column('adjudicator_id', sa.String(length=36), sa.ForeignKey('users.id', ondelete='SET NULL'), nullable=True),
        sa.Column('adjudication_comments', sa.Text(), nullable=True),
        sa.Column('adjudicated_at', sa.DateTime(), nullable=True),
        sa.Column('finalized_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.func.now())
    )
    op.create_index('ix_review_cases_milestone_id', 'review_cases', ['milestone_id'])
    op.create_index('ix_review_cases_project_id', 'review_cases', ['project_id'])
    op.create_index('ix_review_cases_organization_id', 'review_cases', ['organization_id'])
    op.create_index('ix_review_cases_status', 'review_cases', ['status'])
    op.create_index('ix_review_cases_snapshot_digest', 'review_cases', ['snapshot_digest'])

    # 2. Create review_case_evidence table
    op.create_table(
        'review_case_evidence',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('review_case_id', sa.String(length=36), sa.ForeignKey('review_cases.id', ondelete='CASCADE'), nullable=False),
        sa.Column('evidence_file_id', sa.String(length=36), sa.ForeignKey('evidence_files.id', ondelete='CASCADE'), nullable=False),
        sa.Column('evidence_version', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('evidence_hash', sa.String(length=64), nullable=False, server_default=''),
        sa.Column('verification_report_id', sa.String(length=36), sa.ForeignKey('verification_reports.id', ondelete='SET NULL'), nullable=True),
        sa.Column('ai_pipeline_version', sa.String(length=32), nullable=True),
        sa.Column('evidence_role', sa.String(length=64), nullable=False, server_default='PRIMARY'),
        sa.Column('snapshot_timestamp', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.func.now())
    )
    op.create_index('ix_review_case_evidence_review_case_id', 'review_case_evidence', ['review_case_id'])
    op.create_index('ix_review_case_evidence_evidence_file_id', 'review_case_evidence', ['evidence_file_id'])

    # 3. Extend review_tasks table
    with op.batch_alter_table('review_tasks') as batch_op:
        batch_op.add_column(sa.Column('review_case_id', sa.String(length=36), sa.ForeignKey('review_cases.id', ondelete='CASCADE', name='fk_review_tasks_case_id'), nullable=True))
        batch_op.add_column(sa.Column('reviewer_slot', sa.Integer(), nullable=False, server_default='1'))
        batch_op.add_column(sa.Column('assignment_state', sa.String(length=50), nullable=False, server_default='ASSIGNED'))
        batch_op.add_column(sa.Column('assigned_at', sa.DateTime(), nullable=False, server_default=sa.func.now()))
        batch_op.add_column(sa.Column('accepted_at', sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column('due_at', sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column('decided_at', sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column('decline_reason', sa.Text(), nullable=True))
        batch_op.add_column(sa.Column('conflict_declared', sa.Boolean(), nullable=False, server_default='0'))
        batch_op.add_column(sa.Column('conflict_reason', sa.Text(), nullable=True))
        batch_op.add_column(sa.Column('reassigned_from_task_id', sa.String(length=36), nullable=True))
        batch_op.alter_column('evidence_file_id', existing_type=sa.String(length=36), nullable=True)
        batch_op.create_index('ix_review_tasks_review_case_id', ['review_case_id'])
        batch_op.create_index('ix_review_tasks_milestone_id', ['milestone_id'])
        batch_op.create_index('ix_review_tasks_assigned_reviewer_id', ['assigned_reviewer_id'])

    # 4. Extend review_decision_records table
    with op.batch_alter_table('review_decision_records') as batch_op:
        batch_op.add_column(sa.Column('review_case_id', sa.String(length=36), sa.ForeignKey('review_cases.id', ondelete='CASCADE', name='fk_review_decisions_case_id'), nullable=True))
        batch_op.add_column(sa.Column('structured_revision_data_json', sa.JSON(), nullable=True))
        batch_op.add_column(sa.Column('rejection_reason_category', sa.String(length=100), nullable=True))
        batch_op.add_column(sa.Column('evidence_snapshot_digest', sa.String(length=64), nullable=True))
        batch_op.add_column(sa.Column('ai_report_id', sa.String(length=36), nullable=True))
        batch_op.add_column(sa.Column('submission_round', sa.Integer(), nullable=False, server_default='1'))
        batch_op.add_column(sa.Column('is_superseded', sa.Boolean(), nullable=False, server_default='0'))
        batch_op.create_index('ix_review_decision_records_review_case_id', ['review_case_id'])
        batch_op.create_index('ix_review_decision_records_task_id', ['task_id'])
        batch_op.create_index('ix_review_decision_records_reviewer_id', ['reviewer_id'])

    # 5. Create review_disputes table
    op.create_table(
        'review_disputes',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('review_case_id', sa.String(length=36), sa.ForeignKey('review_cases.id', ondelete='CASCADE'), nullable=False),
        sa.Column('milestone_id', sa.String(length=36), sa.ForeignKey('milestones.id', ondelete='CASCADE'), nullable=False),
        sa.Column('project_id', sa.String(length=36), sa.ForeignKey('projects.id', ondelete='CASCADE'), nullable=False),
        sa.Column('organization_id', sa.String(length=36), sa.ForeignKey('organizations.id', ondelete='CASCADE'), nullable=False),
        sa.Column('original_decision_record_id', sa.String(length=36), nullable=True),
        sa.Column('raised_by_id', sa.String(length=36), sa.ForeignKey('users.id', ondelete='SET NULL'), nullable=True),
        sa.Column('status', sa.String(length=50), nullable=False, server_default='OPEN'),
        sa.Column('dispute_category', sa.String(length=100), nullable=False),
        sa.Column('dispute_reason', sa.Text(), nullable=False),
        sa.Column('supporting_evidence_json', sa.JSON(), nullable=True),
        sa.Column('assigned_adjudicator_id', sa.String(length=36), sa.ForeignKey('users.id', ondelete='SET NULL'), nullable=True),
        sa.Column('resolution', sa.String(length=50), nullable=True),
        sa.Column('resolution_explanation', sa.Text(), nullable=True),
        sa.Column('resolved_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.func.now())
    )
    op.create_index('ix_review_disputes_review_case_id', 'review_disputes', ['review_case_id'])
    op.create_index('ix_review_disputes_milestone_id', 'review_disputes', ['milestone_id'])
    op.create_index('ix_review_disputes_project_id', 'review_disputes', ['project_id'])
    op.create_index('ix_review_disputes_organization_id', 'review_disputes', ['organization_id'])
    op.create_index('ix_review_disputes_status', 'review_disputes', ['status'])

    # 6. Create review_dispute_messages table
    op.create_table(
        'review_dispute_messages',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('dispute_id', sa.String(length=36), sa.ForeignKey('review_disputes.id', ondelete='CASCADE'), nullable=False),
        sa.Column('sender_id', sa.String(length=36), sa.ForeignKey('users.id', ondelete='SET NULL'), nullable=True),
        sa.Column('sender_role', sa.String(length=50), nullable=False),
        sa.Column('message', sa.Text(), nullable=False),
        sa.Column('attachments_json', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.func.now())
    )
    op.create_index('ix_review_dispute_messages_dispute_id', 'review_dispute_messages', ['dispute_id'])

    # 7. Create review_audit_events table
    op.create_table(
        'review_audit_events',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('review_case_id', sa.String(length=36), nullable=True),
        sa.Column('milestone_id', sa.String(length=36), nullable=False),
        sa.Column('actor_id', sa.String(length=36), nullable=True),
        sa.Column('event_type', sa.String(length=100), nullable=False),
        sa.Column('details_json', sa.JSON(), nullable=True),
        sa.Column('previous_event_hash', sa.String(length=64), nullable=False),
        sa.Column('current_event_hash', sa.String(length=64), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.func.now())
    )
    op.create_index('ix_review_audit_events_review_case_id', 'review_audit_events', ['review_case_id'])
    op.create_index('ix_review_audit_events_milestone_id', 'review_audit_events', ['milestone_id'])
    op.create_index('ix_review_audit_events_actor_id', 'review_audit_events', ['actor_id'])
    op.create_index('ix_review_audit_events_event_type', 'review_audit_events', ['event_type'])
    op.create_index('ix_review_audit_events_current_event_hash', 'review_audit_events', ['current_event_hash'])


def downgrade() -> None:
    op.drop_table('review_audit_events')
    op.drop_table('review_dispute_messages')
    op.drop_table('review_disputes')

    with op.batch_alter_table('review_decision_records') as batch_op:
        batch_op.drop_index('ix_review_decision_records_reviewer_id')
        batch_op.drop_index('ix_review_decision_records_task_id')
        batch_op.drop_index('ix_review_decision_records_review_case_id')
        batch_op.drop_column('is_superseded')
        batch_op.drop_column('submission_round')
        batch_op.drop_column('ai_report_id')
        batch_op.drop_column('evidence_snapshot_digest')
        batch_op.drop_column('rejection_reason_category')
        batch_op.drop_column('structured_revision_data_json')
        batch_op.drop_column('review_case_id')

    with op.batch_alter_table('review_tasks') as batch_op:
        batch_op.drop_index('ix_review_tasks_assigned_reviewer_id')
        batch_op.drop_index('ix_review_tasks_milestone_id')
        batch_op.drop_index('ix_review_tasks_review_case_id')
        batch_op.drop_column('reassigned_from_task_id')
        batch_op.drop_column('conflict_reason')
        batch_op.drop_column('conflict_declared')
        batch_op.drop_column('decline_reason')
        batch_op.drop_column('decided_at')
        batch_op.drop_column('due_at')
        batch_op.drop_column('accepted_at')
        batch_op.drop_column('assigned_at')
        batch_op.drop_column('assignment_state')
        batch_op.drop_column('reviewer_slot')
        batch_op.drop_column('review_case_id')

    op.drop_table('review_case_evidence')
    op.drop_table('review_cases')
