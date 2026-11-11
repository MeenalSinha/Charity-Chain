"""001_initial_schema

Revision ID: 001_initial_schema
Revises: 
Create Date: 2026-11-11 14:00:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '001_initial_schema'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Users & Profiles
    op.create_table(
        'users',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('email', sa.String(length=255), unique=True, nullable=False),
        sa.Column('hashed_password', sa.String(length=255), nullable=False),
        sa.Column('full_name', sa.String(length=255), nullable=False),
        sa.Column('role', sa.String(length=50), nullable=False, server_default='DONOR'),
        sa.Column('is_verified', sa.Boolean(), nullable=False, server_default='0'),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='1')
    )
    op.create_index('ix_users_email', 'users', ['email'])

    op.create_table(
        'user_profiles',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('user_id', sa.String(length=36), sa.ForeignKey('users.id', ondelete='CASCADE'), unique=True, nullable=False),
        sa.Column('bio', sa.Text(), nullable=True),
        sa.Column('phone_number', sa.String(length=32), nullable=True),
        sa.Column('avatar_url', sa.String(length=512), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='1')
    )

    # Organizations & NGO Profiles
    op.create_table(
        'organizations',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('name', sa.String(length=255), unique=True, nullable=False),
        sa.Column('registration_number', sa.String(length=100), unique=True, nullable=True),
        sa.Column('tax_id', sa.String(length=100), nullable=True),
        sa.Column('status', sa.String(length=50), nullable=False, server_default='PENDING'),
        sa.Column('website', sa.String(length=255), nullable=True),
        sa.Column('country', sa.String(length=3), nullable=False, server_default='USA'),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='1')
    )

    op.create_table(
        'organization_memberships',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('user_id', sa.String(length=36), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('organization_id', sa.String(length=36), sa.ForeignKey('organizations.id', ondelete='CASCADE'), nullable=False),
        sa.Column('role', sa.String(length=50), nullable=False, server_default='NGO_MEMBER'),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='1')
    )

    op.create_table(
        'ngo_profiles',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('organization_id', sa.String(length=36), sa.ForeignKey('organizations.id', ondelete='CASCADE'), unique=True, nullable=False),
        sa.Column('mission_statement', sa.Text(), nullable=True),
        sa.Column('reputation_score', sa.Integer(), nullable=False, server_default='100'),
        sa.Column('completed_milestones', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('rejected_milestones', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('total_raised_minor_units', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='1')
    )

    # Categories, Projects & Milestones
    op.create_table(
        'project_categories',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('name', sa.String(length=100), unique=True, nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='1')
    )

    op.create_table(
        'projects',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('organization_id', sa.String(length=36), sa.ForeignKey('organizations.id', ondelete='CASCADE'), nullable=False),
        sa.Column('category_id', sa.String(length=36), sa.ForeignKey('project_categories.id'), nullable=True),
        sa.Column('title', sa.String(length=255), nullable=False),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('status', sa.String(length=50), nullable=False, server_default='DRAFT'),
        sa.Column('target_amount_minor_units', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('currency', sa.String(length=3), nullable=False, server_default='USD'),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='1')
    )

    op.create_table(
        'milestones',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('project_id', sa.String(length=36), sa.ForeignKey('projects.id', ondelete='CASCADE'), nullable=False),
        sa.Column('title', sa.String(length=255), nullable=False),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('sequence_number', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('target_amount_minor_units', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('raised_amount_minor_units', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('currency', sa.String(length=3), nullable=False, server_default='USD'),
        sa.Column('status', sa.String(length=50), nullable=False, server_default='DRAFT'),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='1')
    )

    op.create_table(
        'project_locations',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('milestone_id', sa.String(length=36), sa.ForeignKey('milestones.id', ondelete='CASCADE'), unique=True, nullable=False),
        sa.Column('latitude', sa.Float(), nullable=False),
        sa.Column('longitude', sa.Float(), nullable=False),
        sa.Column('address', sa.String(length=255), nullable=True),
        sa.Column('city', sa.String(length=100), nullable=True),
        sa.Column('country', sa.String(length=3), nullable=True),
        sa.Column('radius_meters', sa.Float(), nullable=False, server_default='1000.0'),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='1')
    )

    # Financial & Donations
    op.create_table(
        'donations',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('donor_id', sa.String(length=36), sa.ForeignKey('users.id', ondelete='SET NULL'), nullable=True),
        sa.Column('milestone_id', sa.String(length=36), sa.ForeignKey('milestones.id', ondelete='CASCADE'), nullable=False),
        sa.Column('amount_minor_units', sa.Integer(), nullable=False),
        sa.Column('currency', sa.String(length=3), nullable=False, server_default='USD'),
        sa.Column('impact_tokens_earned', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='1')
    )

    op.create_table(
        'payment_orders',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('donation_id', sa.String(length=36), sa.ForeignKey('donations.id', ondelete='CASCADE'), nullable=False),
        sa.Column('provider', sa.String(length=50), nullable=False, server_default='PAYPAL'),
        sa.Column('provider_order_id', sa.String(length=255), unique=True, nullable=True),
        sa.Column('amount_minor_units', sa.Integer(), nullable=False),
        sa.Column('currency', sa.String(length=3), nullable=False, server_default='USD'),
        sa.Column('status', sa.String(length=50), nullable=False, server_default='CREATED'),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='1')
    )

    op.create_table(
        'payment_authorizations',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('order_id', sa.String(length=36), sa.ForeignKey('payment_orders.id', ondelete='CASCADE'), nullable=False),
        sa.Column('provider_authorization_id', sa.String(length=255), unique=True, nullable=True),
        sa.Column('amount_minor_units', sa.Integer(), nullable=False),
        sa.Column('currency', sa.String(length=3), nullable=False, server_default='USD'),
        sa.Column('status', sa.String(length=50), nullable=False, server_default='CREATED'),
        sa.Column('expires_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='1')
    )

    op.create_table(
        'payment_captures',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('authorization_id', sa.String(length=36), sa.ForeignKey('payment_authorizations.id', ondelete='CASCADE'), nullable=False),
        sa.Column('provider_capture_id', sa.String(length=255), unique=True, nullable=True),
        sa.Column('amount_minor_units', sa.Integer(), nullable=False),
        sa.Column('currency', sa.String(length=3), nullable=False, server_default='USD'),
        sa.Column('status', sa.String(length=50), nullable=False, server_default='PENDING'),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='1')
    )

    op.create_table(
        'payment_refunds',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('capture_id', sa.String(length=36), sa.ForeignKey('payment_captures.id', ondelete='CASCADE'), nullable=False),
        sa.Column('provider_refund_id', sa.String(length=255), unique=True, nullable=True),
        sa.Column('amount_minor_units', sa.Integer(), nullable=False),
        sa.Column('currency', sa.String(length=3), nullable=False, server_default='USD'),
        sa.Column('reason', sa.String(length=255), nullable=True),
        sa.Column('status', sa.String(length=50), nullable=False, server_default='PENDING'),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='1')
    )

    op.create_table(
        'disbursements',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('milestone_id', sa.String(length=36), sa.ForeignKey('milestones.id', ondelete='CASCADE'), nullable=False),
        sa.Column('organization_id', sa.String(length=36), sa.ForeignKey('organizations.id', ondelete='CASCADE'), nullable=False),
        sa.Column('amount_minor_units', sa.Integer(), nullable=False),
        sa.Column('currency', sa.String(length=3), nullable=False, server_default='USD'),
        sa.Column('status', sa.String(length=50), nullable=False, server_default='SCHEDULED'),
        sa.Column('payout_batch_id', sa.String(length=255), nullable=True),
        sa.Column('payout_item_id', sa.String(length=255), nullable=True),
        sa.Column('disbursed_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='1')
    )

    # Evidence & Verification
    op.create_table(
        'evidence_files',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('milestone_id', sa.String(length=36), sa.ForeignKey('milestones.id', ondelete='CASCADE'), nullable=False),
        sa.Column('uploader_id', sa.String(length=36), sa.ForeignKey('users.id', ondelete='SET NULL'), nullable=True),
        sa.Column('file_path', sa.String(length=512), nullable=False),
        sa.Column('file_hash', sa.String(length=64), nullable=False),
        sa.Column('content_address', sa.String(length=128), nullable=False),
        sa.Column('mime_type', sa.String(length=64), nullable=False),
        sa.Column('file_size_bytes', sa.Integer(), nullable=False),
        sa.Column('status', sa.String(length=50), nullable=False, server_default='UPLOADED'),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='1')
    )

    op.create_table(
        'verification_reports',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('evidence_file_id', sa.String(length=36), sa.ForeignKey('evidence_files.id', ondelete='CASCADE'), nullable=False),
        sa.Column('overall_confidence', sa.Float(), nullable=False),
        sa.Column('overall_pass', sa.Boolean(), nullable=False),
        sa.Column('reviewed_by_id', sa.String(length=36), sa.ForeignKey('users.id', ondelete='SET NULL'), nullable=True),
        sa.Column('summary_notes', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='1')
    )

    op.create_table(
        'verification_check_records',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('report_id', sa.String(length=36), sa.ForeignKey('verification_reports.id', ondelete='CASCADE'), nullable=False),
        sa.Column('check_type', sa.String(length=50), nullable=False),
        sa.Column('status', sa.String(length=50), nullable=False),
        sa.Column('passed', sa.Boolean(), nullable=False),
        sa.Column('message', sa.Text(), nullable=False),
        sa.Column('details_json', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='1')
    )

    # Reviews, Impact & Audit
    op.create_table(
        'review_tasks',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('milestone_id', sa.String(length=36), sa.ForeignKey('milestones.id', ondelete='CASCADE'), nullable=False),
        sa.Column('evidence_file_id', sa.String(length=36), sa.ForeignKey('evidence_files.id', ondelete='CASCADE'), nullable=False),
        sa.Column('assigned_reviewer_id', sa.String(length=36), sa.ForeignKey('users.id', ondelete='SET NULL'), nullable=True),
        sa.Column('status', sa.String(length=50), nullable=False, server_default='PENDING'),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='1')
    )

    op.create_table(
        'review_decision_records',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('task_id', sa.String(length=36), sa.ForeignKey('review_tasks.id', ondelete='CASCADE'), nullable=False),
        sa.Column('reviewer_id', sa.String(length=36), sa.ForeignKey('users.id', ondelete='RESTRICT'), nullable=False),
        sa.Column('decision', sa.String(length=50), nullable=False),
        sa.Column('comments', sa.Text(), nullable=False),
        sa.Column('decided_at', sa.DateTime(), nullable=False),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='1')
    )

    op.create_table(
        'impact_certificates',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('donation_id', sa.String(length=36), sa.ForeignKey('donations.id', ondelete='CASCADE'), unique=True, nullable=False),
        sa.Column('milestone_id', sa.String(length=36), sa.ForeignKey('milestones.id', ondelete='CASCADE'), nullable=False),
        sa.Column('certificate_number', sa.String(length=64), unique=True, nullable=False),
        sa.Column('recipient_name', sa.String(length=255), nullable=False),
        sa.Column('certificate_hash', sa.String(length=64), nullable=False),
        sa.Column('token_uri', sa.String(length=512), nullable=False),
        sa.Column('metadata_json', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='1')
    )

    op.create_table(
        'audit_events',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('actor_id', sa.String(length=36), nullable=True),
        sa.Column('action', sa.String(length=100), nullable=False),
        sa.Column('entity_type', sa.String(length=100), nullable=False),
        sa.Column('entity_id', sa.String(length=64), nullable=False),
        sa.Column('ip_address', sa.String(length=45), nullable=True),
        sa.Column('correlation_id', sa.String(length=64), nullable=True),
        sa.Column('details_json', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='1')
    )


def downgrade() -> None:
    op.drop_table('audit_events')
    op.drop_table('impact_certificates')
    op.drop_table('review_decision_records')
    op.drop_table('review_tasks')
    op.drop_table('verification_check_records')
    op.drop_table('verification_reports')
    op.drop_table('evidence_files')
    op.drop_table('disbursements')
    op.drop_table('payment_refunds')
    op.drop_table('payment_captures')
    op.drop_table('payment_authorizations')
    op.drop_table('payment_orders')
    op.drop_table('donations')
    op.drop_table('project_locations')
    op.drop_table('milestones')
    op.drop_table('projects')
    op.drop_table('project_categories')
    op.drop_table('ngo_profiles')
    op.drop_table('organization_memberships')
    op.drop_table('organizations')
    op.drop_table('user_profiles')
    op.drop_table('users')

