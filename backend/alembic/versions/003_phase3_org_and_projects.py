"""003_phase3_org_and_projects

Revision ID: 003_phase3_org_and_projects
Revises: 002_phase2_identity_and_auth
Create Date: 2026-11-11 16:30:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '003_phase3_org_and_projects'
down_revision: Union[str, None] = '002_phase2_identity_and_auth'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Update organizations table with Phase 3 profile and verification fields
    with op.batch_alter_table('organizations') as batch_op:
        batch_op.add_column(sa.Column('legal_name', sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column('display_name', sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column('description', sa.Text(), nullable=True))
        batch_op.add_column(sa.Column('organization_type', sa.String(length=50), nullable=False, server_default='NON_PROFIT'))
        batch_op.add_column(sa.Column('registration_authority', sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column('registration_date', sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column('classification', sa.String(length=100), nullable=True))
        batch_op.add_column(sa.Column('registered_address', sa.Text(), nullable=True))
        batch_op.add_column(sa.Column('operating_address', sa.Text(), nullable=True))
        batch_op.add_column(sa.Column('official_email', sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column('official_phone', sa.String(length=32), nullable=True))
        batch_op.add_column(sa.Column('mission_statement', sa.Text(), nullable=True))
        batch_op.add_column(sa.Column('vision', sa.Text(), nullable=True))
        batch_op.add_column(sa.Column('focus_areas', sa.Text(), nullable=True))
        batch_op.add_column(sa.Column('operating_regions', sa.Text(), nullable=True))
        batch_op.add_column(sa.Column('beneficiary_categories', sa.Text(), nullable=True))
        batch_op.add_column(sa.Column('years_of_operation', sa.Integer(), nullable=False, server_default='0'))
        batch_op.add_column(sa.Column('submitted_at', sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column('verified_at', sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column('reviewer_id', sa.String(length=36), nullable=True))
        batch_op.add_column(sa.Column('rejection_reason', sa.Text(), nullable=True))
        batch_op.add_column(sa.Column('suspension_reason', sa.Text(), nullable=True))
        batch_op.add_column(sa.Column('revision_notes', sa.Text(), nullable=True))

    # 2. Create organization_documents table
    op.create_table(
        'organization_documents',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('organization_id', sa.String(length=36), sa.ForeignKey('organizations.id', ondelete='CASCADE'), nullable=False),
        sa.Column('document_type', sa.String(length=100), nullable=False),
        sa.Column('original_filename', sa.String(length=255), nullable=False),
        sa.Column('storage_filename', sa.String(length=255), nullable=False),
        sa.Column('storage_path', sa.String(length=512), nullable=False),
        sa.Column('mime_type', sa.String(length=100), nullable=False),
        sa.Column('file_size_bytes', sa.Integer(), nullable=False),
        sa.Column('file_hash_sha256', sa.String(length=64), nullable=False),
        sa.Column('uploaded_by_id', sa.String(length=36), sa.ForeignKey('users.id'), nullable=False),
        sa.Column('version_number', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('expires_at', sa.DateTime(), nullable=True),
        sa.Column('review_status', sa.String(length=50), nullable=False, server_default='PENDING'),
        sa.Column('reviewer_id', sa.String(length=36), sa.ForeignKey('users.id'), nullable=True),
        sa.Column('reviewer_comments', sa.Text(), nullable=True),
        sa.Column('access_classification', sa.String(length=50), nullable=False, server_default='CONFIDENTIAL'),
        sa.Column('created_at', sa.DateTime(), nullable=False),
        sa.Column('updated_at', sa.DateTime(), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='1')
    )
    op.create_index('ix_org_docs_org_id', 'organization_documents', ['organization_id'])
    op.create_index('ix_org_docs_file_hash', 'organization_documents', ['file_hash_sha256'])

    # 3. Update project_categories table
    with op.batch_alter_table('project_categories') as batch_op:
        batch_op.add_column(sa.Column('slug', sa.String(length=100), nullable=True))
        batch_op.add_column(sa.Column('is_active', sa.Boolean(), nullable=False, server_default='1'))

    # 4. Update projects table with Phase 3 attributes
    with op.batch_alter_table('projects') as batch_op:
        batch_op.add_column(sa.Column('summary', sa.String(length=500), nullable=True))
        batch_op.add_column(sa.Column('objectives', sa.Text(), nullable=True))
        batch_op.add_column(sa.Column('project_type', sa.String(length=50), nullable=False, server_default='CHARITABLE_PROJECT'))
        batch_op.add_column(sa.Column('public_visibility', sa.String(length=20), nullable=False, server_default='PUBLIC'))
        batch_op.add_column(sa.Column('raised_amount_minor_units', sa.Integer(), nullable=False, server_default='0'))
        batch_op.add_column(sa.Column('target_beneficiaries', sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column('expected_outcomes', sa.Text(), nullable=True))
        batch_op.add_column(sa.Column('deliverables', sa.Text(), nullable=True))
        batch_op.add_column(sa.Column('impact_metrics', sa.Text(), nullable=True))
        batch_op.add_column(sa.Column('start_date', sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column('end_date', sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column('country', sa.String(length=3), nullable=True, server_default='USA'))
        batch_op.add_column(sa.Column('state_region', sa.String(length=100), nullable=True))
        batch_op.add_column(sa.Column('city', sa.String(length=100), nullable=True))
        batch_op.add_column(sa.Column('address', sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column('latitude', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('longitude', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('radius_meters', sa.Float(), nullable=False, server_default='1000.0'))
        batch_op.add_column(sa.Column('creator_id', sa.String(length=36), nullable=True))
        batch_op.add_column(sa.Column('reviewer_id', sa.String(length=36), nullable=True))
        batch_op.add_column(sa.Column('submitted_at', sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column('reviewed_at', sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column('activated_at', sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column('rejection_reason', sa.Text(), nullable=True))
        batch_op.add_column(sa.Column('revision_notes', sa.Text(), nullable=True))
        batch_op.add_column(sa.Column('version_number', sa.Integer(), nullable=False, server_default='1'))
        batch_op.add_column(sa.Column('is_archived', sa.Boolean(), nullable=False, server_default='0'))

    # 5. Update milestones table with Phase 3 attributes
    with op.batch_alter_table('milestones') as batch_op:
        batch_op.add_column(sa.Column('objective', sa.Text(), nullable=True))
        batch_op.add_column(sa.Column('deliverables', sa.Text(), nullable=True))
        batch_op.add_column(sa.Column('expected_outcome', sa.Text(), nullable=True))
        batch_op.add_column(sa.Column('success_criteria', sa.Text(), nullable=True))
        batch_op.add_column(sa.Column('required_evidence_desc', sa.Text(), nullable=True))
        batch_op.add_column(sa.Column('planned_start_date', sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column('planned_completion_date', sa.DateTime(), nullable=True))
        batch_op.add_column(sa.Column('latitude', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('longitude', sa.Float(), nullable=True))
        batch_op.add_column(sa.Column('radius_meters', sa.Float(), nullable=False, server_default='1000.0'))
        batch_op.add_column(sa.Column('address', sa.String(length=255), nullable=True))
        batch_op.add_column(sa.Column('city', sa.String(length=100), nullable=True))
        batch_op.add_column(sa.Column('country', sa.String(length=3), nullable=True))
        batch_op.add_column(sa.Column('version_number', sa.Integer(), nullable=False, server_default='1'))
        batch_op.create_unique_constraint('uq_milestone_project_seq', ['project_id', 'sequence_number'])


def downgrade() -> None:
    with op.batch_alter_table('milestones') as batch_op:
        batch_op.drop_constraint('uq_milestone_project_seq', type_='unique')
        batch_op.drop_column('version_number')
        batch_op.drop_column('country')
        batch_op.drop_column('city')
        batch_op.drop_column('address')
        batch_op.drop_column('radius_meters')
        batch_op.drop_column('longitude')
        batch_op.drop_column('latitude')
        batch_op.drop_column('planned_completion_date')
        batch_op.drop_column('planned_start_date')
        batch_op.drop_column('required_evidence_desc')
        batch_op.drop_column('success_criteria')
        batch_op.drop_column('expected_outcome')
        batch_op.drop_column('deliverables')
        batch_op.drop_column('objective')

    with op.batch_alter_table('projects') as batch_op:
        batch_op.drop_column('is_archived')
        batch_op.drop_column('version_number')
        batch_op.drop_column('revision_notes')
        batch_op.drop_column('rejection_reason')
        batch_op.drop_column('activated_at')
        batch_op.drop_column('reviewed_at')
        batch_op.drop_column('submitted_at')
        batch_op.drop_column('reviewer_id')
        batch_op.drop_column('creator_id')
        batch_op.drop_column('radius_meters')
        batch_op.drop_column('longitude')
        batch_op.drop_column('latitude')
        batch_op.drop_column('address')
        batch_op.drop_column('city')
        batch_op.drop_column('state_region')
        batch_op.drop_column('country')
        batch_op.drop_column('end_date')
        batch_op.drop_column('start_date')
        batch_op.drop_column('impact_metrics')
        batch_op.drop_column('deliverables')
        batch_op.drop_column('expected_outcomes')
        batch_op.drop_column('target_beneficiaries')
        batch_op.drop_column('raised_amount_minor_units')
        batch_op.drop_column('public_visibility')
        batch_op.drop_column('project_type')
        batch_op.drop_column('objectives')
        batch_op.drop_column('summary')

    with op.batch_alter_table('project_categories') as batch_op:
        batch_op.drop_column('is_active')
        batch_op.drop_column('slug')

    op.drop_index('ix_org_docs_file_hash', table_name='organization_documents')
    op.drop_index('ix_org_docs_org_id', table_name='organization_documents')
    op.drop_table('organization_documents')

    with op.batch_alter_table('organizations') as batch_op:
        batch_op.drop_column('revision_notes')
        batch_op.drop_column('suspension_reason')
        batch_op.drop_column('rejection_reason')
        batch_op.drop_column('reviewer_id')
        batch_op.drop_column('verified_at')
        batch_op.drop_column('submitted_at')
        batch_op.drop_column('years_of_operation')
        batch_op.drop_column('beneficiary_categories')
        batch_op.drop_column('operating_regions')
        batch_op.drop_column('focus_areas')
        batch_op.drop_column('vision')
        batch_op.drop_column('mission_statement')
        batch_op.drop_column('official_phone')
        batch_op.drop_column('official_email')
        batch_op.drop_column('operating_address')
        batch_op.drop_column('registered_address')
        batch_op.drop_column('classification')
        batch_op.drop_column('registration_date')
        batch_op.drop_column('registration_authority')
        batch_op.drop_column('organization_type')
        batch_op.drop_column('description')
        batch_op.drop_column('display_name')
        batch_op.drop_column('legal_name')

