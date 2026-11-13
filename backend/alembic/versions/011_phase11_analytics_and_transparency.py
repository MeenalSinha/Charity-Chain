"""011_phase11_analytics_and_transparency

Revision ID: 011_phase11_analytics_and_transparency
Revises: 010_phase10_notifications_and_webhooks
Create Date: 2026-11-13 13:30:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '011_phase11_analytics_and_transparency'
down_revision: Union[str, None] = '010_phase10_notifications_and_webhooks'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Add is_location_sensitive to projects
    with op.batch_alter_table('projects') as batch_op:
        batch_op.add_column(
            sa.Column('is_location_sensitive', sa.Boolean(), nullable=False, server_default=sa.false())
        )

    # 2. Add is_location_sensitive to milestones
    with op.batch_alter_table('milestones') as batch_op:
        batch_op.add_column(
            sa.Column('is_location_sensitive', sa.Boolean(), nullable=False, server_default=sa.false())
        )

    # 3. Create analytics_snapshots table
    op.create_table(
        'analytics_snapshots',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('scope', sa.String(length=50), nullable=False, index=True),
        sa.Column('scope_id', sa.String(length=36), nullable=True, index=True),
        sa.Column('snapshot_type', sa.String(length=50), nullable=False, index=True),
        sa.Column('period_start', sa.DateTime(), nullable=True),
        sa.Column('period_end', sa.DateTime(), nullable=True),
        sa.Column('as_of_date', sa.DateTime(), nullable=False),
        sa.Column('currency', sa.String(length=3), nullable=False, server_default='USD'),
        sa.Column('metrics_json', sa.Text(), nullable=False),
        sa.Column('is_valid', sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index(
        'ix_analytics_snapshots_scope_type',
        'analytics_snapshots',
        ['scope', 'snapshot_type', 'is_valid']
    )

    # 4. Create analytics_export_jobs table
    op.create_table(
        'analytics_export_jobs',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('user_id', sa.String(length=36), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False, index=True),
        sa.Column('organization_id', sa.String(length=36), sa.ForeignKey('organizations.id', ondelete='SET NULL'), nullable=True, index=True),
        sa.Column('export_type', sa.String(length=50), nullable=False, index=True),
        sa.Column('export_format', sa.String(length=10), nullable=False, server_default='CSV'),
        sa.Column('status', sa.String(length=50), nullable=False, server_default='PENDING', index=True),
        sa.Column('file_path', sa.String(length=512), nullable=True),
        sa.Column('download_url', sa.String(length=512), nullable=True),
        sa.Column('file_size_bytes', sa.Integer(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('parameters_json', sa.Text(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )

    # 5. Create public_transparency_publications table
    op.create_table(
        'public_transparency_publications',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('project_id', sa.String(length=36), sa.ForeignKey('projects.id', ondelete='CASCADE'), nullable=True, index=True),
        sa.Column('organization_id', sa.String(length=36), sa.ForeignKey('organizations.id', ondelete='CASCADE'), nullable=True, index=True),
        sa.Column('published_by_id', sa.String(length=36), sa.ForeignKey('users.id', ondelete='SET NULL'), nullable=True),
        sa.Column('publication_type', sa.String(length=50), nullable=False, server_default='PROJECT_SUMMARY', index=True),
        sa.Column('is_published', sa.Boolean(), nullable=False, server_default=sa.true(), index=True),
        sa.Column('published_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('methodology_version', sa.String(length=50), nullable=False, server_default='v1.0'),
        sa.Column('summary_data_json', sa.Text(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table('public_transparency_publications')
    op.drop_table('analytics_export_jobs')
    op.drop_index('ix_analytics_snapshots_scope_type', table_name='analytics_snapshots')
    op.drop_table('analytics_snapshots')
    with op.batch_alter_table('milestones') as batch_op:
        batch_op.drop_column('is_location_sensitive')
    with op.batch_alter_table('projects') as batch_op:
        batch_op.drop_column('is_location_sensitive')

