"""010_phase10_notifications_and_webhooks

Revision ID: 010_phase10_notifications_and_webhooks
Revises: 009_phase9_impact_ledger_and_certificates
Create Date: 2026-11-12 21:15:00.000000

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '010_phase10_notifications_and_webhooks'
down_revision: Union[str, None] = '009_phase9_impact_ledger_and_certificates'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Create notification_outbox table
    op.create_table(
        'notification_outbox',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('event_id', sa.String(length=64), unique=True, nullable=False),
        sa.Column('event_type', sa.String(length=64), nullable=False),
        sa.Column('entity_type', sa.String(length=64), nullable=False),
        sa.Column('entity_id', sa.String(length=64), nullable=False),
        sa.Column('organization_id', sa.String(length=36), nullable=True),
        sa.Column('actor_user_id', sa.String(length=36), nullable=True),
        sa.Column('correlation_id', sa.String(length=64), nullable=True),
        sa.Column('idempotency_key', sa.String(length=128), unique=True, nullable=False),
        sa.Column('payload_json', sa.Text(), nullable=False),
        sa.Column('status', sa.String(length=32), nullable=False, server_default='PENDING'),
        sa.Column('retry_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('max_retries', sa.Integer(), nullable=False, server_default='5'),
        sa.Column('next_retry_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('processed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='1'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now())
    )
    op.create_index('ix_notification_outbox_event_id', 'notification_outbox', ['event_id'])
    op.create_index('ix_notification_outbox_event_type', 'notification_outbox', ['event_type'])
    op.create_index('ix_notification_outbox_org_id', 'notification_outbox', ['organization_id'])
    op.create_index('ix_notification_outbox_idempotency', 'notification_outbox', ['idempotency_key'])
    op.create_index('ix_notification_outbox_status', 'notification_outbox', ['status'])
    op.create_index('ix_notification_outbox_status_next_retry', 'notification_outbox', ['status', 'next_retry_at'])

    # 2. Create notifications table
    op.create_table(
        'notifications',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('user_id', sa.String(length=36), sa.ForeignKey('users.id', ondelete='CASCADE'), nullable=False),
        sa.Column('organization_id', sa.String(length=36), sa.ForeignKey('organizations.id', ondelete='SET NULL'), nullable=True),
        sa.Column('category', sa.String(length=32), nullable=False),
        sa.Column('event_type', sa.String(length=64), nullable=False),
        sa.Column('title', sa.String(length=255), nullable=False),
        sa.Column('message', sa.Text(), nullable=False),
        sa.Column('action_url', sa.String(length=512), nullable=True),
        sa.Column('entity_type', sa.String(length=64), nullable=True),
        sa.Column('entity_id', sa.String(length=64), nullable=True),
        sa.Column('is_read', sa.Boolean(), nullable=False, server_default='0'),
        sa.Column('read_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('is_archived', sa.Boolean(), nullable=False, server_default='0'),
        sa.Column('archived_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='1'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now())
    )
    op.create_index('ix_notifications_user_id', 'notifications', ['user_id'])
    op.create_index('ix_notifications_org_id', 'notifications', ['organization_id'])
    op.create_index('ix_notifications_category', 'notifications', ['category'])
    op.create_index('ix_notifications_event_type', 'notifications', ['event_type'])
    op.create_index('ix_notifications_is_read', 'notifications', ['is_read'])
    op.create_index('ix_notifications_is_archived', 'notifications', ['is_archived'])
    op.create_index('ix_notifications_user_unread', 'notifications', ['user_id', 'is_read'])

    # 3. Create notification_preferences table
    op.create_table(
        'notification_preferences',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('user_id', sa.String(length=36), sa.ForeignKey('users.id', ondelete='CASCADE'), unique=True, nullable=False),
        sa.Column('email_enabled', sa.Boolean(), nullable=False, server_default='1'),
        sa.Column('in_app_enabled', sa.Boolean(), nullable=False, server_default='1'),
        sa.Column('donation_updates', sa.Boolean(), nullable=False, server_default='1'),
        sa.Column('project_updates', sa.Boolean(), nullable=False, server_default='1'),
        sa.Column('milestone_updates', sa.Boolean(), nullable=False, server_default='1'),
        sa.Column('verification_updates', sa.Boolean(), nullable=False, server_default='1'),
        sa.Column('certificate_notifications', sa.Boolean(), nullable=False, server_default='1'),
        sa.Column('security_alerts', sa.Boolean(), nullable=False, server_default='1'),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='1'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now())
    )
    op.create_index('ix_notification_preferences_user_id', 'notification_preferences', ['user_id'])

    # 4. Create notification_delivery_attempts table
    op.create_table(
        'notification_delivery_attempts',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('notification_id', sa.String(length=36), sa.ForeignKey('notifications.id', ondelete='CASCADE'), nullable=True),
        sa.Column('channel', sa.String(length=32), nullable=False),
        sa.Column('recipient', sa.String(length=255), nullable=False),
        sa.Column('status', sa.String(length=32), nullable=False, server_default='PENDING'),
        sa.Column('attempt_number', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('response_code', sa.String(length=32), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('attempted_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='1'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now())
    )
    op.create_index('ix_notification_delivery_attempts_notif_id', 'notification_delivery_attempts', ['notification_id'])
    op.create_index('ix_notification_delivery_attempts_channel', 'notification_delivery_attempts', ['channel'])

    # 5. Create webhook_subscriptions table
    op.create_table(
        'webhook_subscriptions',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('organization_id', sa.String(length=36), sa.ForeignKey('organizations.id', ondelete='CASCADE'), nullable=True),
        sa.Column('created_by_user_id', sa.String(length=36), sa.ForeignKey('users.id', ondelete='RESTRICT'), nullable=False),
        sa.Column('endpoint_url', sa.String(length=512), nullable=False),
        sa.Column('description', sa.String(length=255), nullable=True),
        sa.Column('subscribed_events', sa.JSON(), nullable=False),
        sa.Column('consecutive_failures', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('suspended_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='1'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now())
    )
    op.create_index('ix_webhook_subscriptions_org_id', 'webhook_subscriptions', ['organization_id'])
    op.create_index('ix_webhook_subscriptions_user_id', 'webhook_subscriptions', ['created_by_user_id'])
    op.create_index('ix_webhook_subscriptions_is_active', 'webhook_subscriptions', ['is_active'])

    # 6. Create webhook_endpoint_secrets table
    op.create_table(
        'webhook_endpoint_secrets',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('subscription_id', sa.String(length=36), sa.ForeignKey('webhook_subscriptions.id', ondelete='CASCADE'), nullable=False),
        sa.Column('secret_encrypted', sa.Text(), nullable=False),
        sa.Column('secret_prefix', sa.String(length=16), nullable=False),
        sa.Column('rotated_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='1'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now())
    )
    op.create_index('ix_webhook_endpoint_secrets_sub_id', 'webhook_endpoint_secrets', ['subscription_id'])
    op.create_index('ix_webhook_endpoint_secrets_is_active', 'webhook_endpoint_secrets', ['is_active'])

    # 7. Create webhook_deliveries table
    op.create_table(
        'webhook_deliveries',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('subscription_id', sa.String(length=36), sa.ForeignKey('webhook_subscriptions.id', ondelete='CASCADE'), nullable=False),
        sa.Column('event_id', sa.String(length=64), nullable=False),
        sa.Column('event_type', sa.String(length=64), nullable=False),
        sa.Column('payload_json', sa.Text(), nullable=False),
        sa.Column('status', sa.String(length=32), nullable=False, server_default='PENDING'),
        sa.Column('retry_count', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('max_retries', sa.Integer(), nullable=False, server_default='5'),
        sa.Column('next_retry_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='1'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now())
    )
    op.create_index('ix_webhook_deliveries_sub_id', 'webhook_deliveries', ['subscription_id'])
    op.create_index('ix_webhook_deliveries_event_id', 'webhook_deliveries', ['event_id'])
    op.create_index('ix_webhook_deliveries_event_type', 'webhook_deliveries', ['event_type'])
    op.create_index('ix_webhook_deliveries_status', 'webhook_deliveries', ['status'])
    op.create_index('ix_webhook_deliveries_status_next_retry', 'webhook_deliveries', ['status', 'next_retry_at'])

    # 8. Create webhook_delivery_attempts table
    op.create_table(
        'webhook_delivery_attempts',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('delivery_id', sa.String(length=36), sa.ForeignKey('webhook_deliveries.id', ondelete='CASCADE'), nullable=False),
        sa.Column('attempt_number', sa.Integer(), nullable=False, server_default='1'),
        sa.Column('status_code', sa.Integer(), nullable=True),
        sa.Column('response_headers_json', sa.Text(), nullable=True),
        sa.Column('response_body', sa.Text(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('execution_duration_ms', sa.Integer(), nullable=False, server_default='0'),
        sa.Column('attempted_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='1'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now())
    )
    op.create_index('ix_webhook_delivery_attempts_deliv_id', 'webhook_delivery_attempts', ['delivery_id'])

    # 9. Create notification_template_versions table
    op.create_table(
        'notification_template_versions',
        sa.Column('id', sa.String(length=36), primary_key=True),
        sa.Column('template_key', sa.String(length=64), nullable=False),
        sa.Column('version', sa.String(length=32), nullable=False, server_default='v1.0.0'),
        sa.Column('subject_template', sa.String(length=255), nullable=False),
        sa.Column('html_template', sa.Text(), nullable=False),
        sa.Column('text_template', sa.Text(), nullable=False),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='1'),
        sa.Column('created_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint('template_key', 'version', name='uq_notification_template_key_version')
    )
    op.create_index('ix_notification_template_versions_key', 'notification_template_versions', ['template_key'])


def downgrade() -> None:
    op.drop_table('notification_template_versions')
    op.drop_table('webhook_delivery_attempts')
    op.drop_table('webhook_deliveries')
    op.drop_table('webhook_endpoint_secrets')
    op.drop_table('webhook_subscriptions')
    op.drop_table('notification_delivery_attempts')
    op.drop_table('notification_preferences')
    op.drop_table('notifications')
    op.drop_table('notification_outbox')

