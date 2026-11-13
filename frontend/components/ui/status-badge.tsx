import React from 'react';
import { Badge, type BadgeProps } from './badge';

const statusConfig: Record<string, { label: string; variant: BadgeProps['variant'] }> = {
  ACTIVE: { label: 'Active', variant: 'success' },
  COMPLETED: { label: 'Completed', variant: 'success' },
  APPROVED: { label: 'Approved', variant: 'success' },
  VERIFIED: { label: 'Verified', variant: 'success' },
  CAPTURED: { label: 'Captured', variant: 'success' },
  DRAFT: { label: 'Draft', variant: 'secondary' },
  PENDING: { label: 'Pending', variant: 'warning' },
  SUBMITTED: { label: 'Submitted', variant: 'warning' },
  UNDER_REVIEW: { label: 'Under Review', variant: 'warning' },
  EVIDENCE_SUBMITTED: { label: 'Evidence Submitted', variant: 'warning' },
  AUTHORIZED: { label: 'Authorized', variant: 'default' },
  PAUSED: { label: 'Paused', variant: 'outline' },
  REJECTED: { label: 'Rejected', variant: 'danger' },
  FAILED: { label: 'Failed', variant: 'danger' },
  CANCELLED: { label: 'Cancelled', variant: 'danger' },
  SUSPENDED: { label: 'Suspended', variant: 'danger' },
  ARCHIVED: { label: 'Archived', variant: 'secondary' },
};

export function StatusBadge({ status }: { status: string }) {
  const config = statusConfig[status] ?? { label: status, variant: 'secondary' as const };
  return <Badge variant={config.variant}>{config.label}</Badge>;
}

