export * from './api';

export type Role = 'DONOR' | 'REVIEWER' | 'PLATFORM_ADMIN';

export type ProjectStatus =
  | 'DRAFT'
  | 'SUBMITTED'
  | 'APPROVED'
  | 'ACTIVE'
  | 'PAUSED'
  | 'COMPLETED'
  | 'ARCHIVED';

export type MilestoneStatus =
  | 'PENDING'
  | 'EVIDENCE_SUBMITTED'
  | 'UNDER_REVIEW'
  | 'APPROVED'
  | 'COMPLETED'
  | 'REJECTED';

export type DonationStatus =
  | 'PENDING'
  | 'AUTHORIZED'
  | 'CAPTURED'
  | 'FAILED'
  | 'CANCELLED'
  | 'REFUNDED';

export type OrganizationStatus =
  | 'DRAFT'
  | 'PENDING'
  | 'VERIFIED'
  | 'REJECTED'
  | 'SUSPENDED';

