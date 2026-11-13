// Standard API response wrappers matching FastAPI backend

export interface StandardResponse<T> {
  success: boolean;
  data: T;
  message?: string;
  correlation_id?: string;
  timestamp: string;
}

export interface PaginationMeta {
  page: number;
  page_size: number;
  total_items: number;
  total_pages: number;
}

export interface PaginatedResponse<T> {
  success: boolean;
  data: T[];
  pagination: PaginationMeta;
  correlation_id?: string;
  timestamp: string;
}

export interface ApiError {
  success: false;
  error: {
    code: string;
    message: string;
    details?: Record<string, unknown>;
  };
  correlation_id?: string;
  timestamp: string;
}

// Auth types
export interface TokenResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
}

export interface UserProfile {
  id: string;
  email: string;
  full_name: string;
  role: 'DONOR' | 'REVIEWER' | 'PLATFORM_ADMIN';
  status: string;
  created_at: string;
}

// Project types
export interface Project {
  id: string;
  title: string;
  description: string;
  status: string;
  public_visibility: string;
  funding_target_minor_units: number;
  currency: string;
  raised_amount_minor_units: number;
  category?: string;
  country?: string;
  organization_id: string;
  organization_name?: string;
  created_at: string;
  updated_at: string;
  milestones?: Milestone[];
}

// Organization types
export interface Organization {
  id: string;
  name: string;
  legal_name?: string;
  status: string;
  mission_statement?: string;
  country?: string;
  created_at: string;
  total_projects?: number;
}

// Donation types
export interface Donation {
  id: string;
  project_id: string;
  project_title?: string;
  organization_name?: string;
  donor_id: string;
  amount_minor_units: number;
  currency: string;
  status: string;
  created_at: string;
  paypal_order_id?: string;
}

// Milestone types
export interface Milestone {
  id: string;
  project_id: string;
  title: string;
  description: string;
  status: string;
  target_amount_minor_units: number;
  sequence_number: number;
  target_lat?: number;
  target_lng?: number;
  is_location_sensitive?: boolean;
}

// Certificate types
export interface ImpactCertificate {
  id: string;
  certificate_id: string;
  project_id: string;
  project_title?: string;
  milestone_id: string;
  milestone_title?: string;
  organization_id: string;
  organization_name?: string;
  donor_id?: string;
  status: string;
  issued_at: string;
  signature?: string;
  public_verification_url?: string;
}

// Notification types
export interface NotificationItem {
  id: string;
  title: string;
  message: string;
  category: string;
  read: boolean;
  created_at: string;
  action_url?: string;
}

