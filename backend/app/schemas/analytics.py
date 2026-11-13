"""
analytics.py - Pydantic Request & Response Schemas for Phase 11 Analytics & Transparency
========================================================================================
"""

from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# NGO Analytics Schemas
# ---------------------------------------------------------------------------

class NGOOverviewResponse(BaseModel):
    organization_id: str
    organization_name: str
    verification_status: str
    projects: Dict[str, int]
    milestones: Dict[str, int]
    financial: Dict[str, Any]
    as_of_date: str


class NGOFundingResponse(BaseModel):
    organization_id: str
    total_raised_cents: int
    total_donations: int
    average_donation_cents: int
    average_donation_formatted: str
    donor_retention: Dict[str, Any]
    funding_by_project: List[Dict[str, Any]]
    size_distribution: Dict[str, int]


class ProjectPerformanceItem(BaseModel):
    project_id: str
    title: str
    status: str
    target_amount_cents: int
    raised_amount_cents: int
    funding_progress_percent: float
    milestones_total: int
    milestones_completed: int
    milestones_verified: int
    completion_percent: float
    created_at: Optional[str] = None


class EvidenceVerificationAnalyticsResponse(BaseModel):
    evidence_files_submitted: int
    review_cases_total: int
    approved: int
    rejected: int
    revisions_requested: int
    under_review: int
    approval_rate: float


# ---------------------------------------------------------------------------
# Donor Analytics Schemas
# ---------------------------------------------------------------------------

class DonorOverviewResponse(BaseModel):
    donor_id: str
    total_captured_cents: int
    refunded_cents: int
    net_contributions_cents: int
    total_captured_formatted: str
    net_contributions_formatted: str
    supported_projects_count: int
    supported_ngos_count: int
    completed_milestones_count: int
    certificates_count: int
    currencies: Dict[str, Any]
    as_of_date: str


class DonorPortfolioResponse(BaseModel):
    donor_id: str
    total_donations: int
    total_given_cents: int
    average_donation_cents: int
    by_category: List[Dict[str, Any]]
    by_organization: List[Dict[str, Any]]
    recent_donations: List[Dict[str, Any]]


class DonorImpactJourneyItem(BaseModel):
    donation_id: str
    amount_cents: int
    amount_formatted: str
    currency: str
    donated_at: Optional[str] = None
    project: Dict[str, Any]
    milestone: Dict[str, Any]
    evidence_submitted: bool
    evidence_count: int
    is_completed: bool
    certificate: Dict[str, Any]


class AnnualGivingReportResponse(BaseModel):
    report_id: str
    donor_id: str
    tax_year: int
    generated_at: str
    total_contributions_cents: int
    total_contributions_formatted: str
    currency: str
    donation_count: int
    contributions: List[Dict[str, Any]]
    tax_disclaimer: str


# ---------------------------------------------------------------------------
# Platform & Administrative Analytics Schemas
# ---------------------------------------------------------------------------

class PlatformOverviewResponse(BaseModel):
    users: Dict[str, int]
    organizations: Dict[str, int]
    projects: Dict[str, int]
    milestones: Dict[str, int]
    financial: Dict[str, Any]
    impact: Dict[str, int]
    as_of_date: str


class FinancialOperationsResponse(BaseModel):
    payment_operations: Dict[str, Any]
    payout_operations: Dict[str, Any]
    reconciliation: Dict[str, Any]
    financial_ledger_integrity: Dict[str, Any]


class ReviewVerificationOperationsResponse(BaseModel):
    human_review_pipeline: Dict[str, Any]
    ai_verification_engine: Dict[str, Any]


class OperationalHealthResponse(BaseModel):
    notification_outbox: Dict[str, int]
    webhook_deliveries: Dict[str, Any]
    impact_ledger_integrity: Dict[str, Any]


# ---------------------------------------------------------------------------
# Public Transparency Schemas
# ---------------------------------------------------------------------------

class PublicTransparencyOverviewResponse(BaseModel):
    organizations: Dict[str, int]
    projects: Dict[str, int]
    milestones: Dict[str, int]
    financial_transparency: Dict[str, Any]
    impact_certificates_issued: int
    as_of_date: str


class PublicProjectListItem(BaseModel):
    project_id: str
    title: str
    summary: Optional[str] = None
    organization_name: str
    category: str
    status: str
    target_amount_cents: int
    raised_amount_cents: int
    target_formatted: str
    raised_formatted: str
    funding_percent: float
    milestones_total: int
    milestones_completed: int
    milestones_verified: int
    country: Optional[str] = None
    city: Optional[str] = None


class PublicProjectDetailsResponse(BaseModel):
    project_id: str
    title: str
    summary: Optional[str] = None
    description: str
    objectives: Optional[str] = None
    organization: Dict[str, Any]
    category: str
    status: str
    target_amount_cents: int
    raised_amount_cents: int
    target_formatted: str
    raised_formatted: str
    currency: str
    milestones: List[Dict[str, Any]]
    certificates: List[Dict[str, Any]]
    country: Optional[str] = None
    city: Optional[str] = None
    updated_at: str


class PublicOrganizationProfileResponse(BaseModel):
    organization_id: str
    name: str
    status: str
    country: Optional[str] = None
    total_public_projects: int
    active_projects: int
    completed_projects: int
    total_raised_cents: int
    total_raised_formatted: str
    confirmed_disbursements_cents: int
    confirmed_disbursements_formatted: str


class PublicMapMarkerItem(BaseModel):
    project_id: str
    title: str
    organization_name: str
    category: str
    status: str
    latitude: float
    longitude: float
    is_obfuscated: bool
    target_amount_cents: int
    raised_amount_cents: int
    target_formatted: str
    raised_formatted: str
    country: str
    city: str


class RegionalDensityResponse(BaseModel):
    total_countries: int
    density: List[Dict[str, Any]]


class MethodologyResponse(BaseModel):
    methodology_version: str
    principles: List[str]
    financial_calculation: Dict[str, str]
    verification_and_review: Dict[str, str]
    privacy_and_exclusions: Dict[str, str]
    refresh_cadence: str

