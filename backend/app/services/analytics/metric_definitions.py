"""
metric_definitions.py - Central Metric Definitions and Taxonomy
===============================================================
Phase 11: Analytics, Impact Reporting & Public Transparency
Provides a single, authoritative metric catalog with formal formulas,
underlying data sources, currency treatment, and privacy classifications.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any


class MetricPrivacy(str, Enum):
    PUBLIC = "PUBLIC"                        # Safe for unauthenticated public transparency
    DONOR_RESTRICTED = "DONOR_RESTRICTED"    # Accessible only by the donating user
    NGO_RESTRICTED = "NGO_RESTRICTED"        # Accessible only by authorized NGO members
    ADMIN_ONLY = "ADMIN_ONLY"                # Accessible only by platform administrators


@dataclass(frozen=True)
class MetricDefinition:
    metric_id: str
    name: str
    description: str
    formula: str
    data_source: str
    included_statuses: List[str]
    excluded_statuses: List[str]
    currency_treatment: str
    privacy: MetricPrivacy
    unit: str  # cents, count, percentage, days, ratio
    metadata: Dict[str, Any] = field(default_factory=dict)


# Central Catalog of Platform Metrics
METRIC_CATALOG: Dict[str, MetricDefinition] = {
    "total_captured_donations": MetricDefinition(
        metric_id="total_captured_donations",
        name="Total Captured Donations",
        description="Gross sum of successfully settled and captured charitable contributions.",
        formula="SUM(PaymentCapture.amount_minor_units) WHERE status == 'COMPLETED'",
        data_source="payment_captures",
        included_statuses=["COMPLETED"],
        excluded_statuses=["PENDING", "FAILED", "VOIDED"],
        currency_treatment="Integer minor units (cents), segregated by currency code.",
        privacy=MetricPrivacy.PUBLIC,
        unit="cents",
    ),
    "net_donations_after_refunds": MetricDefinition(
        metric_id="net_donations_after_refunds",
        name="Net Donations After Refunds",
        description="Net charitable funding retained after deducting all approved customer refunds and reversals.",
        formula="total_captured_donations - SUM(PaymentRefund.amount_minor_units WHERE status == 'COMPLETED')",
        data_source="payment_captures, payment_refunds",
        included_statuses=["COMPLETED"],
        excluded_statuses=["PENDING", "FAILED"],
        currency_treatment="Integer minor units (cents), segregated by currency code.",
        privacy=MetricPrivacy.PUBLIC,
        unit="cents",
    ),
    "confirmed_disbursements": MetricDefinition(
        metric_id="confirmed_disbursements",
        name="Confirmed Disbursements",
        description="Actual fund disbursements successfully paid out to verified NGO recipient accounts.",
        formula="SUM(Disbursement.amount_minor_units) WHERE status == 'COMPLETED'",
        data_source="disbursements",
        included_statuses=["COMPLETED"],
        excluded_statuses=["REQUESTED", "APPROVED", "SUBMITTED", "FAILED", "RETURNED"],
        currency_treatment="Integer minor units (cents), segregated by currency code.",
        privacy=MetricPrivacy.PUBLIC,
        unit="cents",
    ),
    "active_projects": MetricDefinition(
        metric_id="active_projects",
        name="Active Projects",
        description="Count of publicly accessible projects actively accepting donations or executing milestones.",
        formula="COUNT(DISTINCT Project.id) WHERE status == 'ACTIVE'",
        data_source="projects",
        included_statuses=["ACTIVE"],
        excluded_statuses=["DRAFT", "PENDING_REVIEW", "REJECTED", "COMPLETED", "SUSPENDED", "CANCELLED"],
        currency_treatment="N/A",
        privacy=MetricPrivacy.PUBLIC,
        unit="count",
    ),
    "completed_projects": MetricDefinition(
        metric_id="completed_projects",
        name="Completed Projects",
        description="Count of projects where all project deliverables and milestones have concluded.",
        formula="COUNT(DISTINCT Project.id) WHERE status == 'COMPLETED'",
        data_source="projects",
        included_statuses=["COMPLETED"],
        excluded_statuses=["DRAFT", "ACTIVE", "PENDING_REVIEW", "SUSPENDED"],
        currency_treatment="N/A",
        privacy=MetricPrivacy.PUBLIC,
        unit="count",
    ),
    "total_milestones": MetricDefinition(
        metric_id="total_milestones",
        name="Total Milestones",
        description="Total distinct milestones created across all published projects.",
        formula="COUNT(DISTINCT Milestone.id)",
        data_source="milestones",
        included_statuses=["ALL"],
        excluded_statuses=[],
        currency_treatment="N/A",
        privacy=MetricPrivacy.PUBLIC,
        unit="count",
    ),
    "completed_milestones": MetricDefinition(
        metric_id="completed_milestones",
        name="Completed Milestones",
        description="Milestones whose funds have been released and deliverables verified.",
        formula="COUNT(DISTINCT Milestone.id) WHERE status IN ('COMPLETED', 'RELEASED')",
        data_source="milestones",
        included_statuses=["COMPLETED", "RELEASED"],
        excluded_statuses=["DRAFT", "PENDING", "ACTIVE", "IN_REVIEW", "REJECTED"],
        currency_treatment="N/A",
        privacy=MetricPrivacy.PUBLIC,
        unit="count",
    ),
    "verified_milestones": MetricDefinition(
        metric_id="verified_milestones",
        name="Verified Milestones",
        description="Milestones with approved dual-key human review decisions and completed evidence verification.",
        formula="COUNT(DISTINCT Milestone.id) WHERE status IN ('VERIFIED', 'RELEASED')",
        data_source="milestones, review_cases",
        included_statuses=["VERIFIED", "RELEASED"],
        excluded_statuses=["DRAFT", "PENDING", "SUBMITTED", "REJECTED"],
        currency_treatment="N/A",
        privacy=MetricPrivacy.PUBLIC,
        unit="count",
    ),
    "evidence_review_completion_rate": MetricDefinition(
        metric_id="evidence_review_completion_rate",
        name="Evidence Review Completion Rate",
        description="Percentage of submitted human review cases that have reached a final approval or rejection.",
        formula="(decided_cases / total_submitted_cases) * 100",
        data_source="review_cases",
        included_statuses=["APPROVED", "REJECTED"],
        excluded_statuses=["PENDING", "IN_REVIEW", "REVISION_REQUESTED"],
        currency_treatment="N/A",
        privacy=MetricPrivacy.ADMIN_ONLY,
        unit="percentage",
    ),
    "donor_retention_rate": MetricDefinition(
        metric_id="donor_retention_rate",
        name="Donor Retention Rate",
        description="Percentage of distinct donors who have completed two or more donations.",
        formula="(repeat_donors / total_distinct_donors) * 100",
        data_source="donations",
        included_statuses=["COMPLETED"],
        excluded_statuses=["CREATED", "PENDING", "CANCELLED", "FAILED"],
        currency_treatment="N/A",
        privacy=MetricPrivacy.NGO_RESTRICTED,
        unit="percentage",
    ),
    "ngo_project_completion_rate": MetricDefinition(
        metric_id="ngo_project_completion_rate",
        name="NGO Project Completion Rate",
        description="Percentage of an organization's projects that have reached final completion.",
        formula="(completed_projects / total_launched_projects) * 100",
        data_source="projects",
        included_statuses=["COMPLETED"],
        excluded_statuses=["DRAFT"],
        currency_treatment="N/A",
        privacy=MetricPrivacy.PUBLIC,
        unit="percentage",
    ),
    "average_project_completion_days": MetricDefinition(
        metric_id="average_project_completion_days",
        name="Average Project Completion Days",
        description="Average number of calendar days elapsed between project activation and final milestone release.",
        formula="AVG(DATEDIFF(final_release_date, activated_at))",
        data_source="projects, milestones",
        included_statuses=["COMPLETED"],
        excluded_statuses=["DRAFT", "ACTIVE", "CANCELLED"],
        currency_treatment="N/A",
        privacy=MetricPrivacy.PUBLIC,
        unit="days",
    ),
    "total_certificates_issued": MetricDefinition(
        metric_id="total_certificates_issued",
        name="Total Proof-of-Impact Certificates Issued",
        description="Total active and verified cryptographic proof-of-impact certificates issued to donors.",
        formula="COUNT(DISTINCT ImpactCertificate.id) WHERE status IN ('ACTIVE', 'ISSUED')",
        data_source="impact_certificates",
        included_statuses=["ACTIVE", "ISSUED"],
        excluded_statuses=["REVOKED", "SUSPENDED"],
        currency_treatment="N/A",
        privacy=MetricPrivacy.PUBLIC,
        unit="count",
    ),
    "reconciliation_discrepancies_count": MetricDefinition(
        metric_id="reconciliation_discrepancies_count",
        name="Reconciliation Discrepancies Count",
        description="Count of open financial settlement discrepancies requiring administrative review.",
        formula="COUNT(ReconciliationItem.id) WHERE status == 'DISCREPANCY' AND is_resolved == FALSE",
        data_source="reconciliation_items",
        included_statuses=["DISCREPANCY"],
        excluded_statuses=["MATCHED", "RESOLVED"],
        currency_treatment="N/A",
        privacy=MetricPrivacy.ADMIN_ONLY,
        unit="count",
    ),
}


class MetricDefinitionService:
    """Provides lookup and metadata introspection for standard metrics."""

    @classmethod
    def get_metric(cls, metric_id: str) -> Optional[MetricDefinition]:
        return METRIC_CATALOG.get(metric_id)

    @classmethod
    def list_metrics(cls, privacy: Optional[MetricPrivacy] = None) -> List[MetricDefinition]:
        if privacy is None:
            return list(METRIC_CATALOG.values())
        return [m for m in METRIC_CATALOG.values() if m.privacy == privacy]

    @classmethod
    def format_amount(cls, amount_cents: int, currency: str = "USD") -> str:
        """Deterministically formats minor units into human-readable currency."""
        symbol = "$" if currency == "USD" else f"{currency} "
        dollars = amount_cents / 100.0
        return f"{symbol}{dollars:,.2f}"

