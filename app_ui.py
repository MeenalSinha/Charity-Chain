"""
app_ui.py - Streamlit Presentation Layer
========================================
UI presentation for donors, NGOs, and reviewers.
Separated from business logic and database access in charity_tracker.py.

Principle:
    AI analyzes. Evidence supports. Humans decide. Payment systems execute. CharityChain records.
"""

import streamlit as st
from charity_tracker import (
    Config, Logger, SecurityValidator,
    DatabaseManager, EnhancedAIVerifier, EnhancedIPFSManager,
    VisualizationManager, RoleManager, AIAssistant,
    BadgeSystem, AsyncIOManager, FOLIUM_AVAILABLE,
    apply_custom_css, init_session_state, add_dark_mode_toggle,
    create_project_map, show_nft_gallery, show_analytics_dashboard,
    generate_demo_data
)
from streamlit_folium import folium_static
import asyncio
from datetime import datetime, date, timedelta
import time
from typing import Optional, List, Dict, Any

from backend.app.db.session import SessionFactory
from backend.app.core.config import settings
from backend.app.core.constants import (
    PlatformRole, OrganizationRole, OrganizationStatus, OrganizationType,
    OrganizationDocumentType, ProjectStatus, MilestoneStatus, ProjectVisibility,
    DonationStatus, PaymentOrderStatus, PaymentCaptureStatus, PaymentRefundStatus,
    ReviewDecision, ReviewCaseStatus, ReviewAssignmentStatus,
    ReconciliationResult, DisputeStatus, DisputeCategory, DisputeResolution,
    PayoutRecipientStatus, RecipientType, DisbursementMethod, DisbursementStatus,
    LedgerAccountType, LedgerEntryType, ReconciliationRunStatus, ReconciliationStatus,
    DiscrepancyType, DiscrepancyResolution,
    CertificateType, CertificateStatus, CertificateRevocationReason, ImpactLedgerEventType
)
from backend.app.api.schemas import (
    CreateOrganizationRequest, UpdateOrganizationRequest, OrganizationDecisionRequest,
    CreateProjectRequest, UpdateProjectRequest, ProjectDecisionRequest,
    CreateMilestoneRequest, UpdateMilestoneRequest, CreateDonationRequest
)
from backend.app.services.ngo_onboarding_service import NGOOnboardingService
from backend.app.services.project_service import ProjectService
from backend.app.services.donation_service import DonationService
from backend.app.services.evidence_service import EvidenceService
from backend.app.services.verification_service import VerificationService
from backend.app.services.review_service import ReviewService
from backend.app.services.recipient_service import PayoutRecipientService
from backend.app.services.fund_allocation_service import FundAllocationService
from backend.app.services.ledger_service import LedgerService
from backend.app.services.disbursement_eligibility_service import DisbursementEligibilityService
from backend.app.services.disbursement_service import DisbursementService
from backend.app.services.reconciliation_service import ReconciliationService
from backend.app.services.notification_service import NotificationService
from backend.app.services.webhook_subscription_service import WebhookSubscriptionService
from backend.app.services.certificate_service import CertificateService
from backend.app.db.models.notifications import (
    Notification, NotificationPreference, WebhookSubscription, WebhookDelivery
)
from backend.app.core.constants import NotificationCategory, DomainEventType
from backend.app.db.models.financial import (
    Disbursement, PayoutRecipient, PayoutBatch, PayoutApproval,
    ReconciliationRun, ReconciliationItem, SettlementRecord
)
from backend.app.db.models.ledger import LedgerAccount, JournalEntry, JournalLine
from backend.app.integrations.storage.local import LocalStorageProvider
from backend.app.services.analytics import (
    PlatformAnalyticsService, NGOAnalyticsService, DonorAnalyticsService,
    PublicTransparencyService, GeographicAnalyticsService, AnalyticsExportService,
    MetricDefinitionService
)
from backend.app.db.models.analytics import (
    AnalyticsSnapshot, AnalyticsExportJob, PublicTransparencyPublication
)


# ============================================================================
# HELPER: CURRENT AUTH CONTEXT
# ============================================================================

def get_current_user_info():
    """Extract authenticated user details from session state."""
    auth_user = st.session_state.get('auth_user')
    if auth_user:
        return auth_user.get('id'), auth_user.get('role'), auth_user.get('full_name'), auth_user.get('email')
    return None, 'Guest', 'Guest User', ''


# ============================================================================
# PAGE: HOME
# ============================================================================

def page_home():
    """Home page with verified platform metrics and workflow overview."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🔒 Milestone-Locked Funds</h3>
            <p>Application-managed conditional release</p>
            <small>Evidence-backed release • PayPal ready</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🤖 AI Evidence Analysis</h3>
            <p>Multi-indicator verification</p>
            <small>YOLOv8 • pHash • ELA • EXIF GPS</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>📜 Proof-of-Impact</h3>
            <p>Verifiable Digital Certificates</p>
            <small>Cryptographically signed audit records</small>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Platform Statistics (Authoritative PostgreSQL Domain Models)
    st.subheader("📈 Live Platform Overview")
    
    total_projects = 0
    total_captured_formatted = "$0.00"
    completed_milestones = 0
    verified_orgs = 0
    
    try:
        with SessionFactory() as db_session:
            overview = PlatformAnalyticsService.get_platform_overview(db_session)
            total_projects = overview["projects"]["total"]
            total_captured_formatted = overview["financial"]["captured_formatted"]
            completed_milestones = overview["milestones"]["completed"]
            verified_orgs = overview["organizations"]["verified"]
    except Exception:
        db_mgr: DatabaseManager = st.session_state.db
        total_projects = len(db_mgr.get_all_milestones()) if db_mgr else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Projects", total_projects)
    with col2:
        st.metric("Total Captured", total_captured_formatted)
    with col3:
        st.metric("Completed Milestones", completed_milestones)
    with col4:
        st.metric("Verified NGOs", verified_orgs)
    
    st.markdown("---")
    st.info("💡 **CharityChain Workflow:** Donors fund milestones ➔ NGOs submit photographic evidence ➔ AI analyzes authenticity ➔ Reviewers approve ➔ Funds released ➔ Donors receive Proof-of-Impact certificates.")


# ============================================================================
# PAGE: ORGANIZATION ONBOARDING & PROFILE MANAGEMENT (Phase 3)
# ============================================================================

def page_organization_onboarding():
    """NGO Organization Onboarding, Document Upload, and Profile Management."""
    st.header("🏢 NGO Organization Onboarding & Management")
    
    user_id, role, full_name, email = get_current_user_info()
    if not user_id:
        st.warning("🔒 Please sign in via the sidebar to register or manage your NGO.")
        return

    tab_my_orgs, tab_new_org = st.tabs(["📋 My Organizations", "➕ Register New Organization"])

    with tab_new_org:
        st.subheader("Register a New Organization")
        st.write("Complete the initial legal profile. Once registered, you will be prompted to upload verification documents.")
        
        with st.form("form_register_org"):
            col1, col2 = st.columns(2)
            with col1:
                legal_name = st.text_input("Legal Organization Name *", placeholder="Hope For Wildlife International")
                org_type_choice = st.selectbox(
                    "Organization Legal Structure *",
                    options=[t.value for t in OrganizationType],
                    index=0
                )
                reg_authority = st.text_input("Registration Authority *", placeholder="UK Charity Commission / US IRS")
                reg_number = st.text_input("Registration Number *", placeholder="CR-98765432")
                tax_id = st.text_input("Tax ID / EIN", placeholder="EIN-12-3456789")
                country = st.text_input("Country of Registration (ISO-2) *", value="US", max_chars=2)
            
            with col2:
                reg_address = st.text_input("Registered Address *", placeholder="100 Impact Way, Suite 400")
                operating_region = st.text_input("Primary Operating Region *", placeholder="East Africa & Southeast Asia")
                website = st.text_input("Official Website URL", placeholder="https://hopewildlife.org")
                contact_email = st.text_input("Public Contact Email", value=email)
                contact_phone = st.text_input("Contact Phone Number", placeholder="+1-555-0199")

            mission = st.text_area("Mission Statement *", placeholder="To conserve vulnerable wildlife ecosystems through community-led protection programs.")
            vision = st.text_area("Vision Statement", placeholder="A world where human communities and wildlife thrive symbiotically.")
            focus_areas = st.text_input("Focus Areas (Comma-separated)", placeholder="Wildlife, Conservation, Rainforest")

            submit_org = st.form_submit_button("Register Organization", type="primary", use_container_width=True)

            if submit_org:
                if not legal_name or not reg_authority or not reg_number or not country or not reg_address or not operating_region or not mission:
                    st.error("Please fill in all required fields marked with *.")
                else:
                    try:
                        focus_list = [f.strip() for f in focus_areas.split(",") if f.strip()]
                        req = CreateOrganizationRequest(
                            legal_name=legal_name,
                            organization_type=OrganizationType(org_type_choice),
                            registration_authority=reg_authority,
                            registration_number=reg_number,
                            tax_identification_number=tax_id or None,
                            country_of_registration=country.upper(),
                            registered_address=reg_address,
                            operating_address=reg_address,
                            primary_operating_region=operating_region,
                            website_url=website or None,
                            contact_email=contact_email or None,
                            contact_phone=contact_phone or None,
                            mission_statement=mission,
                            vision_statement=vision or None,
                            focus_areas=focus_list
                        )
                        with SessionFactory() as db:
                            new_org = NGOOnboardingService.create_organization(db, req, user_id)
                        st.success(f"✅ Organization **{new_org.legal_name}** successfully registered in DRAFT state!")
                        st.balloons()
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error registering organization: {e}")

    with tab_my_orgs:
        with SessionFactory() as db:
            orgs = NGOOnboardingService.list_user_organizations(db, user_id)
        
        if not orgs:
            st.info("You do not currently manage any organizations. Use the 'Register New Organization' tab to begin.")
            return

        org_options = {o.id: f"{o.legal_name} ({o.status})" for o in orgs}
        selected_org_id = st.selectbox(
            "Select Organization to Manage",
            options=list(org_options.keys()),
            format_func=lambda x: org_options[x]
        )

        with SessionFactory() as db:
            org = NGOOnboardingService.get_organization_details(db, selected_org_id, user_id)
            docs = NGOOnboardingService.list_organization_documents(db, selected_org_id, user_id)

        # Status Banner
        status_colors = {
            OrganizationStatus.DRAFT.value: "orange",
            OrganizationStatus.VERIFICATION_PENDING.value: "blue",
            OrganizationStatus.VERIFIED.value: "green",
            OrganizationStatus.REVISION_REQUIRED.value: "orange",
            OrganizationStatus.REJECTED.value: "red",
            OrganizationStatus.SUSPENDED.value: "gray"
        }
        st.markdown(f"### Status: :{status_colors.get(org.status, 'blue')}[{org.status}]")
        
        if org.status == OrganizationStatus.DRAFT.value:
            st.info("ℹ️ Your organization is currently in **DRAFT** state. Please upload official verification documents below and submit for reviewer verification.")
        elif org.status == OrganizationStatus.VERIFICATION_PENDING.value:
            st.warning("⏳ Your verification application is pending inspection by authorized CharityChain reviewers.")
        elif org.status == OrganizationStatus.VERIFIED.value:
            st.success("⭐ Congratulations! Your organization is **VERIFIED**. You can now create and submit charitable projects.")
        elif org.status == OrganizationStatus.REVISION_REQUIRED.value:
            st.error(f"⚠️ Revisions requested by reviewer: **{org.review_notes or 'Please update documentation.'}**")
        elif org.status == OrganizationStatus.REJECTED.value:
            st.error(f"❌ Application rejected: **{org.rejection_reason or 'Criteria not met.'}**")

        st.markdown("---")
        
        col_info, col_docs = st.columns([1, 1])

        with col_info:
            st.subheader("Organization Profile")
            st.markdown(f"**Legal Name:** {org.legal_name}")
            st.markdown(f"**Type:** `{org.organization_type}`")
            st.markdown(f"**Registration:** {org.registration_authority} (#{org.registration_number})")
            st.markdown(f"**Country:** {org.country_of_registration} | **Region:** {org.primary_operating_region}")
            st.markdown(f"**Address:** {org.registered_address}")
            if org.website_url:
                st.markdown(f"**Website:** [{org.website_url}]({org.website_url})")
            st.markdown(f"**Contact:** {org.contact_email or 'N/A'}")
            st.markdown(f"**Mission:** {org.mission_statement}")
            if org.focus_areas:
                st.markdown(f"**Focus Areas:** {', '.join(org.focus_areas)}")

        with col_docs:
            st.subheader("Verification Documents")
            if docs:
                for d in docs:
                    st.markdown(f"- **{d.document_type}**: `{d.original_filename}` ({d.review_status})")
                    st.caption(f"SHA-256: `{d.file_hash_sha256[:16]}...` • Size: {d.file_size_bytes / 1024:.1f} KB")
            else:
                st.write("No documents uploaded yet.")

            st.markdown("#### Upload Supporting Document")
            with st.form(f"upload_doc_{selected_org_id}"):
                doc_type_choice = st.selectbox(
                    "Document Type",
                    options=[t.value for t in OrganizationDocumentType]
                )
                uploaded_file = st.file_uploader("Select File (PDF, PNG, JPEG, max 10MB)", type=['pdf', 'png', 'jpg', 'jpeg'])
                btn_upload = st.form_submit_button("Upload Document", use_container_width=True)

                if btn_upload and uploaded_file:
                    content_bytes = uploaded_file.getvalue()
                    storage = LocalStorageProvider(settings.STORAGE_DOCUMENTS_DIR)
                    try:
                        with SessionFactory() as db:
                            new_doc = NGOOnboardingService.upload_verification_document(
                                db=db,
                                organization_id=selected_org_id,
                                user_id=user_id,
                                document_type=doc_type_choice,
                                filename=uploaded_file.name,
                                content=content_bytes,
                                storage_provider=storage
                            )
                        st.success(f"✅ Uploaded `{uploaded_file.name}` as {doc_type_choice}!")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Upload failed: {e}")

        st.markdown("---")
        
        # Submit for verification button
        if org.status in [OrganizationStatus.DRAFT.value, OrganizationStatus.REVISION_REQUIRED.value]:
            st.subheader("Submit for Verification")
            st.write("Submit your profile and uploaded documents for formal reviewer inspection.")
            if not docs:
                st.warning("⚠️ You must upload at least one verification document before submitting.")
            else:
                if st.button("🚀 Submit Organization for Verification", type="primary", use_container_width=True):
                    try:
                        with SessionFactory() as db:
                            res = NGOOnboardingService.submit_for_verification(db, selected_org_id, user_id)
                        st.success(f"✅ Organization submitted for verification! Current status: {res.status}")
                        st.balloons()
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Submission failed: {e}")


# ============================================================================
# PAGE: PROJECT MANAGEMENT & MULTI-MILESTONE CREATION (Phase 3)
# ============================================================================

def page_project_management():
    """Create, edit, validate, and manage multi-milestone charitable projects."""
    st.header("📋 Project & Milestone Management")
    
    user_id, role, full_name, email = get_current_user_info()
    if not user_id:
        st.warning("🔒 Please sign in via the sidebar to manage projects.")
        return

    # Check user organizations
    with SessionFactory() as db:
        user_orgs = NGOOnboardingService.list_user_organizations(db, user_id)
    
    if not user_orgs:
        st.warning("⚠️ You must register an Organization before creating charitable projects.")
        st.info("Navigate to 'Organization Onboarding' to register your NGO.")
        return

    org_map = {o.id: f"{o.legal_name} ({o.status})" for o in user_orgs}
    active_org_id = st.selectbox(
        "Active Organization",
        options=list(org_map.keys()),
        format_func=lambda x: org_map[x]
    )

    tab_projects, tab_new_proj = st.tabs(["📂 Organization Projects", "➕ Create New Project"])

    with tab_new_proj:
        st.subheader("Create a Multi-Milestone Charitable Project")
        
        with SessionFactory() as db:
            categories = ProjectService.list_categories(db)
        
        cat_map = {c.id: c.name for c in categories}
        
        with st.form("form_create_project"):
            col1, col2 = st.columns(2)
            with col1:
                p_title = st.text_input("Project Title *", placeholder="Clean Water Boreholes in Turkana Basin")
                p_cat_id = st.selectbox("Project Category *", options=list(cat_map.keys()), format_func=lambda x: cat_map[x])
                p_target = st.number_input("Total Project Funding Goal (USD) *", min_value=10.0, value=15000.0, step=500.0)
                p_currency = st.selectbox("Currency *", ["USD", "EUR", "GBP", "ETH"], index=0)
                p_visibility = st.selectbox("Visibility", ["PUBLIC", "UNLISTED"], index=0)

            with col2:
                p_start = st.date_input("Planned Start Date", value=date.today())
                p_end = st.date_input("Planned End Date", value=date.today() + timedelta(days=180))
                p_country = st.text_input("Primary Country (ISO-2) *", value="KE", max_chars=2)
                p_region = st.text_input("Operating Region *", value="Turkana County")
                p_beneficiary_loc = st.text_input("Beneficiary Site", value="Lodwar Rural Community")
                p_beneficiaries = st.number_input("Expected Beneficiaries", min_value=1, value=5000)

            p_summary = st.text_area("Executive Summary *", placeholder="Constructing three high-yield solar-powered water boreholes to provide 5,000 pastoralists with potable water.")
            p_objectives = st.text_area("Key Project Objectives *", placeholder="1. Geophysical survey and aquifer testing\n2. Solar pumping installation\n3. Community water committee training")

            btn_submit_proj = st.form_submit_button("Initialize Project (Draft)", type="primary", use_container_width=True)

            if btn_submit_proj:
                if not p_title or not p_summary or not p_objectives or not p_country or not p_region:
                    st.error("Please complete all required fields marked with *.")
                else:
                    try:
                        minor_units = int(round(p_target * 100))
                        req = CreateProjectRequest(
                            organization_id=active_org_id,
                            category_id=p_cat_id,
                            title=p_title,
                            summary=p_summary,
                            objectives=p_objectives,
                            project_type="COMMUNITY_INFRASTRUCTURE",
                            public_visibility=ProjectVisibility(p_visibility),
                            funding_target_minor=minor_units,
                            currency=p_currency,
                            country=p_country.upper(),
                            primary_operating_region=p_region,
                            beneficiary_location=p_beneficiary_loc,
                            expected_beneficiaries=p_beneficiaries,
                            start_date=p_start.isoformat(),
                            end_date=p_end.isoformat()
                        )
                        with SessionFactory() as db:
                            new_p = ProjectService.create_project(db, req, user_id)
                        st.success(f"✅ Project **{new_p.title}** created successfully! Now add milestones below.")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error creating project: {e}")

    with tab_projects:
        with SessionFactory() as db:
            org_projects = ProjectService.list_organization_projects(db, active_org_id, user_id)
        
        if not org_projects:
            st.info("No projects found for this organization. Use the 'Create New Project' tab to create your first project.")
            return

        proj_options = {p.id: f"{p.title} [{p.status}]" for p in org_projects}
        sel_proj_id = st.selectbox("Select Project to Manage", options=list(proj_options.keys()), format_func=lambda x: proj_options[x])

        with SessionFactory() as db:
            project = ProjectService.get_project_details(db, sel_proj_id, user_id)
            milestones = ProjectService.list_project_milestones(db, sel_proj_id, user_id)
            val_report = ProjectService.validate_project_submission(db, sel_proj_id, user_id)

        # Overview Card
        st.markdown(f"### Project: {project.title}")
        col_p1, col_p2, col_p3 = st.columns(3)
        with col_p1:
            st.metric("Status", project.status)
        with col_p2:
            st.metric("Target Goal", f"${project.funding_target_minor/100:,.2f} {project.currency}")
        with col_p3:
            st.metric("Total Milestones", len(milestones))

        st.caption(f"Summary: {project.summary}")
        
        st.markdown("---")
        
        # Milestones Section
        st.subheader("🎯 Project Milestones")
        allocated_total = sum(m.funding_target_minor for m in milestones)
        target_total = project.funding_target_minor
        pct = min(100.0, (allocated_total / target_total * 100.0)) if target_total > 0 else 0.0

        st.progress(pct / 100.0)
        st.caption(f"Allocated: ${allocated_total/100:,.2f} of ${target_total/100:,.2f} {project.currency} ({pct:.1f}%)")

        if milestones:
            for m in milestones:
                with st.expander(f"Milestone #{m.sequence_number}: {m.title} — ${m.funding_target_minor/100:,.2f} [{m.status}]"):
                    st.write(f"**Objective:** {m.objective}")
                    st.write(f"**Deliverables:** {m.deliverables or 'N/A'}")
                    st.write(f"**Success Criteria:** {m.success_criteria or 'N/A'}")
                    if m.target_latitude and m.target_longitude:
                        st.caption(f"📍 Target Coordinates: {m.target_latitude:.5f}, {m.target_longitude:.5f}")
        else:
            st.warning("⚠️ No milestones created yet. Every project requires at least one milestone.")

        # Add Milestone Form
        with st.expander("➕ Add New Milestone to this Project"):
            next_seq = (max([m.sequence_number for m in milestones]) + 1) if milestones else 1
            remaining = max(0, target_total - allocated_total)
            
            with st.form("form_add_milestone"):
                col_m1, col_m2 = st.columns(2)
                with col_m1:
                    m_title = st.text_input("Milestone Title *", placeholder="Geophysical Survey & Drilling Rig Setup")
                    m_seq = st.number_input("Sequence Number *", min_value=1, value=next_seq, step=1)
                    m_target_usd = st.number_input("Milestone Target Amount (USD) *", min_value=1.0, value=float(remaining / 100) if remaining > 0 else 5000.0, step=100.0)
                with col_m2:
                    m_lat = st.number_input("Expected Latitude", value=3.116667, format="%.6f")
                    m_lon = st.number_input("Expected Longitude", value=35.599998, format="%.6f")

                m_objective = st.text_area("Milestone Objective *", placeholder="Conduct site hydrogeological survey and drill exploratory boreholes to 120m depth.")
                m_deliverables = st.text_area("Concrete Deliverables", placeholder="Hydrological report, borehole log, water purity lab test results.")
                m_criteria = st.text_area("Success Criteria", placeholder="Water yield exceeds 15 m3/hr with potable salinity levels.")

                btn_add_m = st.form_submit_button("Add Milestone", type="primary", use_container_width=True)

                if btn_add_m:
                    if not m_title or not m_objective:
                        st.error("Milestone title and objective are required.")
                    else:
                        try:
                            m_minor = int(round(m_target_usd * 100))
                            m_req = CreateMilestoneRequest(
                                sequence_number=m_seq,
                                title=m_title,
                                objective=m_objective,
                                deliverables=m_deliverables or None,
                                success_criteria=m_criteria or None,
                                funding_target_minor=m_minor,
                                currency=project.currency,
                                target_latitude=m_lat if m_lat != 0 else None,
                                target_longitude=m_lon if m_lon != 0 else None
                            )
                            with SessionFactory() as db:
                                created_m = ProjectService.create_milestone(db, sel_proj_id, m_req, user_id)
                            st.success(f"✅ Milestone #{created_m.sequence_number} '{created_m.title}' added!")
                            time.sleep(1)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error adding milestone: {e}")

        st.markdown("---")

        # Live Pre-Submission Validation Checklist
        st.subheader("🔍 Pre-Submission Validation Report")
        if val_report.is_valid:
            st.success("✅ **All Pre-Submission Checks Passed!** This project is fully compliant and ready to submit for reviewer approval.")
        else:
            st.warning(f"⚠️ **{len(val_report.issues)} compliance issue(s) detected.** Please address them before submitting:")
            for issue in val_report.issues:
                st.error(f"- **[{issue.field}]** {issue.message} (`{issue.code}`)")

        st.markdown("---")

        # Lifecycle Transitions
        st.subheader("⚡ Project Lifecycle Controls")
        col_act1, col_act2, col_act3 = st.columns(3)
        
        with col_act1:
            if project.status in [ProjectStatus.DRAFT.value, ProjectStatus.REVISION_REQUIRED.value]:
                if st.button("🚀 Submit Project for Review", type="primary", disabled=not val_report.is_valid, use_container_width=True):
                    try:
                        with SessionFactory() as db:
                            res = ProjectService.submit_project(db, sel_proj_id, user_id)
                        st.success(f"✅ Project submitted! Status: {res.status}")
                        st.balloons()
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Submission error: {e}")

        with col_act2:
            if project.status == ProjectStatus.UNDER_REVIEW.value:
                if st.button("↩️ Withdraw Submission", use_container_width=True):
                    try:
                        with SessionFactory() as db:
                            res = ProjectService.withdraw_project(db, sel_proj_id, user_id)
                        st.info("Submission withdrawn back to DRAFT.")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Withdrawal error: {e}")

        with col_act3:
            if project.status == ProjectStatus.APPROVED.value:
                if st.button("🟢 Activate Project", type="primary", use_container_width=True):
                    try:
                        with SessionFactory() as db:
                            res = ProjectService.activate_project(db, sel_proj_id, user_id)
                        st.success("Project activated! Now publicly discoverable for donations.")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Activation error: {e}")


# ============================================================================
# PAGE: REVIEWER VERIFICATION QUEUE (Phase 3)
# ============================================================================

def page_reviewer_queue():
    """Reviewer & Administrator verification interface for NGOs and Projects."""
    st.header("⚖️ Reviewer Verification Queue")

    user_id, role, full_name, email = get_current_user_info()
    if role not in [PlatformRole.REVIEWER.value, PlatformRole.PLATFORM_ADMIN.value]:
        st.warning("🔒 Reviewer verification queue requires `REVIEWER` or `PLATFORM_ADMIN` platform role.")
        return

    tab_orgs, tab_projs, tab_milestones, tab_my_queue, tab_disputes, tab_audit = st.tabs([
        "🏢 Organizations",
        "📋 Projects",
        "🔍 Milestone Review Cases",
        "👤 My Assigned Reviews",
        "⚖️ Formal Disputes",
        "📜 Cryptographic Audit"
    ])

    with tab_orgs:
        st.subheader("Pending Organization Verifications")
        with SessionFactory() as db:
            queue = NGOOnboardingService.get_review_queue(db, page=1, page_size=50)

        if not queue.items:
            st.info("🎉 No organizations currently awaiting verification.")
        else:
            st.markdown(f"**{queue.total_count} organization(s) in review queue**")
            
            sel_rev_org_id = st.selectbox(
                "Select Organization to Inspect",
                options=[o.id for o in queue.items],
                format_func=lambda x: next(f"{o.legal_name} ({o.country_of_registration}) — {o.status}" for o in queue.items if o.id == x)
            )

            with SessionFactory() as db:
                org_detail = NGOOnboardingService.get_organization_details(db, sel_rev_org_id, user_id)
                org_docs = NGOOnboardingService.list_organization_documents(db, sel_rev_org_id, user_id)
                # Check conflict of interest
                from backend.app.db.models.users import OrganizationMember
                from sqlalchemy import select
                coi_check = db.execute(
                    select(OrganizationMember).where(
                        OrganizationMember.organization_id == sel_rev_org_id,
                        OrganizationMember.user_id == user_id
                    )
                ).scalar_one_or_none()

            col_o1, col_o2 = st.columns([1, 1])
            with col_o1:
                st.markdown(f"**Legal Name:** {org_detail.legal_name}")
                st.markdown(f"**Structure:** `{org_detail.organization_type}`")
                st.markdown(f"**Reg Authority:** {org_detail.registration_authority} (#{org_detail.registration_number})")
                st.markdown(f"**Country:** {org_detail.country_of_registration} | **Region:** {org_detail.primary_operating_region}")
                st.markdown(f"**Address:** {org_detail.registered_address}")
                st.markdown(f"**Mission:** {org_detail.mission_statement}")
                if org_detail.website_url:
                    st.markdown(f"**Website:** [{org_detail.website_url}]({org_detail.website_url})")

            with col_o2:
                st.markdown("#### Uploaded Verification Documents")
                if org_docs:
                    for doc in org_docs:
                        st.markdown(f"- **{doc.document_type}**: `{doc.original_filename}`")
                        st.caption(f"Hash: `{doc.file_hash_sha256[:16]}...` • Size: {doc.file_size_bytes/1024:.1f} KB")
                else:
                    st.warning("⚠️ No documents uploaded!")

            st.markdown("---")

            if coi_check:
                st.error("⚠️ **Conflict of Interest**: You are an active member of this organization. CharityChain governance strictly prohibits reviewing or verifying your own organization.")
            else:
                st.subheader("Reviewer Decision")
                with st.form(f"decision_org_{sel_rev_org_id}"):
                    dec = st.radio("Decision", ["APPROVE", "REQUEST_REVISION", "REJECT"], horizontal=True)
                    notes = st.text_area("Reviewer Notes / Feedback *", placeholder="All legal registration checks verified against official government registry.")
                    btn_dec = st.form_submit_button("Submit Verification Decision", type="primary", use_container_width=True)

                    if btn_dec:
                        if not notes:
                            st.error("Reviewer notes are required.")
                        else:
                            try:
                                d_req = OrganizationDecisionRequest(decision=dec, review_notes=notes)
                                with SessionFactory() as db:
                                    res = NGOOnboardingService.review_organization(db, sel_rev_org_id, d_req, user_id)
                                st.success(f"✅ Organization decision recorded! Status: {res.status}")
                                time.sleep(1)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {e}")

    with tab_projs:
        st.subheader("Pending Project Submissions")
        with SessionFactory() as db:
            p_queue = ProjectService.get_review_queue(db, page=1, page_size=50)

        if not p_queue.items:
            st.info("🎉 No projects currently awaiting review.")
        else:
            st.markdown(f"**{p_queue.total_count} project(s) in review queue**")
            sel_rev_p_id = st.selectbox(
                "Select Project to Inspect",
                options=[p.id for p in p_queue.items],
                format_func=lambda x: next(f"{p.title} (${p.funding_target_minor/100:,.2f}) — {p.status}" for p in p_queue.items if p.id == x)
            )

            with SessionFactory() as db:
                p_detail = ProjectService.get_project_details(db, sel_rev_p_id, user_id)
                p_milestones = ProjectService.list_project_milestones(db, sel_rev_p_id, user_id)
                val = ProjectService.validate_project_submission(db, sel_rev_p_id, user_id)
                # Check COI
                from backend.app.db.models.users import OrganizationMember
                from sqlalchemy import select
                coi_proj = db.execute(
                    select(OrganizationMember).where(
                        OrganizationMember.organization_id == p_detail.organization_id,
                        OrganizationMember.user_id == user_id
                    )
                ).scalar_one_or_none()

            st.markdown(f"**Title:** {p_detail.title}")
            st.markdown(f"**Funding Goal:** ${p_detail.funding_target_minor/100:,.2f} {p_detail.currency}")
            st.markdown(f"**Summary:** {p_detail.summary}")
            st.markdown(f"**Objectives:** {p_detail.objectives}")

            st.markdown("#### Milestones Breakdown")
            for m in p_milestones:
                st.markdown(f"- **#{m.sequence_number}: {m.title}** — ${m.funding_target_minor/100:,.2f} {m.currency}")
                st.caption(f"Deliverables: {m.deliverables} • Criteria: {m.success_criteria}")

            st.markdown("---")

            if coi_proj:
                st.error("⚠️ **Conflict of Interest**: You are an active member of the submitting organization and cannot review this project.")
            else:
                st.subheader("Reviewer Project Decision")
                with st.form(f"decision_proj_{sel_rev_p_id}"):
                    p_dec = st.radio("Decision", ["APPROVE", "REQUEST_REVISION", "REJECT"], horizontal=True)
                    p_notes = st.text_area("Reviewer Notes / Feedback *", placeholder="Project milestones and deliverables verified against regional feasibility parameters.")
                    btn_p_dec = st.form_submit_button("Submit Project Decision", type="primary", use_container_width=True)

                    if btn_p_dec:
                        if not p_notes:
                            st.error("Reviewer notes are required.")
                        else:
                            try:
                                dec_req = ProjectDecisionRequest(decision=p_dec, review_notes=p_notes)
                                with SessionFactory() as db:
                                    res = ProjectService.review_project(db, sel_rev_p_id, dec_req, user_id)
                                st.success(f"✅ Project decision recorded! Status: {res.status}")
                                time.sleep(1)
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {e}")

    with tab_milestones:
        st.subheader("Milestone Review Cases (Dual-Key Governance)")
        with SessionFactory() as db:
            from backend.app.db.models.reviews_impact import ReviewCase, ReviewTask
            from backend.app.db.models.users import User
            all_cases = db.execute(select(ReviewCase).order_by(ReviewCase.created_at.desc())).scalars().all()
            eligible_reviewers = db.execute(
                select(User).where(User.role.in_([PlatformRole.REVIEWER, PlatformRole.PLATFORM_ADMIN]))
            ).scalars().all()

        if not all_cases:
            st.info("No milestone review cases submitted yet.")
        else:
            for c in all_cases:
                with st.expander(f"Case `{c.id[:8]}...` • Round #{c.submission_round} • Status: {c.status}", expanded=False):
                    st.markdown(f"**Milestone ID:** `{c.milestone_id}` | **Organization ID:** `{c.organization_id}`")
                    st.markdown(f"**Snapshot Digest:** `{c.snapshot_digest}`")
                    st.markdown(f"**Status:** `{c.status}` | **Final Decision:** `{c.final_decision or 'Pending'}`")
                    if c.submission_notes:
                        st.caption(f"NGO Notes: {c.submission_notes}")

                    # Admin Assignment controls
                    if role == PlatformRole.PLATFORM_ADMIN.value and c.status in ("SUBMITTED", "ESCALATED"):
                        st.markdown("##### 👥 Assign Dual Independent Reviewers")
                        rev_options = {u.id: f"{u.full_name or u.email} ({u.role.value})" for u in eligible_reviewers}
                        col_r1, col_r2 = st.columns(2)
                        with col_r1:
                            r1_id = st.selectbox(f"Reviewer 1 (Slot 1) - Case {c.id[:8]}", options=list(rev_options.keys()), format_func=lambda x: rev_options[x], key=f"r1_{c.id}")
                        with col_r2:
                            r2_id = st.selectbox(f"Reviewer 2 (Slot 2) - Case {c.id[:8]}", options=list(rev_options.keys()), format_func=lambda x: rev_options[x], key=f"r2_{c.id}")

                        if st.button("Assign Reviewers", key=f"btn_assign_{c.id}"):
                            try:
                                with SessionFactory() as db:
                                    admin_u = db.get(User, user_id)
                                    ReviewService.assign_reviewers(
                                        db=db,
                                        case_id=c.id,
                                        admin_user=admin_u,
                                        reviewer_1_id=r1_id,
                                        reviewer_2_id=r2_id
                                    )
                                st.success("✅ Dual reviewers assigned successfully!")
                                time.sleep(1)
                                st.rerun()
                            except Exception as ex:
                                st.error(f"❌ Assignment failed: {ex}")

    with tab_my_queue:
        st.subheader("👤 My Assigned Review Tasks")
        with SessionFactory() as db:
            from backend.app.db.models.users import User
            curr_u = db.get(User, user_id)
            my_tasks = ReviewService.get_reviewer_queue(db=db, reviewer=curr_u)

        if not my_tasks:
            st.info("You have no pending assigned review tasks.")
        else:
            sel_task_id = st.selectbox(
                "Select Assigned Task to Inspect",
                options=[t["task_id"] for t in my_tasks],
                format_func=lambda x: next(f"Slot {t['slot']} • {t['project_title']} - {t['milestone_title']} [{t['assignment_state']}]" for t in my_tasks if t["task_id"] == x)
            )
            selected_task = next(t for t in my_tasks if t["task_id"] == sel_task_id)

            col_t1, col_t2 = st.columns(2)
            with col_t1:
                st.markdown(f"**Project:** {selected_task['project_title']}")
                st.markdown(f"**Milestone:** {selected_task['milestone_title']}")
                st.markdown(f"**Organization:** {selected_task['organization_name']}")
            with col_t2:
                st.markdown(f"**Slot:** #{selected_task['slot']} | **Assignment Status:** `{selected_task['assignment_state']}`")
                st.markdown(f"**Due Date:** `{selected_task['due_at']}`")

            # Actions if ASSIGNED
            if selected_task["assignment_state"] == "ASSIGNED":
                col_a1, col_a2, col_a3 = st.columns(3)
                with col_a1:
                    if st.button("Accept Assignment", key=f"acc_{sel_task_id}", type="primary"):
                        with SessionFactory() as db:
                            ReviewService.accept_assignment(db, sel_task_id, curr_u)
                        st.success("Assignment accepted! You may now review evidence.")
                        time.sleep(1)
                        st.rerun()
                with col_a2:
                    dec_reason = st.text_input("Decline Reason", key=f"dec_r_{sel_task_id}")
                    if st.button("Decline", key=f"dec_{sel_task_id}"):
                        if dec_reason:
                            with SessionFactory() as db:
                                ReviewService.decline_assignment(db, sel_task_id, curr_u, dec_reason)
                            st.info("Assignment declined.")
                            time.sleep(1)
                            st.rerun()
                with col_a3:
                    coi_reason = st.text_input("COI Reason", key=f"coi_r_{sel_task_id}")
                    if st.button("Declare Conflict", key=f"coi_{sel_task_id}"):
                        if coi_reason:
                            with SessionFactory() as db:
                                ReviewService.declare_conflict(db, sel_task_id, curr_u, coi_reason)
                            st.warning("Conflict of interest recorded.")
                            time.sleep(1)
                            st.rerun()

            elif selected_task["assignment_state"] in ("ACCEPTED", "IN_PROGRESS"):
                st.markdown("---")
                st.markdown("### 📋 Evidence Package & Phase 6 AI Forensic Analysis")
                with SessionFactory() as db:
                    case_detail = ReviewService.get_case_detail_for_reviewer(db, selected_task["case_id"], curr_u)

                st.markdown(f"**Submission Round:** #{case_detail['submission_round']}")
                if case_detail.get("submission_notes"):
                    st.info(f"NGO Notes: {case_detail['submission_notes']}")

                # Display evidence files
                for ev_item in case_detail["evidence_package"]:
                    with st.expander(f"📄 {ev_item['original_filename']} (v{ev_item['version']})", expanded=True):
                        st.caption(f"SHA-256 Hash: `{ev_item['file_hash']}`")
                        vr = ev_item.get("verification_report")
                        if vr:
                            st.markdown(f"**AI Assessment:** `{vr['overall_assessment']}` | **Confidence:** `{vr['confidence_score']}` ({vr['calibration_status']})")
                            st.markdown(f"*{vr['summary_notes']}*")
                            for chk in vr.get("checks", []):
                                st.markdown(f"- **{chk['check_type']}**: `{chk['status']}` — {chk['message']}")
                        else:
                            st.warning("⚠️ No AI verification report available for this file.")

                # Decisions & Blind Review Notice
                st.markdown("---")
                st.markdown("### ⚖️ Independent Review Decision Form")
                if case_detail.get("blind_review_active"):
                    st.info("🔒 **Blind Review Active**: Reviewer 1's decision and comments are masked to ensure independent, unanchored evaluation.")

                with st.form(f"form_decision_{sel_task_id}"):
                    dec_val = st.radio("Your Independent Decision", ["APPROVE", "REQUEST_REVISION", "REJECT"], horizontal=True)
                    dec_comments = st.text_area("Review Comments & Observations *", placeholder="Document physical progress observed, compliance with criteria, or required corrections.")
                    dec_rev_data = st.text_input("Structured Revision Requirements (if REQUEST_REVISION)", placeholder="e.g. Higher resolution photo of well output pressure gauge")
                    btn_submit_dec = st.form_submit_button("Submit Immutable Decision", type="primary", use_container_width=True)

                    if btn_submit_dec:
                        try:
                            rev_data = {"required_actions": [dec_rev_data]} if dec_rev_data else None
                            with SessionFactory() as db:
                                ReviewService.submit_decision(
                                    db=db,
                                    task_id=sel_task_id,
                                    reviewer=curr_u,
                                    decision=ReviewDecision(dec_val),
                                    comments=dec_comments,
                                    structured_revision_data=rev_data
                                )
                            st.success("✅ Decision recorded and dual-key reconciled!")
                            time.sleep(1)
                            st.rerun()
                        except Exception as ex:
                            st.error(f"❌ Decision submission failed: {ex}")

            elif selected_task["assignment_state"] == "DECIDED":
                st.success("✅ You have submitted your immutable decision for this review task slot.")

    with tab_disputes:
        st.subheader("⚖️ Formal Milestone Appeals & Disputes")
        with SessionFactory() as db:
            from backend.app.db.models.reviews_impact import ReviewDispute
            disputes = db.execute(select(ReviewDispute).order_by(ReviewDispute.created_at.desc())).scalars().all()

        if not disputes:
            st.info("No formal disputes or appeals have been raised.")
        else:
            for disp in disputes:
                with st.expander(f"Dispute `{disp.id[:8]}` • Category: {disp.dispute_category} • Status: {disp.status}", expanded=False):
                    st.markdown(f"**Milestone ID:** `{disp.milestone_id}` | **Organization ID:** `{disp.organization_id}`")
                    st.markdown(f"**Dispute Reason:** {disp.dispute_reason}")
                    if disp.resolution:
                        st.success(f"**Resolution:** `{disp.resolution}` — {disp.resolution_explanation}")

                    # Thread messages
                    if disp.messages:
                        st.markdown("##### 💬 Discussion Thread")
                        for m in disp.messages:
                            st.markdown(f"**{m.sender_role}**: {m.message}")
                            st.caption(f"Time: {m.created_at}")

                    # Add message form
                    with st.form(f"msg_disp_{disp.id}"):
                        msg_text = st.text_input("Add Statement / Response", key=f"inp_msg_{disp.id}")
                        if st.form_submit_button("Send"):
                            if msg_text:
                                with SessionFactory() as db:
                                    ReviewService.add_dispute_message(db, disp.id, curr_u, msg_text)
                                st.success("Message recorded.")
                                time.sleep(1)
                                st.rerun()

                    # Admin resolution
                    if role == PlatformRole.PLATFORM_ADMIN.value and disp.status != DisputeStatus.RESOLVED.value:
                        st.markdown("##### 🏛️ Adjudicate Dispute Resolution")
                        with st.form(f"res_disp_{disp.id}"):
                            res_choice = st.selectbox(
                                "Resolution",
                                [
                                    "UPHOLD_ORIGINAL_DECISION",
                                    "OVERTURN_ORIGINAL_DECISION",
                                    "REQUEST_NEW_EVIDENCE",
                                    "ORDER_RE_REVIEW",
                                    "DISMISS_DISPUTE"
                                ]
                            )
                            res_exp = st.text_area("Resolution Explanation *")
                            if st.form_submit_button("Issue Binding Resolution", type="primary"):
                                if res_exp:
                                    try:
                                        with SessionFactory() as db:
                                            ReviewService.resolve_dispute(
                                                db=db,
                                                dispute_id=disp.id,
                                                adjudicator=curr_u,
                                                resolution=DisputeResolution(res_choice),
                                                explanation=res_exp
                                            )
                                        st.success("✅ Dispute resolution issued!")
                                        time.sleep(1)
                                        st.rerun()
                                    except Exception as ex:
                                        st.error(f"Error: {ex}")

    with tab_audit:
        st.subheader("📜 Milestone Review Cryptographic Audit Trail")
        with SessionFactory() as db:
            from backend.app.db.models.projects import Milestone
            ms_list = db.execute(select(Milestone)).scalars().all()

        if ms_list:
            sel_audit_ms_id = st.selectbox(
                "Select Milestone to Inspect Audit Chain",
                options=[m.id for m in ms_list],
                format_func=lambda x: next(f"Milestone: {m.title} [{m.status}]" for m in ms_list if m.id == x)
            )

            with SessionFactory() as db:
                audit_res = ReviewService.get_review_audit_trail(db, sel_audit_ms_id)

            if audit_res["is_chain_intact"]:
                st.success(f"🔒 **Cryptographic Hash Chain Valid**: All {audit_res['total_events']} review events cryptographically linked with unbroken SHA-256 continuity.")
            else:
                st.error("❌ **Audit Chain Integrity Failure**: Hash mismatch detected in historical records.")

            if audit_res["events"]:
                for ev in audit_res["events"]:
                    st.markdown(f"- **{ev['event_type']}** (Actor: `{ev['actor_id'] or 'SYSTEM'}`) at `{ev['created_at']}`")
                    st.caption(f"Hash: `{ev['current_hash']}`")
                    if ev.get("details"):
                        st.json(ev["details"])
            else:
                st.info("No review audit events recorded for this milestone yet.")


# ============================================================================
# PAGE: FINANCIAL OPERATIONS (Phase 8 Maker-Checker & Ledger Center)
# ============================================================================

def page_financial_operations():
    """Admin & Financial Operations Center for Maker-Checker disbursements and double-entry ledger."""
    user_id, role, full_name, email = get_current_user_info()

    st.header("⚖️ Financial Operations & Reconciliation Center")
    st.write("Milestone-based payouts, Maker-Checker authorization, double-entry ledger explorer, and automated reconciliation.")

    disb_service = DisbursementService()
    recip_service = PayoutRecipientService()

    tab_disb, tab_batches, tab_ledger, tab_rec, tab_recipients = st.tabs([
        "📋 Maker-Checker Disbursements",
        "🚀 Outbound Payout Batches",
        "⚖️ Double-Entry Ledger",
        "🔄 Automated Reconciliation",
        "🏦 NGO Payout Destinations"
    ])

    # -------------------------------------------------------------------------
    # TAB 1: MAKER-CHECKER DISBURSEMENTS
    # -------------------------------------------------------------------------
    with tab_disb:
        st.subheader("Milestone Fund Allocation & Maker-Checker Authorization")

        with SessionFactory() as db:
            from sqlalchemy import select
            from backend.app.db.models.projects import Milestone, Project
            all_disbursements = db.execute(
                select(Disbursement).order_by(Disbursement.created_at.desc())
            ).scalars().all()
            approved_milestones = db.execute(
                select(Milestone).where(Milestone.status.in_([MilestoneStatus.APPROVED, MilestoneStatus.COMPLETED]))
            ).scalars().all()

        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        total_requested = sum(d.amount_minor_units for d in all_disbursements if d.status in (DisbursementStatus.REQUESTED, DisbursementStatus.PENDING_APPROVAL))
        total_approved = sum(d.amount_minor_units for d in all_disbursements if d.status == DisbursementStatus.APPROVED)
        total_succeeded = sum(d.amount_minor_units for d in all_disbursements if d.status == DisbursementStatus.SUCCEEDED)

        col_m1.metric("Pending Approval", f"${total_requested/100:,.2f}")
        col_m2.metric("Committed / Approved", f"${total_approved/100:,.2f}")
        col_m3.metric("Settled / Succeeded", f"${total_succeeded/100:,.2f}")
        col_m4.metric("Total Disbursements", len(all_disbursements))

        st.markdown("---")

        col_maker, col_checker = st.columns([1, 1])

        # MAKER COLUMN
        with col_maker:
            st.markdown("#### 1. Maker: Request Milestone Disbursement")
            st.caption("Initiates a formal disbursement request against an approved milestone.")

            if approved_milestones:
                ms_opts = {m.id: f"{m.title} (#{m.sequence_number})" for m in approved_milestones}
                selected_ms_id = st.selectbox(
                    "Select Approved Milestone",
                    options=list(ms_opts.keys()),
                    format_func=lambda x: ms_opts[x],
                    key="maker_selected_ms"
                )

                with SessionFactory() as db:
                    eligibility = DisbursementEligibilityService.evaluate_eligibility(db, selected_ms_id)

                if eligibility["is_eligible"]:
                    st.success(f"✅ **Milestone Eligible**: ${eligibility['available_balance_minor_units']/100:,.2f} {eligibility['currency']} available for payout.")
                else:
                    st.error("❌ **Milestone Not Yet Eligible**:")
                    for r in eligibility["rejection_reasons"]:
                        st.markdown(f"- ⚠️ {r}")

                with st.form("form_request_disbursement"):
                    default_amt = eligibility["available_balance_minor_units"] / 100.0 if eligibility["available_balance_minor_units"] > 0 else 100.0
                    req_amount_dollars = st.number_input(
                        "Disbursement Amount ($ USD)",
                        min_value=1.0,
                        max_value=float(max(1.0, eligibility["available_balance_minor_units"] / 100.0)),
                        value=float(default_amt),
                        step=50.0
                    )
                    req_notes = st.text_input("Operational Notes", placeholder="Milestone deliverables approved via dual-key review.")
                    btn_request = st.form_submit_button("Submit Disbursement Request (Maker)", type="primary", use_container_width=True)

                    if btn_request:
                        req_minor = int(req_amount_dollars * 100)
                        try:
                            with SessionFactory() as db:
                                d_created = disb_service.request_disbursement(
                                    db=db,
                                    milestone_id=selected_ms_id,
                                    amount_minor_units=req_minor,
                                    requester_id=user_id or "system-operator",
                                    notes=req_notes
                                )
                            st.success(f"✅ Disbursement request {d_created.id} submitted for checker approval!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to request disbursement: {e}")
            else:
                st.info("No approved or completed milestones found in system.")

        # CHECKER COLUMN
        with col_checker:
            st.markdown("#### 2. Checker: Segregation-of-Duties Authorization")
            st.caption("Independent finance approver verifies and authorizes payout into ledger.")

            pending_disbursements = [d for d in all_disbursements if d.status in (DisbursementStatus.REQUESTED, DisbursementStatus.PENDING_APPROVAL)]

            if pending_disbursements:
                for pd in pending_disbursements:
                    with st.expander(f"Disbursement {pd.id[:8]} — ${pd.amount_minor_units/100:,.2f} {pd.currency}", expanded=True):
                        st.markdown(f"**Milestone ID:** `{pd.milestone_id}`")
                        st.markdown(f"**Gross Amount:** `${pd.amount_minor_units/100:,.2f} {pd.currency}`")
                        st.markdown(f"**Requested By:** `{pd.requested_by_id}` at `{pd.requested_at}`")

                        # Segregation of Duties Check
                        is_own_request = (user_id is not None and user_id == pd.requested_by_id)
                        if is_own_request:
                            st.warning("⚠️ **Maker-Checker Segregation**: You initiated this disbursement request. Platform policy strictly prohibits approving your own requests. An independent finance approver must review this.")
                        else:
                            col_app, col_rej = st.columns(2)
                            with col_app:
                                if st.button("Approve & Commit to Ledger", key=f"btn_app_{pd.id}", type="primary", use_container_width=True):
                                    try:
                                        with SessionFactory() as db:
                                            disb_service.approve_disbursement(
                                                db=db,
                                                disbursement_id=pd.id,
                                                approver_id=user_id or "system-approver",
                                                notes="Authorized by independent finance checker"
                                            )
                                        st.success("✅ Disbursement authorized and committed to ledger!")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Approval error: {e}")

                            with col_rej:
                                if st.button("Reject Request", key=f"btn_rej_{pd.id}", use_container_width=True):
                                    try:
                                        with SessionFactory() as db:
                                            disb_service.reject_disbursement(
                                                db=db,
                                                disbursement_id=pd.id,
                                                rejecter_id=user_id or "system-approver",
                                                reason="Rejected during checker review"
                                            )
                                        st.warning("Disbursement rejected.")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Rejection error: {e}")
            else:
                st.info("No disbursements currently pending checker authorization.")

    # -------------------------------------------------------------------------
    # TAB 2: OUTBOUND PAYOUT BATCHES
    # -------------------------------------------------------------------------
    with tab_batches:
        st.subheader("Outbound Payout Batches (PayPal Payouts REST API)")
        st.caption("Groups authorized milestone disbursements into idempotent batches for gateway execution.")

        with SessionFactory() as db:
            approved_for_batch = db.execute(
                select(Disbursement).where(Disbursement.status == DisbursementStatus.APPROVED)
            ).scalars().all()
            batches = db.execute(select(PayoutBatch).order_by(PayoutBatch.created_at.desc())).scalars().all()

        if approved_for_batch:
            st.markdown(f"**Ready for Submission:** {len(approved_for_batch)} approved disbursement(s)")
            selected_ids = []
            for d in approved_for_batch:
                checked = st.checkbox(
                    f"Disbursement {d.id[:8]} — ${d.amount_minor_units/100:,.2f} {d.currency} (Milestone: `{d.milestone_id[:8]}`)",
                    value=True,
                    key=f"batch_chk_{d.id}"
                )
                if checked:
                    selected_ids.append(d.id)

            col_sub1, col_sub2 = st.columns([1, 2])
            with col_sub1:
                batch_subject = st.text_input("PayPal Email Subject", value="CharityChain Verified Milestone Disbursement")
                if st.button("🚀 Submit Payout Batch to PayPal", type="primary", use_container_width=True):
                    if selected_ids:
                        try:
                            with SessionFactory() as db:
                                b_res = disb_service.create_and_submit_payout_batch(
                                    db=db,
                                    disbursement_ids=selected_ids,
                                    submitter_id=user_id or "system-approver",
                                    email_subject=batch_subject
                                )
                            st.success(f"✅ Batch submitted! Reference ID: `{b_res.batch_reference_id}`, Status: `{b_res.status.value}`")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Batch submission failed: {e}")
                    else:
                        st.warning("Please select at least one disbursement.")
        else:
            st.info("No approved disbursements awaiting batch submission.")

        st.markdown("---")
        st.markdown("#### Historical Payout Batches")
        if batches:
            batch_data = []
            for b in batches:
                batch_data.append({
                    "Batch Reference": b.batch_reference_id,
                    "Provider Batch ID": b.provider_batch_id or "Pending",
                    "Status": b.status.value,
                    "Items": b.total_items_count,
                    "Total Amount": f"${b.total_amount_minor_units/100:,.2f} {b.currency}",
                    "Fees": f"${b.total_fees_minor_units/100:,.2f}",
                    "Submitted At": str(b.submitted_at)[:19] if b.submitted_at else ""
                })
            st.dataframe(batch_data, use_container_width=True)
        else:
            st.caption("No payout batches executed yet.")

    # -------------------------------------------------------------------------
    # TAB 3: DOUBLE-ENTRY LEDGER
    # -------------------------------------------------------------------------
    with tab_ledger:
        st.subheader("Double-Entry General Ledger & Trial Balance")
        st.caption("Formal double-entry accounting records where Sum(Debits) == Sum(Credits).")

        with SessionFactory() as db:
            tb = LedgerService.get_trial_balance(db)

        col_tb_status, col_tb_deb, col_tb_cred = st.columns(3)
        if tb["is_balanced"]:
            col_tb_status.success("✅ **Ledger In Balance** (Debits == Credits)")
        else:
            col_tb_status.error(f"❌ **Trial Balance Discrepancy**: ${tb['discrepancy_minor_units']/100:,.2f}")

        col_tb_deb.metric("Total Debits", f"${tb['total_debit_minor_units']/100:,.2f}")
        col_tb_cred.metric("Total Credits", f"${tb['total_credit_minor_units']/100:,.2f}")

        st.markdown("#### Chart of Accounts & Balances")
        accounts_table = []
        for a in tb["accounts"]:
            accounts_table.append({
                "Account #": a["account_number"],
                "Account Name": a["account_name"],
                "Type": a["account_type"],
                "Total Debits": f"${a['total_debit_minor_units']/100:,.2f}",
                "Total Credits": f"${a['total_credit_minor_units']/100:,.2f}",
                "Net Balance": f"${a['net_balance_minor_units']/100:,.2f} {a['currency']}"
            })
        st.dataframe(accounts_table, use_container_width=True)

        st.markdown("---")
        st.markdown("#### Account Statement & Audit Explorer")
        acc_choice = st.selectbox(
            "Select Account to Drill Down",
            options=[a["account_number"] for a in tb["accounts"]],
            format_func=lambda x: next(f"{a['account_number']} - {a['account_name']}" for a in tb["accounts"] if a["account_number"] == x)
        )

        with SessionFactory() as db:
            stmt = LedgerService.get_account_statement(db, acc_choice)

        st.markdown(f"**Current Running Balance:** `${stmt['current_balance_minor_units']/100:,.2f} {stmt['currency']}` ({stmt['entries_count']} entries)")
        if stmt["entries"]:
            st.dataframe(stmt["entries"], use_container_width=True)
        else:
            st.caption("No journal entries posted to this account yet.")

    # -------------------------------------------------------------------------
    # TAB 4: AUTOMATED RECONCILIATION
    # -------------------------------------------------------------------------
    with tab_rec:
        st.subheader("Automated Financial Reconciliation Engine")
        st.caption("Reconciles local captures, refunds, and payouts against gateway settlement files.")

        col_d1, col_d2, col_btn = st.columns([1, 1, 1])
        with col_d1:
            p_start = st.date_input("Period Start", value=date.today() - timedelta(days=30))
        with col_d2:
            p_end = st.date_input("Period End", value=date.today() + timedelta(days=1))
        with col_btn:
            st.write("")
            st.write("")
            btn_run_rec = st.button("🔄 Run Automated Reconciliation", type="primary", use_container_width=True)

        if btn_run_rec:
            dt_start = datetime.combine(p_start, datetime.min.time())
            dt_end = datetime.combine(p_end, datetime.max.time())
            try:
                with SessionFactory() as db:
                    rec_run = ReconciliationService.run_reconciliation(
                        db=db,
                        period_start=dt_start,
                        period_end=dt_end,
                        executed_by_id=user_id or "system-auditor"
                    )
                    run_report = ReconciliationService.get_run_details(db, rec_run.id)

                st.session_state["last_rec_run"] = run_report
                st.success(f"✅ Reconciliation run {rec_run.run_number} complete! Status: {rec_run.status.value}")
            except Exception as e:
                st.error(f"Reconciliation failed: {e}")

        # Display Run Findings
        last_run = st.session_state.get("last_rec_run")
        if last_run:
            st.markdown(f"### Run Summary: `{last_run['run_number']}` ({last_run['status']})")
            c_r1, c_r2, c_r3, c_r4 = st.columns(4)
            c_r1.metric("Analyzed", last_run["total_records_analyzed"])
            c_r2.metric("Matched", last_run["matched_count"])
            c_r3.metric("Discrepancies", last_run["discrepancy_count"])
            c_r4.metric("Local Total", f"${last_run['total_local_minor_units']/100:,.2f}")

            if last_run["items"]:
                st.markdown("#### Itemized Audit Records")
                st.dataframe(last_run["items"], use_container_width=True)

                discrepancies = [it for it in last_run["items"] if it["reconciliation_status"] == "DISCREPANCY"]
                if discrepancies:
                    st.markdown("#### Resolve Identified Discrepancy")
                    target_item_id = st.selectbox(
                        "Select Discrepant Item",
                        options=[d["item_id"] for d in discrepancies],
                        format_func=lambda x: next(f"{d['item_type']} - Ref: {d['provider_transaction_id'] or d['reference_id']} ({d['discrepancy_type']})" for d in discrepancies if d["item_id"] == x)
                    )
                    with st.form("form_resolve_disc"):
                        res_type = st.selectbox("Resolution Action", options=[r.value for r in DiscrepancyResolution])
                        res_notes = st.text_area("Resolution Explanation", placeholder="Document why discrepancy was adjusted or accepted.")
                        btn_resolve = st.form_submit_button("Record Discrepancy Resolution", type="primary")

                        if btn_resolve:
                            try:
                                with SessionFactory() as db:
                                    ReconciliationService.resolve_discrepancy(
                                        db=db,
                                        item_id=target_item_id,
                                        resolution=DiscrepancyResolution(res_type),
                                        notes=res_notes,
                                        resolver_id=user_id or "system-auditor"
                                    )
                                st.success("✅ Discrepancy resolution saved.")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Failed to resolve discrepancy: {e}")
            else:
                st.info("No transaction records found within selected date range.")

    # -------------------------------------------------------------------------
    # TAB 5: NGO PAYOUT DESTINATIONS
    # -------------------------------------------------------------------------
    with tab_recipients:
        st.subheader("NGO Payout Destinations & Verification Management")
        st.caption("Configured receiving accounts for verified charitable organizations.")

        with SessionFactory() as db:
            from backend.app.db.models.users import Organization
            all_orgs = db.execute(select(Organization)).scalars().all()
            all_recipients = db.execute(select(PayoutRecipient).order_by(PayoutRecipient.created_at.desc())).scalars().all()

        col_rec_list, col_rec_add = st.columns([1, 1])

        with col_rec_list:
            st.markdown("#### Active Payout Destinations")
            if all_recipients:
                for r in all_recipients:
                    with st.expander(f"{r.destination_masked} ({r.status.value})", expanded=(r.status == PayoutRecipientStatus.PENDING_VERIFICATION)):
                        st.markdown(f"**Org ID:** `{r.organization_id}`")
                        st.markdown(f"**Method:** `{r.payout_method.value}` ({r.recipient_type.value})")
                        st.markdown(f"**Primary:** `{'Yes' if r.is_primary else 'No'}`")
                        st.markdown(f"**Status:** `{r.status.value}`")

                        if r.status == PayoutRecipientStatus.PENDING_VERIFICATION:
                            if st.button("Verify Recipient Account", key=f"btn_ver_rec_{r.id}", type="primary"):
                                try:
                                    with SessionFactory() as db:
                                        recip_service.verify_recipient(
                                            db=db,
                                            recipient_id=r.id,
                                            verifier_id=user_id or "system-admin",
                                            verification_notes="Verified via Financial Operations Center"
                                        )
                                    st.success("✅ Payout recipient verified successfully!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Verification error: {e}")
            else:
                st.info("No payout destinations registered yet.")

        with col_rec_add:
            st.markdown("#### Register Payout Destination for NGO")
            if all_orgs:
                org_map = {o.id: f"{o.legal_name} ({o.status})" for o in all_orgs}
                selected_org = st.selectbox("Select Organization", options=list(org_map.keys()), format_func=lambda x: org_map[x])

                with st.form("form_reg_payout_recip"):
                    dest_type = st.selectbox("Destination Type", options=[RecipientType.PAYPAL_EMAIL.value, RecipientType.BANK_ACCOUNT.value])
                    dest_value = st.text_input("Destination (e.g. PayPal Email Address)", placeholder="finance@ngo.org")
                    is_prim = st.checkbox("Set as Primary Receiving Destination", value=True)
                    btn_reg_rec = st.form_submit_button("Register Destination", type="primary", use_container_width=True)

                    if btn_reg_rec:
                        try:
                            with SessionFactory() as db:
                                recip_service.register_recipient(
                                    db=db,
                                    organization_id=selected_org,
                                    destination=dest_value,
                                    recipient_type=RecipientType(dest_type),
                                    is_primary=is_prim,
                                    actor_id=user_id or "system-operator"
                                )
                            st.success("✅ Payout destination registered and queued for verification!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Registration failed: {e}")
            else:
                st.info("No organizations registered in system.")


# ============================================================================
# PAGE: DISCOVER PROJECTS (Enhanced Public Discovery)
# ============================================================================

def page_discover_projects():
    """Public discovery for active, verified charitable projects."""
    st.header("🌍 Discover Verified Charitable Projects")
    st.write("Browse projects that have undergone rigorous organization verification and multi-milestone compliance checks.")

    # Fetch public categories for filter
    with SessionFactory() as db:
        cats = ProjectService.list_categories(db)
    
    col_search, col_cat = st.columns([2, 1])
    with col_search:
        search_query = st.text_input("🔍 Search Projects", placeholder="Search by title, location, or keyword...")
    with col_cat:
        cat_filter = st.selectbox(
            "Filter by Category",
            options=["All Categories"] + [c.name for c in cats]
        )

    cat_id = None
    if cat_filter != "All Categories":
        cat_match = next((c for c in cats if c.name == cat_filter), None)
        if cat_match:
            cat_id = cat_match.id

    with SessionFactory() as db:
        res = ProjectService.list_public_projects(
            db=db,
            search=search_query or None,
            category_id=cat_id,
            page=1,
            page_size=20
        )

    # Fallback to existing TinyDB milestones if no database projects are active yet
    db_legacy: DatabaseManager = st.session_state.db
    legacy_milestones = db_legacy.get_all_milestones()

    if not res.items and not legacy_milestones:
        st.info("No active projects found matching your criteria. Try adjusting your search.")
        return

    st.markdown(f"**Found {res.total_count} verified active project(s)**")
    st.markdown("---")

    for proj in res.items:
        with st.expander(f"🌟 {proj.title} — ${proj.funding_target_minor/100:,.2f} {proj.currency} [{proj.country}]", expanded=True):
            col_left, col_right = st.columns([2, 1])
            with col_left:
                st.markdown(f"**NGO:** {proj.organization_name} ⭐ *(Verified)*")
                st.markdown(f"**Category:** {proj.category_name} | **Region:** {proj.primary_operating_region}")
                st.markdown(f"**Summary:** {proj.summary}")
                
                pct = min(100.0, (proj.funding_raised_minor / proj.funding_target_minor * 100.0)) if proj.funding_target_minor > 0 else 0.0
                st.progress(pct / 100.0)
                st.caption(f"{pct:.1f}% funded (${proj.funding_raised_minor/100:,.2f} raised of ${proj.funding_target_minor/100:,.2f} {proj.currency})")

            with col_right:
                st.metric("Goal", f"${proj.funding_target_minor/100:,.2f}")
                st.metric("Raised", f"${proj.funding_raised_minor/100:,.2f}")

            # Fetch project milestones
            with SessionFactory() as db:
                ms = ProjectService.list_project_milestones(db, proj.id)
            
            if ms:
                st.markdown("##### 🎯 Project Milestones")
                for m in ms:
                    st.markdown(f"- **#{m.sequence_number}: {m.title}** (${m.funding_target_minor/100:,.2f}) — `{m.status}`")
                    st.caption(f"Objective: {m.objective}")

            # Genuine Phase 4 PayPal Contribution Experience
            st.markdown("##### 💝 Support this Project via PayPal")
            if not ms:
                st.info("Milestones for this project are being finalized. Contributions will open shortly.")
            else:
                user_id, role, full_name, email = get_current_user_info()
                
                # Check if there is an active pending order in session for this project
                active_order_key = f"active_order_{proj.id}"
                active_order = st.session_state.get(active_order_key)

                if active_order:
                    st.info(f"⏳ **Active PayPal Order Created**: `{active_order.get('provider_order_id')}`")
                    st.markdown(f"**Amount**: ${active_order.get('amount_minor_units', 0)/100:,.2f} {active_order.get('currency', 'USD')}")
                    
                    col_btn1, col_btn2, col_btn3 = st.columns([2, 2, 1])
                    with col_btn1:
                        if active_order.get("approval_url"):
                            st.link_button("💳 Complete Payment on PayPal", active_order["approval_url"], use_container_width=True)
                    with col_btn2:
                        if st.button("✅ Confirm & Capture Payment", type="primary", key=f"cap_{proj.id}", use_container_width=True):
                            with st.spinner("Verifying transaction with PayPal..."):
                                try:
                                    srv = DonationService()
                                    with SessionFactory() as db:
                                        cap_res = asyncio.run(
                                            srv.capture_donation_payment(
                                                db=db,
                                                order_identifier=active_order["provider_order_id"],
                                                user_id=user_id
                                            )
                                        )
                                    st.success(f"🎉 Payment Verified & Captured! Capture ID: `{cap_res['provider_capture_id']}`")
                                    st.balloons()
                                    del st.session_state[active_order_key]
                                    time.sleep(1.5)
                                    st.rerun()
                                except Exception as err:
                                    st.error(f"Capture failed: {err}")
                    with col_btn3:
                        if st.button("❌ Cancel", key=f"cnc_{proj.id}", use_container_width=True):
                            del st.session_state[active_order_key]
                            st.rerun()

                else:
                    with st.form(f"fund_proj_{proj.id}"):
                        c_ms, c_amt = st.columns([2, 1])
                        with c_ms:
                            sel_m_id = st.selectbox(
                                "Select Milestone to Fund",
                                options=[m.id for m in ms],
                                format_func=lambda x: next(f"#{m.sequence_number}: {m.title} (${m.funding_target_minor/100:,.2f})" for m in ms if m.id == x),
                                key=f"ms_sel_{proj.id}"
                            )
                        with c_amt:
                            d_amount = st.number_input(
                                f"Amount ({proj.currency})",
                                min_value=1.0,
                                value=50.0,
                                step=10.0,
                                key=f"damt_{proj.id}"
                            )

                        c_anon, c_info = st.columns([1, 2])
                        with c_anon:
                            is_anon = st.checkbox("Donate Anonymously", value=False, key=f"anon_{proj.id}")
                        with c_info:
                            st.caption("🔒 Payments are securely processed via PayPal REST API. Funds are tied to verifiable milestone evidence.")

                        if st.form_submit_button("💳 Proceed to PayPal Checkout", type="primary", use_container_width=True):
                            amount_minor = int(d_amount * 100)
                            with st.spinner("Generating secure PayPal checkout order..."):
                                try:
                                    srv = DonationService()
                                    with SessionFactory() as db:
                                        res_order = asyncio.run(
                                            srv.create_donation_intent(
                                                db=db,
                                                project_id=proj.id,
                                                milestone_id=sel_m_id,
                                                amount_minor_units=amount_minor,
                                                currency=proj.currency,
                                                donor_id=user_id,
                                                donor_name=full_name if not is_anon else "Anonymous Donor",
                                                donor_email=email,
                                                is_anonymous=is_anon
                                            )
                                        )
                                    st.session_state[active_order_key] = res_order
                                    st.success("✅ PayPal order created! Click the payment button below to complete checkout.")
                                    st.rerun()
                                except Exception as exc:
                                    st.error(f"Checkout initialization failed: {exc}")

    # If legacy items exist, show them under a dedicated section
    if legacy_milestones:
        st.markdown("---")
        st.subheader("📦 Legacy Demonstrative Milestones")
        for m in legacy_milestones:
            m_id = m.get('id', 0)
            desc = m.get('description', '')
            with st.expander(f"🎯 Milestone #{m_id}: {desc[:60]}..."):
                st.markdown(f"**NGO:** {m.get('owner', 'Unknown')} | **Status:** `{m.get('status', 'PENDING')}`")
                g = m.get('goal', 1)
                r = m.get('raised', 0)
                st.caption(f"Goal: {g/10**18:.2f} ETH | Raised: {r/10**18:.4f} ETH")


# Legacy alias functions to maintain 100% backward compatibility
def page_create_project():
    """Legacy alias redirecting to Phase 3 project management."""
    page_project_management()

def page_donate():
    """Legacy alias redirecting to Phase 3 public discovery."""
    page_discover_projects()


# ============================================================================
# PAGE: DONOR DASHBOARD (Phase 4 Payment History & Refunds)
# ============================================================================

def page_donor_dashboard():
    """Donor financial portfolio, transaction history, receipts, and refund requests."""
    st.header("💝 Donor Dashboard & Contribution History")
    st.write("Track all your evidence-backed contributions, inspect verified payment receipts, and manage eligible refunds.")

    user_id, role, full_name, email = get_current_user_info()
    if not user_id or user_id == "demo_user":
        st.warning("🔒 Please sign in to view your personal donation history and manage refunds.")
        return

    srv = DonationService()
    with SessionFactory() as db:
        donations, total_count = srv.list_donor_donations(db, donor_id=user_id, page=1, page_size=100)

    # Calculate portfolio summary
    paid_donations = [d for d in donations if d.status in (DonationStatus.PAID, DonationStatus.PARTIALLY_REFUNDED, DonationStatus.REFUNDED)]
    gross_minor = sum(d.amount_minor_units for d in paid_donations)
    
    with SessionFactory() as db:
        from backend.app.db.models.financial import PaymentRefund, PaymentCapture
        from sqlalchemy import select, func
        donation_ids = [d.id for d in donations]
        refunded_minor = 0
        if donation_ids:
            refunded_minor = db.execute(
                select(func.coalesce(func.sum(PaymentRefund.amount_minor_units), 0))
                .select_from(PaymentRefund)
                .join(PaymentCapture, PaymentRefund.capture_id == PaymentCapture.id)
                .join(Donation, PaymentCapture.order_id == Donation.id)
                .where(Donation.donor_id == user_id)
            ).scalar() or 0

    net_minor = max(0, gross_minor - refunded_minor)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Contributed", f"${gross_minor/100:,.2f}")
    with col2:
        st.metric("Successful Contributions", len(paid_donations))
    with col3:
        st.metric("Total Refunded", f"${refunded_minor/100:,.2f}")
    with col4:
        st.metric("Net Impact Funding", f"${net_minor/100:,.2f}")

    st.markdown("---")
    tab_records, tab_journey, tab_annual = st.tabs([
        "📜 Contributions & Receipts",
        "🌟 Personal Impact Journey",
        "📄 Annual Giving Report"
    ])

    with tab_records:
        st.subheader(f"📜 Contribution Records ({total_count})")

        if not donations:
            st.info("You haven't made any contributions yet. Explore verified projects to get started!")
        else:
            for don in donations:
                with SessionFactory() as db:
                    from backend.app.db.models.projects import Project, Milestone
                    from backend.app.db.models.financial import PaymentOrder, PaymentCapture, PaymentRefund
                    proj = db.execute(select(Project).where(Project.id == don.project_id)).scalar_one_or_none()
                    ms = db.execute(select(Milestone).where(Milestone.id == don.milestone_id)).scalar_one_or_none()
                    p_order = db.execute(select(PaymentOrder).where(PaymentOrder.donation_id == don.id)).scalar_one_or_none()
                    p_capture = db.execute(select(PaymentCapture).where(PaymentCapture.order_id == p_order.id)).scalar_one_or_none() if p_order else None
                    refunds = db.execute(select(PaymentRefund).where(PaymentRefund.capture_id == p_capture.id)).scalars().all() if p_capture else []

                badge_color = "green" if don.status == DonationStatus.PAID else ("orange" if don.status == DonationStatus.PAYMENT_PENDING else ("blue" if "REFUND" in don.status.value else "red"))
                card_title = f"🏷️ {proj.title if proj else 'Unknown Project'} — ${don.amount_minor_units/100:,.2f} {don.currency} [{don.status.value}]"

                with st.expander(card_title, expanded=(don.status == DonationStatus.PAID)):
                    c_left, c_right = st.columns([2, 1])
                    with c_left:
                        st.markdown(f"**Target Milestone:** #{ms.sequence_number if ms else '?'}: {ms.title if ms else 'General'}")
                        st.markdown(f"**Donation Reference:** `{don.id}`")
                        if p_order and p_order.provider_order_id:
                            st.markdown(f"**PayPal Order ID:** `{p_order.provider_order_id}`")
                        if p_capture and p_capture.provider_capture_id:
                            st.markdown(f"**Provider Capture ID:** `{p_capture.provider_capture_id}`")
                        st.caption(f"Created: {don.created_at.strftime('%Y-%m-%d %H:%M UTC') if don.created_at else 'N/A'}")

                    with c_right:
                        st.metric("Amount", f"${don.amount_minor_units/100:,.2f} {don.currency}")
                        st.markdown(f"**Status:** `{don.status.value}`")

                    # Show existing refunds
                    if refunds:
                        st.markdown("##### 💸 Processed Refunds")
                        for r in refunds:
                            st.markdown(f"- **${r.amount_minor_units/100:,.2f} {r.currency}** — *{r.reason}* (`{r.provider_refund_id}`)")

                    # Refund Request Form for PAID or PARTIALLY_REFUNDED donations
                    if don.status in (DonationStatus.PAID, DonationStatus.PARTIALLY_REFUNDED) and p_capture:
                        already_ref = sum(r.amount_minor_units for r in refunds if r.status == PaymentRefundStatus.COMPLETED)
                        remaining_ref = p_capture.amount_minor_units - already_ref

                        if remaining_ref > 0:
                            with st.expander("🔄 Request Full or Partial Refund"):
                                with st.form(f"refund_form_{don.id}"):
                                    max_val = float(remaining_ref / 100.0)
                                    ref_amt = st.number_input(
                                        f"Refund Amount ({don.currency}) - Max: ${max_val:,.2f}",
                                        min_value=1.0,
                                        max_value=max_val,
                                        value=max_val,
                                        step=5.0,
                                        key=f"ref_amt_{don.id}"
                                    )
                                    ref_reason = st.text_input(
                                        "Reason for Refund *",
                                        value="Donor requested refund",
                                        key=f"ref_rsn_{don.id}"
                                    )
                                    btn_ref = st.form_submit_button("Submit Refund Request", type="primary")

                                    if btn_ref:
                                        if not ref_reason:
                                            st.error("Please provide a reason for the refund.")
                                        else:
                                            with st.spinner("Processing refund with PayPal..."):
                                                try:
                                                    with SessionFactory() as db_ref:
                                                        ref_res = asyncio.run(
                                                            srv.process_refund(
                                                                db=db_ref,
                                                                donation_id=don.id,
                                                                amount_minor_units=int(ref_amt * 100),
                                                                reason=ref_reason,
                                                                actor_id=user_id,
                                                                is_admin=False
                                                            )
                                                        )
                                                    st.success(f"✅ Refund of ${ref_amt:,.2f} processed successfully! Refund ID: `{ref_res['provider_refund_id']}`")
                                                    time.sleep(1.5)
                                                    st.rerun()
                                                except Exception as exc:
                                                    st.error(f"Refund failed: {exc}")

    with tab_journey:
        st.subheader("🌟 Contribution-to-Impact Journey")
        st.write("Track how your contributions directly support verified real-world milestones and yield cryptographic proof-of-impact certificates.")
        try:
            with SessionFactory() as db_session:
                journeys = DonorAnalyticsService.get_personal_impact_journey(db_session, user_id)
            if not journeys:
                st.info("No verified impact journeys recorded yet.")
            else:
                for j in journeys:
                    with st.container():
                        st.markdown(f"#### 🎗️ {j['project']['title']} — {j['amount_formatted']}")
                        st.write(f"**Target Milestone:** #{j['milestone']['sequence_number']}: {j['milestone']['title']} ({j['milestone']['status']})")
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.caption(f"Donated: {j['donated_at'][:10] if j['donated_at'] else 'N/A'}")
                        with c2:
                            st.caption(f"Evidence: {'✅ Submitted (' + str(j['evidence_count']) + ' files)' if j['evidence_submitted'] else '⏳ Awaiting evidence'}")
                        with c3:
                            if j['certificate']['has_certificate']:
                                st.success(f"🎓 Certificate #{j['certificate']['certificate_code']}")
                            else:
                                st.caption("Status: In progress")
                        st.markdown("---")
        except Exception as e:
            st.error(f"Error loading impact journey: {e}")

    with tab_annual:
        st.subheader("📄 Annual Charitable Contribution Statement")
        col_y, col_dl = st.columns([1, 2])
        with col_y:
            selected_year = st.selectbox("Tax Year", [2026, 2025, 2024], index=0)
        
        try:
            with SessionFactory() as db_session:
                annual_rep = DonorAnalyticsService.generate_annual_giving_report(db_session, user_id, selected_year)
                donor_csv = AnalyticsExportService.export_donor_contributions_csv(db_session, user_id)
            
            st.metric("Total Annual Giving", annual_rep["total_contributions_formatted"])
            st.caption(f"Total Contributions in {selected_year}: {annual_rep['donation_count']}")
            
            with col_dl:
                st.download_button(
                    label="📥 Download CSV Contribution Statement",
                    data=donor_csv,
                    file_name=f"charitychain_annual_giving_{user_id[:8]}_{selected_year}.csv",
                    mime="text/csv"
                )
            
            st.info(f"⚖️ **Disclaimer:** {annual_rep['tax_disclaimer']}")
            if annual_rep["contributions"]:
                st.table(annual_rep["contributions"])
            else:
                st.caption(f"No contributions recorded for calendar year {selected_year}.")
        except Exception as e:
            st.error(f"Error generating annual statement: {e}")



# ============================================================================
# PAGE: SUBMIT EVIDENCE & EVIDENCE MANAGEMENT (Phase 5 Engine + Legacy Demo)
# ============================================================================

def page_submit_evidence():
    """NGO evidence submission with automated multi-indicator verification and Phase 5 Evidence Management."""
    st.header("📸 Evidence Management & Milestone Verification")

    tab_engine, tab_legacy = st.tabs([
        "🔒 Phase 5 Evidence Engine (PostgreSQL / S3)",
        "🧪 Legacy Demo Milestones (TinyDB)"
    ])

    with tab_engine:
        user_id, role, full_name, email = get_current_user_info()
        if not user_id:
            st.warning("🔒 Please sign in via the sidebar to manage evidence for your organization's projects.")
        else:
            with SessionFactory() as db:
                user_orgs = NGOOnboardingService.list_user_organizations(db, user_id)

            if not user_orgs and role != "ADMIN":
                st.warning("⚠️ You must belong to a registered NGO organization to manage evidence.")
            else:
                org_options = {o.id: o.name for o in user_orgs}
                if not org_options and role == "ADMIN":
                    with SessionFactory() as db:
                        from backend.app.db.models.users import Organization
                        from sqlalchemy import select
                        all_orgs = db.execute(select(Organization)).scalars().all()
                        org_options = {o.id: o.name for o in all_orgs}

                if not org_options:
                    st.info("No organizations available. Please register an NGO organization first.")
                else:
                    sel_org_id = st.selectbox("Select Organization", options=list(org_options.keys()), format_func=lambda x: org_options[x], key="ev_org_sel")

                    with SessionFactory() as db:
                        org_projects = ProjectService.list_organization_projects(db, sel_org_id, user_id)

                    if not org_projects:
                        st.info("No projects found for this organization.")
                    else:
                        proj_options = {p.id: f"{p.title} [{p.status}]" for p in org_projects}
                        sel_proj_id = st.selectbox("Select Project", options=list(proj_options.keys()), format_func=lambda x: proj_options[x], key="ev_proj_sel")

                        with SessionFactory() as db:
                            milestones = ProjectService.list_project_milestones(db, sel_proj_id, user_id)

                        if not milestones:
                            st.info("No milestones found for this project.")
                        else:
                            ms_options = {m.id: f"Milestone #{m.sequence_number}: {m.title} [{m.status}]" for m in milestones}
                            sel_ms_id = st.selectbox("Select Milestone", options=list(ms_options.keys()), format_func=lambda x: ms_options[x], key="ev_ms_sel")
                            selected_ms = next(m for m in milestones if m.id == sel_ms_id)

                            st.markdown(f"**Objective:** {selected_ms.objective}")
                            if selected_ms.deliverables:
                                st.caption(f"**Deliverables:** {selected_ms.deliverables}")

                            st.markdown("---")
                            st.subheader("📤 Multi-File Evidence Upload")
                            st.caption(f"Allowed types: JPEG, PNG, PDF • Max size: {settings.MAX_UPLOAD_SIZE_MB} MB • Batch limit: {settings.MAX_UPLOAD_BATCH_SIZE} files")

                            uploaded_files = st.file_uploader(
                                "Upload Evidence Files",
                                accept_multiple_files=True,
                                type=['jpg', 'jpeg', 'png', 'pdf'],
                                key="p5_ev_uploader"
                            )

                            if uploaded_files:
                                st.write(f"Selected **{len(uploaded_files)}** file(s):")
                                for uf in uploaded_files:
                                    st.caption(f"📄 **{uf.name}** ({len(uf.getvalue()) / (1024*1024):.2f} MB, {uf.type})")

                                if st.button("🚀 Securely Upload & Record to Hash Chain", type="primary", use_container_width=True):
                                    file_tuples = []
                                    for uf in uploaded_files:
                                        file_tuples.append((uf.getvalue(), uf.name, uf.type))

                                    try:
                                        with SessionFactory() as db:
                                            res = EvidenceService.upload_evidence_files(
                                                db=db,
                                                milestone_id=sel_ms_id,
                                                files=file_tuples,
                                                uploader_id=user_id
                                            )
                                        st.success(f"✅ Successfully uploaded {res.total_uploaded} file(s) into immutable hash chain!")
                                        time.sleep(1)
                                        st.rerun()
                                    except Exception as ex:
                                        st.error(f"❌ Upload failed: {ex}")

                            st.markdown("---")
                            st.subheader("📂 Milestone Evidence Records")

                            with SessionFactory() as db:
                                evidence_records = EvidenceService.list_milestone_evidence(db, sel_ms_id, include_archived=True)

                            if not evidence_records:
                                st.info("No evidence submitted yet for this milestone.")
                            else:
                                for ev in evidence_records:
                                    arch_badge = " 📦 [ARCHIVED]" if ev.is_archived else ""
                                    with st.expander(f"📄 {ev.original_filename or ev.id} (v{ev.version_number}) — {ev.file_size_bytes/1024:.1f} KB [{ev.status}]{arch_badge}"):
                                        c1, c2 = st.columns(2)
                                        with c1:
                                            st.write(f"**SHA-256 Hash:** `{ev.file_hash}`")
                                            st.write(f"**Storage Provider:** `{ev.storage_provider}`")
                                            st.write(f"**MIME Type:** `{ev.mime_type}`")
                                        with c2:
                                            st.write(f"**Uploaded At:** {ev.created_at}")
                                            st.write(f"**Storage Key:** `{ev.storage_key}`")
                                            st.write(f"**Version:** {ev.version_number}")

                                        # Action buttons
                                        b_col1, b_col2, b_col3 = st.columns(3)
                                        with b_col1:
                                            if st.button("🔍 Verify SHA-256 Integrity", key=f"verify_{ev.id}"):
                                                with SessionFactory() as db:
                                                    chk = EvidenceService.verify_evidence_integrity(db, ev.id, user_id)
                                                if chk.is_valid:
                                                    st.success(f"✅ Cryptographic Match! Recorded: `{chk.expected_hash[:16]}...` Computed: `{chk.computed_hash[:16]}...`")
                                                else:
                                                    st.error("❌ INTEGRITY VIOLATION DETECTED!")
                                        with b_col2:
                                            if not ev.is_archived:
                                                if st.button("📦 Soft-Archive", key=f"arch_{ev.id}"):
                                                    with SessionFactory() as db:
                                                        EvidenceService.archive_evidence(db, ev.id, user_id, reason="Archived via UI")
                                                    st.warning("Asset marked as archived.")
                                                    st.rerun()
                                        with b_col3:
                                            if not ev.is_archived:
                                                if st.button("🤖 Run AI Verification", key=f"btn_run_ai_{ev.id}", type="primary"):
                                                    try:
                                                        with SessionFactory() as db:
                                                            from backend.app.db.models.users import User
                                                            from sqlalchemy import select
                                                            u_obj = db.execute(select(User).where(User.id == user_id)).scalar_one_or_none()
                                                            VerificationService.submit_verification_job(db, ev.id, u_obj)
                                                        st.success("✅ AI verification analysis executed!")
                                                        time.sleep(1)
                                                        st.rerun()
                                                    except Exception as e_ai:
                                                        st.error(f"AI Verification failed: {e_ai}")

                                        # AI Verification Report Expander
                                        with st.expander("🤖 Phase 6 AI Verification Findings & Forensic Indicators", expanded=True):
                                            with SessionFactory() as db:
                                                from backend.app.db.models.users import User
                                                from sqlalchemy import select
                                                u_obj = db.execute(select(User).where(User.id == user_id)).scalar_one_or_none()
                                                try:
                                                    ai_report = VerificationService.get_latest_report(db, ev.id, u_obj)
                                                except Exception:
                                                    ai_report = None

                                            if not ai_report:
                                                st.info("No AI verification report generated yet for this evidence asset.")
                                                # Optional baseline linking
                                                other_evs = [other for other in evidence_records if other.id != ev.id and not other.is_archived]
                                                baseline_sel = None
                                                if other_evs:
                                                    b_opts = {None: "None (Single evidence analysis)"}
                                                    for o in other_evs:
                                                        b_opts[o.id] = f"{o.original_filename or o.id} (v{o.version_number})"
                                                    baseline_sel = st.selectbox(
                                                        "Link Baseline Evidence (Before / After Comparison)",
                                                        options=list(b_opts.keys()),
                                                        format_func=lambda x: b_opts[x],
                                                        key=f"base_sel_{ev.id}"
                                                    )

                                                if st.button("🚀 Run Analysis Now", key=f"run_now_{ev.id}"):
                                                    try:
                                                        with SessionFactory() as db:
                                                            u_obj = db.execute(select(User).where(User.id == user_id)).scalar_one_or_none()
                                                            from backend.app.api.schemas import StartVerificationRequest
                                                            s_req = StartVerificationRequest(baseline_evidence_id=baseline_sel) if baseline_sel else None
                                                            VerificationService.submit_verification_job(db, ev.id, u_obj, req=s_req)
                                                        st.success("✅ Analysis completed!")
                                                        time.sleep(1)
                                                        st.rerun()
                                                    except Exception as e_start:
                                                        st.error(f"Analysis trigger failed: {e_start}")
                                            else:
                                                # Render rich verification report
                                                oa = ai_report.overall_assessment
                                                if oa == "CONSISTENT":
                                                    st.success("### ✅ Overall Assessment: CONSISTENT WITH DECLARATIONS")
                                                elif oa == "NEEDS_FURTHER_REVIEW":
                                                    st.warning("### ⚠️ Overall Assessment: NEEDS FURTHER REVIEW (Anomalies Flagged)")
                                                elif oa == "INCONSISTENT":
                                                    st.error("### ❌ Overall Assessment: INCONSISTENT / CONFLICT DETECTED")
                                                else:
                                                    st.info(f"### ℹ️ Overall Assessment: {oa}")

                                                conf_pct = (ai_report.confidence_score * 100) if ai_report.confidence_score is not None else 0.0
                                                st.caption(f"**Confidence Indicator:** {conf_pct:.1f}% • **Calibration Status:** `{ai_report.calibration_status}` • **Pipeline:** `{ai_report.pipeline_version}` • **Report Date:** {ai_report.created_at.strftime('%Y-%m-%d %H:%M:%S')}")

                                                if ai_report.summary_notes:
                                                    st.info(f"**Reviewer Summary:**\n\n{ai_report.summary_notes}")

                                                st.markdown("#### 🔬 Technical Forensic Checks")
                                                for chk in ai_report.checks:
                                                    badge = "✅" if chk.passed else ("⚠️" if chk.status == "FLAGGED" else "ℹ️")
                                                    with st.expander(f"{badge} {chk.check_type} — [{chk.status}]"):
                                                        st.write(f"**Status:** `{chk.status}` | **Passed:** `{chk.passed}`")
                                                        st.write(f"**Explanation:** {chk.message}")
                                                        if chk.technical_result:
                                                            st.json(chk.technical_result)
                                                        if chk.diagnostic_artifact_url:
                                                            art_filename = os.path.basename(chk.diagnostic_artifact_url)
                                                            with SessionFactory() as db:
                                                                try:
                                                                    u_obj = db.execute(select(User).where(User.id == user_id)).scalar_one_or_none()
                                                                    art_bytes, art_mime = VerificationService.get_diagnostic_artifact(db, ai_report.id, art_filename, u_obj)
                                                                    st.image(art_bytes, caption=f"Diagnostic Visualization: {chk.check_type}")
                                                                except Exception:
                                                                    pass

                                                if ai_report.limitations:
                                                    with st.expander("⚠️ System Limitations & Disclaimers"):
                                                        for lim in ai_report.limitations:
                                                            st.caption(f"• {lim}")

                                        # Audit Trail Expander
                                        with st.expander("📜 Tamper-Evident SHA-256 Audit Trail"):
                                            with SessionFactory() as db:
                                                audit_data = EvidenceService.get_audit_trail(db, ev.id)
                                            if audit_data.is_chain_intact:
                                                st.success(f"✅ Continuous Hash Chain Intact ({audit_data.total_events} events)")
                                            else:
                                                st.error("❌ HASH CHAIN TAMPERED!")
                                            for a_ev in audit_data.events:
                                                st.markdown(f"**{a_ev.created_at.strftime('%Y-%m-%d %H:%M:%S')}** — `{a_ev.event_type}`")
                                                st.caption(f"Prev Hash: `{a_ev.previous_event_hash[:16]}...` ➔ Current Hash: `{a_ev.current_event_hash[:16]}...`")
                                                if a_ev.details_json:
                                                    st.json(a_ev.details_json)

                                        # New Version Expander
                                        if not ev.is_archived:
                                            with st.expander("🔄 Upload Replacement Version"):
                                                with st.form(f"form_version_{ev.id}"):
                                                    v_file = st.file_uploader("Select Replacement File", type=['jpg', 'jpeg', 'png', 'pdf'], key=f"v_file_{ev.id}")
                                                    v_reason = st.text_input("Reason for New Version", placeholder="Clarified high-resolution photos", key=f"v_reason_{ev.id}")
                                                    submit_v = st.form_submit_button("Submit New Version", type="primary")
                                                    if submit_v and v_file:
                                                        try:
                                                            with SessionFactory() as db:
                                                                new_v = EvidenceService.upload_evidence_version(
                                                                    db=db,
                                                                    parent_evidence_id=ev.id,
                                                                    file_bytes=v_file.getvalue(),
                                                                    filename=v_file.name,
                                                                    mime_type=v_file.type,
                                                                    uploader_id=user_id,
                                                                    reason=v_reason
                                                                )
                                                            st.success(f"✅ Created version {new_v.version_number}!")
                                                            time.sleep(1)
                                                            st.rerun()
                                                        except Exception as e:
                                                            st.error(f"Version upload failed: {e}")

    with tab_legacy:
        user_id = st.session_state.get('user_id', 'demo_ngo')
        if not SecurityValidator.check_rate_limit(user_id):
            st.error(f"⚠️ Upload rate limit reached: maximum {Config.RATE_LIMIT_UPLOADS_PER_HOUR} uploads per hour.")
            return
        
        db: DatabaseManager = st.session_state.db
        milestones = db.get_all_milestones()
        pending = [m for m in milestones if m.get('status') in ['PENDING', 'SUBMITTED', 'REJECTED']]
        
        if not pending:
            st.info("No milestones currently awaiting evidence submission.")
            return
        
        milestone_id = st.selectbox(
            "Select Milestone for Evidence Submission",
            [m['id'] for m in pending],
            format_func=lambda x: f"#{x}: {next(m['description'] for m in pending if m['id']==x)[:50]}...",
            key="legacy_ms_sel"
        )
        selected = next(m for m in pending if m['id'] == milestone_id)
        
        st.markdown("---")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📤 Upload Visual Evidence")
            evidence_file = st.file_uploader("Evidence Photograph (Required)", type=['jpg', 'jpeg', 'png'], key="legacy_ev_file")
            before_file = st.file_uploader("Baseline / 'Before' Photograph (Optional for comparative analysis)", type=['jpg', 'jpeg', 'png'], key="legacy_before_file")
            
            if evidence_file:
                file_bytes = evidence_file.getvalue()
                is_valid_mime, mime = SecurityValidator.validate_mime_type(file_bytes)
                is_valid_size, size_mb = SecurityValidator.validate_file_size(file_bytes)
                
                if not is_valid_mime:
                    st.error(f"❌ Unsupported image format: {mime}. Allowed: JPEG, PNG.")
                    return
                if not is_valid_size:
                    st.error(f"❌ File exceeds limit: {size_mb:.2f}MB (max {Config.MAX_UPLOAD_SIZE_MB}MB).")
                    return
                
                st.image(evidence_file, caption="Uploaded Evidence Preview", use_container_width=True)
                st.caption(f"Format: {mime} • Size: {size_mb:.2f} MB")
        
        with col2:
            st.subheader("📍 Target Milestone Coordinates")
            lat = selected.get('latitude', 0.0)
            lon = selected.get('longitude', 0.0)
            st.metric("Latitude", f"{lat:.6f}")
            st.metric("Longitude", f"{lon:.6f}")
            st.caption(f"Verification tolerance: within {Config.GPS_RADIUS_METERS:.0f}m if EXIF GPS metadata is present.")
        
        if evidence_file:
            st.markdown("---")
            if st.button("🚀 Analyze & Submit Evidence", type="primary", use_container_width=True, key="legacy_btn_submit"):
                evidence_bytes = evidence_file.getvalue()
                before_bytes = before_file.getvalue() if before_file else None
                
                SecurityValidator.record_upload(user_id)
                
                # Step 1: Storage & Content Addressing
                with st.spinner("Computing cryptographic content address & storing evidence..."):
                    progress = st.progress(20)
                    cid, access_url, storage_mode = EnhancedIPFSManager.upload_to_web3storage(
                        evidence_bytes, f"evidence_{milestone_id}.jpg"
                    )
                    progress.progress(50)
                    st.caption(f"Content Address: `{cid}` ({storage_mode})")
                
                # Step 2: Multi-Indicator Verification
                with st.spinner("Executing computer vision & forensic checks..."):
                    verifier: EnhancedAIVerifier = st.session_state.verifier
                    result = verifier.run_parallel_verification(
                        evidence_bytes=evidence_bytes,
                        before_bytes=before_bytes,
                        expected_lat=selected.get('latitude') if selected.get('latitude') != 0 else None,
                        expected_lon=selected.get('longitude') if selected.get('longitude') != 0 else None
                    )
                    progress.progress(85)
                    
                    # Persist verification report
                    db.submit_evidence(milestone_id, cid, result)
                    progress.progress(100)
                    
                    Logger.log('INFO', f"Evidence submitted for Milestone #{milestone_id}", {
                        'confidence': result.get('confidence'),
                        'pass': result.get('pass')
                    })
                
                st.markdown("---")
                st.subheader("🤖 AI Verification Analysis Report")
                
                # Visual Panels
                col_g1, col_g2 = st.columns(2)
                with col_g1:
                    fig = st.session_state.viz.create_confidence_gauge(result['confidence'])
                    st.plotly_chart(fig, use_container_width=True)
                with col_g2:
                    fig = st.session_state.viz.create_confidence_explanation_panel(result['checks'], result['confidence'])
                    st.plotly_chart(fig, use_container_width=True)
                
                # Detailed Forensic Check Badges
                st.subheader("🔍 Individual Forensic Check Findings")
                for check_name, check_data in result.get('checks', {}).items():
                    status = check_data.get('status', 'PASSED' if check_data.get('passed') else 'FAILED')
                    title = f"{check_name.replace('_', ' ').title()} — [{status}]"
                    with st.expander(title, expanded=True):
                        if status == "PASSED":
                            st.success(f"✅ PASSED: {check_data.get('message', 'Check passed successfully.')}")
                        elif status == "FAILED":
                            st.error(f"❌ FAILED: {check_data.get('message', 'Indicator threshold failed.')}")
                        elif status == "UNAVAILABLE":
                            st.warning(f"⚠️ UNAVAILABLE: {check_data.get('message', 'Inspection capability currently unavailable.')}")
                        elif status == "NOT_APPLICABLE":
                            st.info(f"ℹ️ NOT APPLICABLE: {check_data.get('message', 'Check was not required for this submission.')}")
                        st.json(check_data)
                
                # Annotated Object Detection Overlay
                if result.get('annotated_image'):
                    st.subheader("🎯 Object Detection Inspection Overlay")
                    st.image(result['annotated_image'], use_container_width=True)
                
                st.success("✅ Evidence submitted and recorded in the audit trail. Human Reviewers can now review findings.")
                time.sleep(2)
                st.rerun()


# ============================================================================
# PHASE 9: PROOF-OF-IMPACT CERTIFICATES & IMPACT LEDGER
# ============================================================================

def page_certificates():
    st.header("📜 Proof-of-Impact Certificates & Cryptographic Ledger")
    st.markdown("Verifiable digital impact attestations backed by asymmetric Ed25519 signatures, RFC 8785 canonical claims, and an append-only event ledger.")
    
    tab_gallery, tab_verify, tab_ledger, tab_admin = st.tabs([
        "🎓 My Certificates",
        "🔍 Public Verification",
        "⛓️ Impact Ledger Explorer",
        "⚙️ Certificate Governance"
    ])
    
    with tab_gallery:
        st.subheader("🎓 Verifiable Impact Attestations")
        with SessionFactory() as db:
            from backend.app.db.models.reviews_impact import ImpactCertificate
            from backend.app.db.models.financial import Donation
            from backend.app.db.models.projects import Milestone, Project
            
            # Fetch all or user certificates
            certs = db.query(ImpactCertificate).order_by(ImpactCertificate.created_at.desc()).all()
            
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            total_certs = len(certs)
            active_certs = len([c for c in certs if c.status in ("ISSUED", "ACTIVE")])
            suspended_certs = len([c for c in certs if c.status == "SUSPENDED"])
            revoked_certs = len([c for c in certs if c.status == "REVOKED"])
            
            col_m1.metric("Total Attestations", total_certs)
            col_m2.metric("Active & Verified", active_certs)
            col_m3.metric("Suspended", suspended_certs)
            col_m4.metric("Revoked", revoked_certs)
            
            st.markdown("---")
            if not certs:
                st.info("📜 No Proof-of-Impact certificates issued yet. Certificates are issued automatically upon verified milestone completion.")
            else:
                for c in certs:
                    status_color = "#10B981" if c.status in ("ISSUED", "ACTIVE") else ("#EF4444" if c.status == "REVOKED" else "#F59E0B")
                    with st.expander(f"📜 {c.certificate_number} — [{c.status}] ({c.recipient_name or 'Anonymous'})", expanded=(c.status in ("ISSUED", "ACTIVE"))):
                        c_col1, c_col2 = st.columns([3, 1])
                        with c_col1:
                            st.markdown(f"**Certificate Type:** `{c.certificate_type}`")
                            st.markdown(f"**Verification ID:** `{c.verification_id}`")
                            st.markdown(f"**Status:** <span style='color:{status_color};font-weight:bold;'>{c.status}</span>", unsafe_allow_html=True)
                            st.markdown(f"**Signing Key ID:** `{c.signing_key_id}` | **Algorithm:** `{c.signature_algorithm}`")
                            st.markdown(f"**Payload SHA-256:** `{c.payload_hash}`")
                            
                            # Render download button
                            from backend.app.services.certificate_rendering_service import CertificateRenderingService
                            import json
                            claims_dict = json.loads(c.canonical_payload) if c.canonical_payload else {}
                            v_url = c.qr_code_url or f"http://localhost:8000/api/v1/public/certificates/{c.verification_id}"
                            try:
                                pdf_bytes, _ = CertificateRenderingService.render_certificate_pdf(
                                    claims=claims_dict,
                                    signature_b64=c.signature,
                                    signing_key_id=c.signing_key_id,
                                    verification_url=v_url
                                )
                                st.download_button(
                                    label="📥 Download Signed PDF Certificate",
                                    data=pdf_bytes,
                                    file_name=f"{c.certificate_number}.pdf",
                                    mime="application/pdf",
                                    key=f"dl_{c.id}"
                                )
                            except Exception as ex:
                                st.caption(f"PDF generator ready: {ex}")
                                
                        with c_col2:
                            # Render QR Code
                            from backend.app.services.certificate_rendering_service import CertificateRenderingService
                            qr_bytes = CertificateRenderingService.generate_qr_code_png(v_url)
                            st.image(qr_bytes, caption="Scan to Verify", width=140)
                            st.caption(f"[Verify Link]({v_url})")
                            
                        with st.expander("🔍 Inspect Signed Canonical Claims"):
                            st.code(c.canonical_payload, language="json")
                            st.markdown(f"**Ed25519 Signature:** `{c.signature}`")
                            
            # Legacy TinyDB records section
            st.markdown("---")
            with st.expander("🕰️ Legacy Prototype Local Certificate Records (TinyDB)"):
                st.caption("These records represent legacy digital certificates generated by the prototype before Phase 9 cryptographic upgrade.")
                show_nft_gallery(st.session_state.user_id, st.session_state.db)

    with tab_verify:
        st.subheader("🔍 Public Proof-of-Impact Verification")
        st.markdown("Enter any Certificate Number or high-entropy Verification Token to independently verify authenticity.")
        v_token = st.text_input("Verification Token or ID", placeholder="e.g. CC-CERT-20261112-... or verification_id", key="pub_v_token")
        if st.button("🔍 Verify Certificate", key="btn_pub_verify"):
            if not v_token.strip():
                st.warning("Please enter a token or certificate ID.")
            else:
                with SessionFactory() as db:
                    from backend.app.services.public_verification_service import PublicVerificationService
                    from backend.app.db.models.reviews_impact import ImpactCertificate
                    
                    # Resolve ID
                    resolved_id = v_token.strip()
                    cert_by_num = db.query(ImpactCertificate).filter(
                        (ImpactCertificate.certificate_number == resolved_id) |
                        (ImpactCertificate.verification_id == resolved_id)
                    ).first()
                    
                    if cert_by_num:
                        resolved_id = cert_by_num.verification_id
                        
                    res = PublicVerificationService.verify_certificate(db, resolved_id)
                    
                    outcome = res.get("verification_outcome")
                    if outcome == "VALID":
                        st.success(f"✅ CERTIFICATE VERIFIED: This certificate is authentic, digitally signed by CharityChain, and currently ACTIVE.")
                    elif outcome in ("REVOKED", "SUSPENDED"):
                        st.warning(f"⚠️ {outcome}: Signature is cryptographically valid, but the certificate has been {outcome.lower()} under governance rules.")
                    elif outcome == "PAYLOAD_MODIFIED":
                        st.error("❌ TAMPERING DETECTED: The claim payload does not match the recorded cryptographic hash!")
                    elif outcome == "INVALID_SIGNATURE":
                        st.error("❌ INVALID SIGNATURE: The Ed25519 cryptographic signature failed verification!")
                    else:
                        st.info(f"ℹ️ Status: {outcome} ({res.get('message', '')})")
                        
                    st.json(res)

    with tab_ledger:
        st.subheader("⛓️ Proof-of-Impact Event Ledger")
        st.markdown("Cryptographically chained, tamper-evident audit stream of verified milestone progress, reviews, and releases.")
        with SessionFactory() as db:
            from backend.app.db.models.projects import Milestone
            from backend.app.db.models.reviews_impact import ImpactLedgerStream, ImpactLedgerEvent
            from backend.app.services.impact_ledger_service import ImpactLedgerService
            
            milestones = db.query(Milestone).all()
            if not milestones:
                st.info("No milestones found.")
            else:
                m_map = {f"Milestone #{m.sequence_number}: {m.title} ({m.id[:8]})": m.id for m in milestones}
                selected_label = st.selectbox("Select Milestone Stream", list(m_map.keys()))
                m_id = m_map[selected_label]
                stream_id = f"milestone-{m_id}"
                
                # Action: Verify Stream Chain
                if st.button("🔐 Verify Stream Cryptographic Integrity"):
                    report = ImpactLedgerService.verify_stream_chain(db, stream_id)
                    if report.get("is_valid"):
                        st.success(f"✅ STREAM INTEGRITY VALID: All {report.get('total_events', 0)} events verified without chain breaks or tampering!")
                    else:
                        st.error(f"❌ STREAM TAMPERING DETECTED: {', '.join(report.get('errors', []))}")
                    st.json(report)
                
                # Fetch stream events
                events = db.query(ImpactLedgerEvent).filter(
                    ImpactLedgerEvent.stream_id == stream_id
                ).order_by(ImpactLedgerEvent.sequence_number.asc()).all()
                
                if not events:
                    st.caption("No events recorded in this milestone stream yet.")
                else:
                    for ev in events:
                        st.markdown(f"""
                        <div style="background:#F8FAFC; padding:10px; border-left:4px solid #0284C7; margin-bottom:8px; border-radius:4px;">
                            <strong>Seq #{ev.sequence_number} | {ev.event_type}</strong> — <small>{ev.timestamp.isoformat()}</small><br/>
                            <small><b>Prev Hash:</b> <code>{ev.previous_event_hash[:16]}...</code> | <b>Current Hash:</b> <code>{ev.current_event_hash[:16]}...</code></small>
                        </div>
                        """, unsafe_allow_html=True)
                        with st.expander(f"View Event #{ev.sequence_number} Payload"):
                            st.code(ev.payload_json, language="json")

    with tab_admin:
        st.subheader("⚙️ Certificate Governance & Key Management")
        with SessionFactory() as db:
            from backend.app.db.models.reviews_impact import ImpactCertificate, IssuerSigningKey
            from backend.app.services.certificate_service import CertificateService
            from backend.app.services.key_management_service import KeyManagementService
            
            st.markdown("#### 🔑 Active Ed25519 Issuer Keys")
            keys = db.query(IssuerSigningKey).all()
            if keys:
                for k in keys:
                    st.markdown(f"- **Key ID:** `{k.key_id}` | **Algorithm:** `{k.algorithm}` | **Status:** `{k.status}` (Activated: {k.activated_at.strftime('%Y-%m-%d')})")
            else:
                st.caption("No keys in database. Ensuring active key is registered...")
                KeyManagementService.ensure_active_key_registered(db)
                st.rerun()
                
            st.markdown("---")
            st.markdown("#### 🛑 Certificate Lifecycle Control")
            certs = db.query(ImpactCertificate).all()
            if certs:
                c_map = {f"{c.certificate_number} — [{c.status}]": c.id for c in certs}
                target_cert_label = st.selectbox("Select Certificate to Manage", list(c_map.keys()))
                target_cert_id = c_map[target_cert_label]
                
                gov_col1, gov_col2 = st.columns(2)
                with gov_col1:
                    st.markdown("##### Suspend Certificate")
                    susp_reason = st.text_input("Suspension Reason", placeholder="e.g. Under investigation for disputed evidence", key="susp_r")
                    if st.button("⚠️ Suspend Certificate", key="btn_susp"):
                        if not susp_reason.strip():
                            st.error("Reason is required.")
                        else:
                            try:
                                CertificateService.suspend_certificate(db, target_cert_id, reason_code="GOVERNANCE_HOLD", reason_description=susp_reason)
                                st.success("Certificate suspended.")
                                time.sleep(1)
                                st.rerun()
                            except Exception as e:
                                st.error(str(e))
                                
                with gov_col2:
                    st.markdown("##### Revoke Certificate")
                    rev_reason = st.selectbox("Revocation Reason Code", [
                        "FULL_REFUND", "CHARGEBACK", "REVIEW_OVERTURNED", "FRAUDULENT_EVIDENCE", "INCORRECT_CLAIM", "ADMINISTRATIVE_CORRECTION"
                    ], key="rev_r_code")
                    rev_desc = st.text_input("Detailed Revocation Description", placeholder="e.g. Donor contribution fully refunded", key="rev_desc")
                    if st.button("🚫 Revoke Certificate", key="btn_rev"):
                        try:
                            CertificateService.revoke_certificate(db, target_cert_id, reason_code=rev_reason, reason_description=rev_desc)
                            st.success("Certificate revoked.")
                            time.sleep(1)
                            st.rerun()
                        except Exception as e:
                            st.error(str(e))


# ============================================================================
# PAGE: NOTIFICATIONS & WEBHOOKS (PHASE 10)
# ============================================================================

def page_notifications_and_webhooks():
    """Interactive Notification Center, Communication Preferences, and Webhook Subscriptions."""
    st.header("🔔 Notifications, Webhooks & Communications")
    st.markdown("Multi-channel transactional notifications, donor & NGO preference management, and HMAC-signed outbound webhooks.")

    user_id, role, full_name, email = get_current_user_info()

    with SessionFactory() as db:
        # Fallback for guest/demo browsing: select first user in system if not logged in
        effective_user_id = user_id
        if not effective_user_id:
            from backend.app.db.models.users import User
            sample_user = db.query(User).first()
            if sample_user:
                effective_user_id = sample_user.id
                st.info(f"Viewing as demo user: **{sample_user.full_name or sample_user.email}** ({sample_user.role.value if hasattr(sample_user.role, 'value') else sample_user.role})")
            else:
                st.warning("Please sign in or register to access notifications.")
                return

        tab_inbox, tab_preferences, tab_webhooks = st.tabs([
            "📥 Notification Center",
            "⚙️ Communication Preferences",
            "🔗 Outbound Webhooks (NGO / Admin)"
        ])

        # --------------------------------------------------------------------
        # TAB 1: NOTIFICATION CENTER
        # --------------------------------------------------------------------
        with tab_inbox:
            unread_count = NotificationService.get_unread_count(db, effective_user_id)
            c1, c2, c3 = st.columns([2, 1, 1])
            with c1:
                st.subheader(f"Your Notifications ({unread_count} unread)")
            with c2:
                show_unread = st.checkbox("Unread only", value=False, key="notif_unread_only")
            with c3:
                if st.button("Mark All Read", key="btn_mark_all_read"):
                    updated = NotificationService.mark_all_as_read(db, effective_user_id)
                    st.success(f"Marked {updated} notifications as read.")
                    time.sleep(0.5)
                    st.rerun()

            notifications, total = NotificationService.list_notifications(
                db, user_id=effective_user_id, unread_only=show_unread, page_size=25
            )

            if not notifications:
                st.info("No notifications to display in this view.")
            else:
                for notif in notifications:
                    card_icon = {
                        "DONATIONS": "💰",
                        "PAYMENTS": "💳",
                        "PROJECTS": "📋",
                        "MILESTONES": "🎯",
                        "EVIDENCE": "📸",
                        "VERIFICATION": "🤖",
                        "REVIEWS": "🔍",
                        "DISBURSEMENTS": "💸",
                        "CERTIFICATES": "📜",
                        "SECURITY": "🔒"
                    }.get(notif.category.value if hasattr(notif.category, 'value') else str(notif.category), "🔔")

                    status_badge = "🔵 **UNREAD**" if not notif.is_read else "⚪ Read"
                    with st.expander(f"{card_icon} {notif.title} — {status_badge} ({notif.created_at.strftime('%Y-%m-%d %H:%M') if notif.created_at else 'Recent'})", expanded=not notif.is_read):
                        st.write(notif.message)
                        if notif.action_url:
                            st.markdown(f"[View Related Record]({notif.action_url})")

                        btn_col1, btn_col2, _ = st.columns([1, 1, 4])
                        with btn_col1:
                            if not notif.is_read:
                                if st.button("Mark Read", key=f"read_{notif.id}"):
                                    NotificationService.mark_as_read(db, effective_user_id, notif.id)
                                    st.rerun()
                        with btn_col2:
                            if st.button("Archive", key=f"arch_{notif.id}"):
                                NotificationService.archive_notification(db, effective_user_id, notif.id)
                                st.rerun()

        # --------------------------------------------------------------------
        # TAB 2: COMMUNICATION PREFERENCES
        # --------------------------------------------------------------------
        with tab_preferences:
            st.subheader("Manage How and When CharityChain Contacts You")
            prefs = NotificationService.get_or_create_preferences(db, effective_user_id)

            st.markdown("#### Delivery Channels")
            ch_col1, ch_col2 = st.columns(2)
            with ch_col1:
                email_en = st.toggle("Email Notifications", value=prefs.email_enabled, key="pref_email_en")
            with ch_col2:
                inapp_en = st.toggle("In-App Notification Center", value=prefs.in_app_enabled, key="pref_inapp_en")

            st.markdown("---")
            st.markdown("#### Notification Topics")
            top_col1, top_col2 = st.columns(2)
            with top_col1:
                don_up = st.checkbox("Donations & Payment Receipts", value=prefs.donation_updates, key="pref_don")
                proj_up = st.checkbox("Project Updates & Announcements", value=prefs.project_updates, key="pref_proj")
                mile_up = st.checkbox("Milestone Status Changes", value=prefs.milestone_updates, key="pref_mile")
            with top_col2:
                verif_up = st.checkbox("AI Evidence Verification Findings", value=prefs.verification_updates, key="pref_verif")
                cert_up = st.checkbox("Impact Certificate Issuance & Revocation", value=prefs.certificate_notifications, key="pref_cert")
                st.checkbox("Security Alerts & Account Actions", value=True, disabled=True, help="Security and critical governance alerts cannot be disabled.", key="pref_sec")

            if st.button("💾 Save Communication Preferences", key="btn_save_prefs"):
                try:
                    NotificationService.update_preferences(
                        db,
                        user_id=effective_user_id,
                        preferences_data={
                            "email_enabled": email_en,
                            "in_app_enabled": inapp_en,
                            "donation_updates": don_up,
                            "project_updates": proj_up,
                            "milestone_updates": mile_up,
                            "verification_updates": verif_up,
                            "certificate_notifications": cert_up
                        }
                    )
                    st.success("Communication preferences saved successfully!")
                except Exception as e:
                    st.error(f"Failed to update preferences: {e}")

        # --------------------------------------------------------------------
        # TAB 3: OUTBOUND WEBHOOK SUBSCRIPTIONS
        # --------------------------------------------------------------------
        with tab_webhooks:
            st.subheader("Automate External Systems via Cryptographically Signed Webhooks")
            st.markdown("CharityChain signs every outbound HTTP POST request with HMAC-SHA256 in the `CharityChain-Signature` header.")

            # Form to register a new subscription
            with st.expander("➕ Register New Webhook Endpoint", expanded=False):
                endpoint_url = st.text_input(
                    "Endpoint Target URL (HTTPS Required in Production)",
                    placeholder="https://your-server.org/webhooks/charitychain",
                    key="wh_url"
                )
                desc = st.text_input("Endpoint Description", placeholder="e.g. Production ERP sync", key="wh_desc")

                event_options = [
                    DomainEventType.DONATION_CREATED.value,
                    DomainEventType.PAYMENT_CAPTURED.value,
                    DomainEventType.PAYMENT_REFUNDED.value,
                    DomainEventType.EVIDENCE_SUBMITTED.value,
                    DomainEventType.AI_VERIFICATION_COMPLETED.value,
                    DomainEventType.REVIEW_APPROVED.value,
                    DomainEventType.DUAL_KEY_APPROVAL_COMPLETED.value,
                    DomainEventType.DISBURSEMENT_COMPLETED.value,
                    DomainEventType.DISBURSEMENT_FAILED.value,
                    DomainEventType.CERTIFICATE_ISSUED.value,
                    DomainEventType.CERTIFICATE_REVOKED.value
                ]
                selected_events = st.multiselect(
                    "Subscribed Event Types (or select all)",
                    options=event_options,
                    default=[DomainEventType.PAYMENT_CAPTURED.value, DomainEventType.CERTIFICATE_ISSUED.value],
                    key="wh_events"
                )

                if st.button("Register Webhook", key="btn_create_wh"):
                    if not endpoint_url:
                        st.error("Please enter a destination URL.")
                    else:
                        try:
                            sub, plaintext_secret = WebhookSubscriptionService.create_subscription(
                                db=db,
                                created_by_user_id=effective_user_id,
                                endpoint_url=endpoint_url.strip(),
                                description=desc.strip() if desc else None,
                                subscribed_events=selected_events or ["*"]
                            )
                            st.success(f"Webhook subscription '{sub.id}' registered successfully!")
                            st.warning(
                                f"⚠️ **HMAC Secret Key:** `{plaintext_secret}`\n\n"
                                "**Important:** Copy this signing secret now. For security purposes, it will never be displayed again."
                            )
                        except Exception as e:
                            st.error(f"Failed to register webhook: {e}")

            # List existing subscriptions
            st.markdown("#### Configured Webhook Subscriptions")
            subs = WebhookSubscriptionService.list_subscriptions(
                db=db, created_by_user_id=effective_user_id if role != "ADMIN" else None
            )

            if not subs:
                st.info("No active webhook subscriptions configured.")
            else:
                for sub in subs:
                    with st.container():
                        sc1, sc2, sc3 = st.columns([3, 1, 2])
                        with sc1:
                            st.markdown(f"**URL:** `{sub.endpoint_url}`")
                            st.caption(f"ID: `{sub.id}` | Events: `{', '.join(sub.subscribed_events or ['*'])}`")
                        with sc2:
                            stat_color = "green" if sub.status.value == "ACTIVE" else "red"
                            st.markdown(f"Status: :{stat_color}[**{sub.status.value}**]")
                            st.caption(f"Failures: {sub.consecutive_failures}")
                        with sc3:
                            act_c1, act_c2 = st.columns(2)
                            with act_c1:
                                if st.button("Rotate Key", key=f"rot_{sub.id}"):
                                    try:
                                        new_secret = WebhookSubscriptionService.rotate_secret(db, sub.id, requesting_user_id=effective_user_id)
                                        st.warning(f"New Secret: `{new_secret}` (Copy now!)")
                                    except Exception as e:
                                        st.error(str(e))
                            with act_c2:
                                if st.button("Delete", key=f"del_{sub.id}"):
                                    try:
                                        WebhookSubscriptionService.delete_subscription(db, sub.id, requesting_user_id=effective_user_id)
                                        st.success("Deleted.")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(str(e))
                        st.markdown("---")


# ============================================================================
# PAGE: ANALYTICS & IMPACT INTELLIGENCE (Phase 11)
# ============================================================================

def page_analytics():
    """Authoritative platform analytics, NGO performance, and administrative health."""
    st.header("📊 Analytics & Impact Intelligence")
    st.markdown("Authoritative financial metrics, NGO milestone progression, and platform health backed by PostgreSQL domain models.")

    user_id, role, full_name, email = get_current_user_info()

    tab_ngo, tab_platform, tab_admin = st.tabs([
        "🏢 NGO Analytics",
        "📈 Platform Overview",
        "⚙️ Administrator Health"
    ])

    with tab_ngo:
        st.subheader("🏢 Organization Impact & Funding Analytics")
        with SessionFactory() as db_session:
            from backend.app.db.models.users import Organization, OrganizationMembership
            from sqlalchemy import select
            
            if role in ["PLATFORM_ADMIN", "Admin", "Platform Administrator"]:
                orgs = db_session.scalars(select(Organization)).all()
            elif user_id and user_id != "demo_user":
                orgs = db_session.scalars(
                    select(Organization)
                    .join(OrganizationMembership, Organization.id == OrganizationMembership.organization_id)
                    .where(OrganizationMembership.user_id == user_id)
                ).all()
            else:
                orgs = db_session.scalars(select(Organization)).all()

            if not orgs:
                st.info("No registered organizations found. Please register an organization to view NGO analytics.")
            else:
                org_options = {o.name: o.id for o in orgs}
                selected_org_name = st.selectbox("Select Organization", list(org_options.keys()))
                selected_org_id = org_options[selected_org_name]

                try:
                    overview = NGOAnalyticsService.get_ngo_overview(db_session, selected_org_id)
                    funding = NGOAnalyticsService.get_funding_analytics(db_session, selected_org_id)
                    projects = NGOAnalyticsService.get_project_performance(db_session, selected_org_id)
                    evidence_stats = NGOAnalyticsService.get_evidence_and_verification_analytics(db_session, selected_org_id)

                    # Top KPI Cards
                    k1, k2, k3, k4 = st.columns(4)
                    with k1:
                        st.metric("Total Captured", overview["financial"]["total_captured_formatted"])
                    with k2:
                        st.metric("Net Funding", overview["financial"]["net_funding_formatted"])
                    with k3:
                        st.metric("Confirmed Payouts", overview["financial"]["disbursed_formatted"])
                    with k4:
                        st.metric("Available Balance", overview["financial"]["available_funds_formatted"])

                    st.markdown("---")
                    p1, p2, p3, p4 = st.columns(4)
                    with p1:
                        st.metric("Active Projects", overview["projects"]["active"])
                    with p2:
                        st.metric("Completed Projects", overview["projects"]["completed"])
                    with p3:
                        st.metric("Completed Milestones", overview["milestones"]["completed"])
                    with p4:
                        st.metric("Verified Milestones", overview["milestones"]["verified"])

                    st.markdown("---")
                    c_left, c_right = st.columns([1, 1])

                    with c_left:
                        st.subheader("🎯 Donation Size Distribution")
                        dist = funding.get("size_distribution", {})
                        if any(dist.values()):
                            labels = ["< $25", "$25-$100", "$100-$500", "> $500"]
                            values = [dist.get("under_25", 0), dist.get("25_to_100", 0), dist.get("100_to_500", 0), dist.get("over_500", 0)]
                            fig_pie = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.4)])
                            fig_pie.update_layout(height=320, margin=dict(t=20, b=20, l=20, r=20))
                            st.plotly_chart(fig_pie, use_container_width=True)
                        else:
                            st.info("No donation distribution data recorded yet.")

                    with c_right:
                        st.subheader("🤖 Evidence & Verification Status")
                        ev_c1, ev_c2 = st.columns(2)
                        with ev_c1:
                            st.metric("Evidence Uploaded", evidence_stats["evidence_files_submitted"])
                            st.metric("Review Cases", evidence_stats["review_cases_total"])
                        with ev_c2:
                            st.metric("Human Approved", evidence_stats["approved"])
                            st.metric("Approval Rate", f"{evidence_stats['approval_rate']}%")

                    st.markdown("---")
                    st.subheader("📋 Project Performance & Progress")
                    if projects:
                        st.dataframe(projects, use_container_width=True)
                    else:
                        st.info("No active projects for this organization.")

                    # Report Export
                    st.markdown("---")
                    st.subheader("📥 Export Impact & Performance Reports")
                    exp_col1, exp_col2 = st.columns([1, 1])
                    with exp_col1:
                        csv_data = AnalyticsExportService.export_ngo_projects_csv(db_session, selected_org_id)
                        st.download_button(
                            label="📥 Download Projects CSV Report",
                            data=csv_data,
                            file_name=f"charitychain_ngo_projects_{selected_org_id[:8]}.csv",
                            mime="text/csv"
                        )
                    with exp_col2:
                        report_data = NGOAnalyticsService.generate_impact_report(db_session, selected_org_id)
                        text_report = AnalyticsExportService.generate_formatted_report_text(
                            report_title=f"Organization Impact Report - {selected_org_name}",
                            organization_or_donor=selected_org_name,
                            metrics={
                                "Total Captured Funding": overview["financial"]["total_captured_formatted"],
                                "Net Funding": overview["financial"]["net_funding_formatted"],
                                "Confirmed Payouts": overview["financial"]["disbursed_formatted"],
                                "Active Projects": overview["projects"]["active"],
                                "Completed Projects": overview["projects"]["completed"],
                                "Verified Milestones": overview["milestones"]["verified"],
                                "Evidence Files Submitted": evidence_stats["evidence_files_submitted"]
                            },
                            disclaimer=report_data["disclaimer"]
                        )
                        st.download_button(
                            label="📄 Download Formatted Impact Report (TXT)",
                            data=text_report,
                            file_name=f"charitychain_impact_report_{selected_org_id[:8]}.txt",
                            mime="text/plain"
                        )

                except Exception as e:
                    st.error(f"Error loading NGO analytics: {e}")

    with tab_platform:
        st.subheader("📈 Platform-Wide KPIs & Operational Metrics")
        try:
            with SessionFactory() as db_session:
                overview = PlatformAnalyticsService.get_platform_overview(db_session)

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Registered Users", overview["users"]["total"])
                st.caption(f"Active Users: {overview['users']['active']}")
            with col2:
                st.metric("Total Projects", overview["projects"]["total"])
                st.caption(f"Active: {overview['projects']['active']} | Completed: {overview['projects']['completed']}")
            with col3:
                st.metric("Verified NGOs", overview["organizations"]["verified"])
                st.caption(f"Total Listed: {overview['organizations']['total']}")
            with col4:
                st.metric("Total Captured", overview["financial"]["captured_formatted"])
                st.caption(f"Net: {overview['financial']['net_funding_formatted']}")

            st.markdown("---")
            # Visual Financial Flow
            st.subheader("💳 Financial Operations Breakdown")
            fin = overview["financial"]
            categories = ["Gross Captured", "Refunds", "Net Funding", "Confirmed Disbursements"]
            amounts = [fin["captured_cents"] / 100.0, fin["refunded_cents"] / 100.0, fin["net_funding_cents"] / 100.0, fin["disbursed_cents"] / 100.0]

            fig_bar = go.Figure(data=[
                go.Bar(
                    x=categories,
                    y=amounts,
                    marker_color=["#2b6cb0", "#e53e3e", "#319795", "#805ad5"],
                    text=[f"${a:,.2f}" for a in amounts],
                    textposition="auto"
                )
            ])
            fig_bar.update_layout(
                title="Platform Financial Lifecycle ($ USD)",
                yaxis_title="Amount ($ USD)",
                height=380,
                margin=dict(t=40, b=20, l=40, r=20)
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                st.metric("Proof-of-Impact Certificates Issued", overview["impact"]["certificates_issued"])
            with c2:
                st.metric("Transactional Notifications Dispatched", overview["impact"]["notifications_processed"])

        except Exception as e:
            st.error(f"Error loading platform analytics: {e}")

    with tab_admin:
        st.subheader("⚙️ Administrator Telemetry & Financial Health")
        try:
            with SessionFactory() as db_session:
                fin_ops = PlatformAnalyticsService.get_financial_operations_analytics(db_session)
                rev_ops = PlatformAnalyticsService.get_review_and_verification_analytics(db_session)

            st.markdown("#### ⚖️ Financial Ledger & Trial Balance Verification")
            ldg = fin_ops["financial_ledger_integrity"]
            bal_col1, bal_col2, bal_col3 = st.columns(3)
            with bal_col1:
                st.metric("Total Debits", f"${ldg['total_debits_cents']/100:,.2f}")
            with bal_col2:
                st.metric("Total Credits", f"${ldg['total_credits_cents']/100:,.2f}")
            with bal_col3:
                if ldg["is_balanced"]:
                    st.success("✅ Ledger Balanced (0 Discrepancy)")
                else:
                    st.error(f"❌ Imbalance: ${ldg['discrepancy_cents']/100:,.2f}")

            st.markdown("---")
            st.markdown("#### ⚡ Operations & Queues")
            o1, o2, o3 = st.columns(3)
            with o1:
                st.metric("Capture Success Rate", f"{fin_ops['payment_operations']['capture_success_rate']}%")
                st.caption(f"Refund Rate: {fin_ops['payment_operations']['refund_rate']}%")
            with o2:
                st.metric("Payout Success Rate", f"{fin_ops['payout_operations']['payout_success_rate']}%")
                st.caption(f"Failed Payouts: {fin_ops['payout_operations']['failed_payouts']}")
            with o3:
                st.metric("Open Discrepancies", fin_ops["reconciliation"]["open_discrepancies_count"])
                st.caption("Reconciliation exceptions requiring review")

            st.markdown("---")
            st.markdown("#### 🤖 Review & Verification Telemetry")
            r1, r2 = st.columns(2)
            with r1:
                st.metric("Review Backlog", rev_ops["human_review_pipeline"]["backlog_pending"])
                st.caption(f"Completion Rate: {rev_ops['human_review_pipeline']['completion_rate']}%")
            with r2:
                st.metric("AI Confidence Average", f"{rev_ops['ai_verification_engine']['average_confidence']}%")
                st.caption(f"Flagged Rate: {rev_ops['ai_verification_engine']['flag_rate']}%")

        except Exception as e:
            st.error(f"Error loading administrator telemetry: {e}")


# ============================================================================
# PAGE: GEOGRAPHIC IMPACT MAP (Phase 11)
# ============================================================================

def page_impact_map():
    """Interactive geographic visualization with location privacy controls."""
    st.header("🗺️ Global Project Impact Locations")
    st.markdown("Explore the geographic reach and verified impact of CharityChain projects. Sensitive locations (such as domestic violence shelters and medical outposts) are aggregated to coarse regional centroids.")

    f_col1, f_col2, f_col3 = st.columns(3)
    with f_col1:
        cat_filter = st.selectbox("Filter Category", ["All", "Community Development", "Clean Energy", "Education", "Healthcare", "Wildlife"], index=0)
    with f_col2:
        status_filter = st.selectbox("Filter Status", ["All", "ACTIVE", "COMPLETED", "DRAFT"], index=0)
    with f_col3:
        country_filter = st.text_input("Country (e.g. USA, IND, KEN)", value="")

    category_arg = None if cat_filter == "All" else cat_filter
    status_arg = None if status_filter == "All" else status_filter
    country_arg = None if not country_filter.strip() else country_filter.strip()

    try:
        with SessionFactory() as db_session:
            markers = GeographicAnalyticsService.get_public_impact_map_data(
                db_session, category=category_arg, status=status_arg, country=country_arg
            )
            density = GeographicAnalyticsService.get_regional_density_aggregates(db_session)

        if not markers:
            st.info("No projects match the selected geographic criteria.")
        else:
            if FOLIUM_AVAILABLE:
                center_lat = sum(m["latitude"] for m in markers) / len(markers)
                center_lon = sum(m["longitude"] for m in markers) / len(markers)
                m_obj = folium.Map(location=[center_lat, center_lon], zoom_start=2, tiles="OpenStreetMap")

                status_colors = {
                    "ACTIVE": "blue",
                    "COMPLETED": "green",
                    "RELEASED": "purple",
                    "PENDING": "orange"
                }

                for item in markers:
                    color = status_colors.get(item["status"], "gray")
                    privacy_badge = "<span style='color: #e53e3e;'>🔒 Regional Centroid (Location Protected)</span><br>" if item["is_obfuscated"] else ""
                    popup_html = f"""
                    <div style="width: 250px;">
                        <h4>{item['title']}</h4>
                        <p><strong>NGO:</strong> {item['organization_name']}</p>
                        <p><strong>Category:</strong> {item['category']}</p>
                        <p><strong>Status:</strong> {item['status']}</p>
                        <p><strong>Target:</strong> {item['target_formatted']}</p>
                        <p><strong>Raised:</strong> {item['raised_formatted']}</p>
                        {privacy_badge}
                    </div>
                    """
                    folium.Marker(
                        location=[item["latitude"], item["longitude"]],
                        popup=folium.Popup(popup_html, max_width=280),
                        tooltip=f"{item['title']} ({item['organization_name']})",
                        icon=folium.Icon(color=color, icon="info-sign")
                    ).add_to(m_obj)

                folium_static(m_obj, width=1200, height=550)

        st.markdown("---")
        st.subheader("🌍 Regional Project Density Breakdown")
        if density and density.get("density"):
            st.dataframe(density["density"], use_container_width=True)

    except Exception as e:
        st.error(f"Error loading impact map: {e}")


# ============================================================================
# PAGE: PUBLIC TRANSPARENCY PORTAL (Phase 11)
# ============================================================================

def page_public_transparency():
    """Unauthenticated public transparency portal displaying verifiable platform statistics."""
    st.header("🌐 Public Transparency Portal")
    st.markdown("Independent, verifiable charitable metrics computed directly from authoritative database records. Open to all donors, NGOs, and the public.")

    tab_overview, tab_projects, tab_ngos, tab_methodology = st.tabs([
        "📊 Global Overview",
        "📁 Public Project Explorer",
        "🏢 Verified NGO Directory",
        "📜 Transparency Methodology"
    ])

    with tab_overview:
        st.subheader("🌐 Global Platform Impact & Financial Transparency")
        try:
            with SessionFactory() as db_session:
                pub_overview = PublicTransparencyService.get_global_overview(db_session)

            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.metric("Listed Organizations", pub_overview["organizations"]["total_listed"])
                st.caption(f"Verified: {pub_overview['organizations']['verified']}")
            with c2:
                st.metric("Published Projects", pub_overview["projects"]["published"])
                st.caption(f"Active: {pub_overview['projects']['active']} | Completed: {pub_overview['projects']['completed']}")
            with c3:
                st.metric("Verified Milestones", pub_overview["milestones"]["verified"])
                st.caption(f"Total: {pub_overview['milestones']['total']}")
            with c4:
                st.metric("Certificates Issued", pub_overview["impact_certificates_issued"])
                st.caption("Cryptographically verifiable attestations")

            st.markdown("---")
            st.subheader("💳 Financial Clarity")
            st.write("CharityChain enforces strict separation between gross donor pledges, settled captures, refunds, and actual NGO payouts.")
            
            f_col1, f_col2, f_col3, f_col4 = st.columns(4)
            fin_t = pub_overview["financial_transparency"]
            with f_col1:
                st.metric("Captured Donations", fin_t["captured_formatted"])
            with f_col2:
                st.metric("Donor Refunds", f"${fin_t['refunded_cents']/100:,.2f}")
            with f_col3:
                st.metric("Net Platform Funding", fin_t["net_donations_formatted"])
            with f_col4:
                st.metric("Confirmed Disbursements", fin_t["disbursed_formatted"])

        except Exception as e:
            st.error(f"Error loading public transparency overview: {e}")

    with tab_projects:
        st.subheader("📁 Published Charitable Projects")
        try:
            with SessionFactory() as db_session:
                pub_projects = PublicTransparencyService.get_public_projects(db_session, limit=25)

            if not pub_projects:
                st.info("No public projects listed at this time.")
            else:
                for proj in pub_projects:
                    with st.expander(f"🎗️ {proj['title']} — {proj['raised_formatted']} / {proj['target_formatted']} ({proj['funding_percent']}%)", expanded=False):
                        st.markdown(f"**NGO:** {proj['organization_name']} | **Category:** {proj['category']} | **Status:** `{proj['status']}`")
                        if proj["summary"]:
                            st.write(proj["summary"])
                        st.progress(min(1.0, proj["funding_percent"] / 100.0))
                        st.caption(f"Milestones: {proj['milestones_completed']} of {proj['milestones_total']} completed ({proj['milestones_verified']} verified)")
        except Exception as e:
            st.error(f"Error loading public projects: {e}")

    with tab_ngos:
        st.subheader("🏢 Verified NGO Profiles")
        try:
            with SessionFactory() as db_session:
                from backend.app.db.models.users import Organization
                from sqlalchemy import select
                orgs = db_session.scalars(select(Organization)).all()

            if not orgs:
                st.info("No organizations registered yet.")
            else:
                for org in orgs:
                    with SessionFactory() as db_s:
                        prof = PublicTransparencyService.get_public_organization_profile(db_s, org.id)
                    if prof:
                        st.markdown(f"#### 🏛️ {prof['name']} — [{prof['status']}]")
                        p1, p2, p3 = st.columns(3)
                        with p1:
                            st.metric("Public Projects", prof["total_public_projects"])
                        with p2:
                            st.metric("Total Raised", prof["total_raised_formatted"])
                        with p3:
                            st.metric("Confirmed Disbursements", prof["confirmed_disbursements_formatted"])
                        st.markdown("---")
        except Exception as e:
            st.error(f"Error loading NGO profiles: {e}")

    with tab_methodology:
        st.subheader("📜 Transparency Principles & Calculation Methodology")
        method = PublicTransparencyService.get_methodology()
        st.info("💡 **Core Guiding Principle:** " + method["principles"][0])

        st.markdown("#### 1. Financial Treatment")
        for k, v in method["financial_calculation"].items():
            st.markdown(f"- **{k.replace('_', ' ').title()}:** {v}")

        st.markdown("#### 2. Verification & Review Invariants")
        for k, v in method["verification_and_review"].items():
            st.markdown(f"- **{k.replace('_', ' ').title()}:** {v}")

        st.markdown("#### 3. Privacy & Exclusion Protections")
        st.markdown(method["privacy_and_exclusions"]["excluded_data"])


# ============================================================================
# MAIN ROUTING
# ============================================================================

def main():
    st.set_page_config(
        page_title="CharityChain — Evidence-Backed Impact",
        page_icon="🌍",
        layout="wide"
    )
    
    apply_custom_css()
    
    # Initialize fresh DatabaseManager and components
    if 'db' not in st.session_state:
        st.session_state.db = DatabaseManager()
    if 'verifier' not in st.session_state:
        st.session_state.verifier = EnhancedAIVerifier()
    if 'viz' not in st.session_state:
        st.session_state.viz = VisualizationManager()
    if 'user_role' not in st.session_state:
        st.session_state.user_role = 'Guest'
    if 'user_id' not in st.session_state:
        st.session_state.user_id = 'demo_user'
    if 'dark_mode' not in st.session_state:
        st.session_state.dark_mode = False
    
    st.markdown('<h1 class="main-header">🌍 CharityChain</h1>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #555;'>Give with Evidence. Release with Confidence.</p>", 
                unsafe_allow_html=True)
    
    # Sidebar Navigation & Utilities
    with st.sidebar:
        st.markdown("### 🌍 CharityChain v3.0")
        add_dark_mode_toggle()
        RoleManager.login_page()
        st.markdown("---")
        
        nav_options = [
            "Home",
            "Discover Projects",
            "Donor Dashboard",
            "Organization Onboarding",
            "Project Management",
            "Reviewer Queue",
            "Financial Operations",
            "Submit Evidence",
            "Analytics",
            "Impact Map",
            "Certificates",
            "Notifications & Webhooks",
            "Public Transparency"
        ]

        page = st.radio(
            "Navigation",
            nav_options,
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        total_projects = len(st.session_state.db.get_all_milestones())
        st.metric("Registered Milestones", total_projects)
        
        # Explicit Seed / Demo Reload Button
        if st.sidebar.button("🔄 Reset / Seed Demo Projects", help="Explicitly resets database with standard verifiable demo milestones"):
            if generate_demo_data(force=True):
                st.success("✅ Sample projects loaded successfully!")
                st.rerun()
            else:
                st.error("❌ Failed to reload sample projects.")
    
    # Page Routing
    if page == "Home":
        page_home()
    elif page in ["Discover Projects", "Donate"]:
        page_discover_projects()
    elif page in ["Donor Dashboard", "Donations"]:
        page_donor_dashboard()
    elif page == "Organization Onboarding":
        page_organization_onboarding()
    elif page in ["Project Management", "Create"]:
        page_project_management()
    elif page == "Reviewer Queue":
        page_reviewer_queue()
    elif page == "Financial Operations":
        page_financial_operations()
    elif page in ["Submit Evidence", "Evidence"]:
        page_submit_evidence()
    elif page == "Analytics":
        page_analytics()
    elif page in ["Impact Map", "Map"]:
        page_impact_map()
    elif page == "Certificates":
        page_certificates()
    elif page == "Notifications & Webhooks":
        page_notifications_and_webhooks()
    elif page == "Public Transparency":
        page_public_transparency()

if __name__ == "__main__":
    try:
        main()
        st.caption("CharityChain — Verifiable Humanitarian Giving Platform")
    except Exception as e:
        st.error(f"Application error: {e}")
        Logger.log('ERROR', str(e))
