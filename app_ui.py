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
    DonationStatus, PaymentOrderStatus, PaymentCaptureStatus, PaymentRefundStatus
)
from backend.app.api.schemas import (
    CreateOrganizationRequest, UpdateOrganizationRequest, OrganizationDecisionRequest,
    CreateProjectRequest, UpdateProjectRequest, ProjectDecisionRequest,
    CreateMilestoneRequest, UpdateMilestoneRequest, CreateDonationRequest
)
from backend.app.services.ngo_onboarding_service import NGOOnboardingService
from backend.app.services.project_service import ProjectService
from backend.app.services.donation_service import DonationService
from backend.app.integrations.storage.local import LocalStorageProvider


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
    
    # Platform Statistics
    st.subheader("📈 Live Platform Overview")
    
    # Try fetching database-backed counts
    db_projects_count = 0
    db_orgs_count = 0
    try:
        with SessionFactory() as db:
            from backend.app.db.models.projects import Project
            from backend.app.db.models.users import Organization
            from sqlalchemy import select, func
            db_projects_count = db.scalar(select(func.count(Project.id))) or 0
            db_orgs_count = db.scalar(select(func.count(Organization.id))) or 0
    except Exception:
        pass

    db: DatabaseManager = st.session_state.db
    milestones = db.get_all_milestones()
    donations = db.donations.all() if (db.db is not None and db.donations is not None) else []
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Projects", max(len(milestones), db_projects_count))
    with col2:
        total_raised = sum(d.get('amount', 0) for d in donations) / 10**18
        st.metric("Total Raised", f"{total_raised:.4f} ETH")
    with col3:
        completed = len([m for m in milestones if m.get('status') == 'RELEASED'])
        st.metric("Completed Milestones", completed)
    with col4:
        st.metric("Verified NGOs", db_orgs_count)
    
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

    tab_orgs, tab_projs = st.tabs(["🏢 Organizations Queue", "📋 Projects Queue"])

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
    st.subheader(f"📜 Contribution Records ({total_count})")

    if not donations:
        st.info("You haven't made any contributions yet. Explore verified projects to get started!")
        return

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



# ============================================================================
# PAGE: SUBMIT EVIDENCE (Preserved from Phase 0/1/2)
# ============================================================================

def page_submit_evidence():
    """NGO evidence submission with automated multi-indicator verification."""
    st.header("📸 Submit Evidence of Completed Impact")
    
    if not RoleManager.require_role(['NGO', 'Reviewer']):
        return
    
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
        format_func=lambda x: f"#{x}: {next(m['description'] for m in pending if m['id']==x)[:50]}..."
    )
    selected = next(m for m in pending if m['id'] == milestone_id)
    
    st.markdown("---")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📤 Upload Visual Evidence")
        evidence_file = st.file_uploader("Evidence Photograph (Required)", type=['jpg', 'jpeg', 'png'])
        before_file = st.file_uploader("Baseline / 'Before' Photograph (Optional for comparative analysis)", type=['jpg', 'jpeg', 'png'])
        
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
        if st.button("🚀 Analyze & Submit Evidence", type="primary", use_container_width=True):
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
            "Submit Evidence",
            "Analytics",
            "Impact Map",
            "Certificates"
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
    elif page in ["Submit Evidence", "Evidence"]:
        page_submit_evidence()
    elif page == "Analytics":
        show_analytics_dashboard()
    elif page in ["Impact Map", "Map"]:
        st.header("🗺️ Global Project Impact Locations")
        m = create_project_map(st.session_state.db.get_all_milestones())
        if m and FOLIUM_AVAILABLE:
            folium_static(m, width=1200, height=600)
    elif page == "Certificates":
        st.header("📜 My Proof-of-Impact Certificates")
        show_nft_gallery(st.session_state.user_id, st.session_state.db)

if __name__ == "__main__":
    try:
        main()
        st.caption("CharityChain — Verifiable Humanitarian Giving Platform")
    except Exception as e:
        st.error(f"Application error: {e}")
        Logger.log('ERROR', str(e))
