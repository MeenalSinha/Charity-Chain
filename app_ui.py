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
from datetime import datetime
import time


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
    
    db: DatabaseManager = st.session_state.db
    milestones = db.get_all_milestones()
    donations = db.donations.all() if (db.db is not None and db.donations is not None) else []
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Projects", len(milestones))
    with col2:
        total_raised = sum(d.get('amount', 0) for d in donations) / 10**18
        st.metric("Total Raised", f"{total_raised:.4f} ETH")
    with col3:
        completed = len([m for m in milestones if m.get('status') == 'RELEASED'])
        st.metric("Completed Milestones", completed)
    with col4:
        nfts = db.nfts.all() if (db.db is not None and db.nfts is not None) else []
        st.metric("Certificates Issued", len(nfts))
    
    st.markdown("---")
    st.info("💡 **CharityChain Workflow:** Donors fund milestones ➔ NGOs submit photographic evidence ➔ AI analyzes authenticity ➔ Reviewers approve ➔ Funds released ➔ Donors receive Proof-of-Impact certificates.")


# ============================================================================
# PAGE: CREATE PROJECT
# ============================================================================

def page_create_project():
    """Project and milestone creation page for NGOs and administrators."""
    st.header("🏗️ Create Charitable Project Milestone")
    
    if not RoleManager.require_role(['NGO', 'Reviewer']):
        return
    
    with st.form("create_project"):
        st.subheader("Project Details")
        
        owner = st.text_input("NGO Name", value="Green Earth Foundation")
        description = st.text_area("Milestone Description", value="Plant 1000 trees in Amazon rainforest to combat deforestation")
        
        goal_eth = st.number_input("Milestone Funding Goal (ETH)", min_value=0.01, value=2.5, step=0.1)
        
        st.subheader("📍 Target Geolocation (Expected Site)")
        col_lat, col_lon = st.columns(2)
        with col_lat:
            latitude = st.number_input("Latitude", value=-3.4653, format="%.6f")
        with col_lon:
            longitude = st.number_input("Longitude", value=-62.2159, format="%.6f")
        
        submitted = st.form_submit_button("🚀 Register Milestone", type="primary", use_container_width=True)
        
        if submitted:
            goal_wei = int(goal_eth * 10**18)
            
            with st.spinner("Registering project milestone..."):
                time.sleep(0.3)
                milestone_id = st.session_state.db.create_milestone(owner, goal_wei, description, latitude, longitude)
                
                Logger.log('INFO', f"Milestone registered: {milestone_id}", {
                    'owner': owner, 'goal': goal_wei, 'lat': latitude, 'lon': longitude
                })
                
                st.success(f"✅ Milestone #{milestone_id} registered successfully!")
                st.balloons()
                
                st.markdown(f"""
                <div class="metric-card">
                    <h4>🎉 Milestone Registered!</h4>
                    <p><strong>Milestone ID:</strong> #{milestone_id}</p>
                    <p><strong>NGO:</strong> {owner}</p>
                    <p><strong>Funding Target:</strong> {goal_eth:.2f} ETH</p>
                    <p><strong>Target Coordinates:</strong> {latitude:.4f}, {longitude:.4f}</p>
                </div>
                """, unsafe_allow_html=True)


# ============================================================================
# PAGE: DONATE
# ============================================================================

def page_donate():
    """Donation page with milestone progress bars and donor recognition."""
    st.header("💰 Fund Charitable Milestones")
    
    db: DatabaseManager = st.session_state.db
    milestones = db.get_all_milestones()
    
    if not milestones:
        st.warning("No projects available yet. Create one or load sample data from the sidebar!")
        return
    
    filter_status = st.multiselect(
        "Filter by Status",
        ['PENDING', 'SUBMITTED', 'VERIFIED', 'RELEASED'],
        default=['PENDING', 'SUBMITTED']
    )
    
    filtered = [m for m in milestones if m.get('status') in filter_status]
    st.markdown(f"**{len(filtered)} project milestone(s) available**")
    st.markdown("---")
    
    for milestone in filtered:
        m_id = milestone.get('id', 0)
        desc = milestone.get('description', '')
        with st.expander(f"🎯 Milestone #{m_id}: {desc[:60]}...", expanded=True):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**NGO:** {milestone.get('owner', 'Unknown')}")
                st.markdown(f"**Status:** `{milestone.get('status', 'PENDING')}`")
                
                profile = db.get_ngo_profile(milestone.get('owner', ''))
                rep = profile.get('reputation_score', 100)
                st.markdown(f"**Reputation:** {'⭐' * (rep // 20)} ({rep}/100)")
                
                goal = milestone.get('goal', 1)
                raised = milestone.get('raised', 0)
                progress = min(100.0, (raised / goal) * 100.0) if goal > 0 else 0.0
                st.progress(progress / 100.0)
                st.caption(f"{progress:.1f}% funded ({raised/10**18:.4f} / {goal/10**18:.2f} ETH)")
            
            with col2:
                st.metric("Goal", f"{goal/10**18:.2f} ETH")
                st.metric("Raised", f"{raised/10**18:.4f} ETH")
            
            # Donation Form
            with st.form(f"donate_{m_id}"):
                col_name, col_amount = st.columns(2)
                with col_name:
                    donor_name = st.text_input("Donor Identity / Organization", value="Alice Donor", key=f"name_{m_id}")
                with col_amount:
                    amount_eth = st.number_input("Contribution Amount (ETH)", min_value=0.001, value=0.1, step=0.01, key=f"amt_{m_id}")
                
                impact_tokens = int(amount_eth * Config.IMPACT_TOKEN_REWARD_RATE)
                st.caption(f"🎁 Rewards: **{impact_tokens} Impact Tokens** + Proof-of-Impact Certificate upon milestone release.")
                
                if st.form_submit_button("💝 Contribute with Confidence", type="primary", use_container_width=True):
                    amount_wei = int(amount_eth * 10**18)
                    with st.spinner("Recording verified contribution..."):
                        time.sleep(0.3)
                        db.add_donation(m_id, donor_name, amount_wei)
                        Logger.log('INFO', f"Donation: {donor_name} -> {amount_eth} ETH", {'milestone': m_id})
                        st.success(f"✅ Successfully contributed {amount_eth:.4f} ETH to Milestone #{m_id}!")
                        st.balloons()
                        time.sleep(1)
                        st.rerun()


# ============================================================================
# PAGE: SUBMIT EVIDENCE
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
        st.markdown("### 🌍 CharityChain v2.0")
        add_dark_mode_toggle()
        RoleManager.login_page()
        st.markdown("---")
        
        page = st.radio(
            "Navigation",
            ["Home", "Create", "Donate", "Evidence", "Analytics", "Map", "Certificates"],
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
    elif page == "Create":
        page_create_project()
    elif page == "Donate":
        page_donate()
    elif page == "Evidence":
        page_submit_evidence()
    elif page == "Analytics":
        show_analytics_dashboard()
    elif page == "Map":
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
