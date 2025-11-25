"""
app.py - Streamlit UI Layer (Clean Separation)
===============================================
This file imports backend logic from charity_tracker.py and provides UI

Usage:
    streamlit run app.py
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
    """Home page with overview"""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🔒 Smart Escrow</h3>
            <p>Blockchain-secured funds</p>
            <small>Solidity 0.8.17 • Polygon</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🤖 AI Verification</h3>
            <p>6-algorithm validation</p>
            <small>YOLOv8 • SSIM • pHash • ORB</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🎁 NFT Rewards</h3>
            <p>Proof of Impact</p>
            <small>ERC-721 • IPFS Metadata</small>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Live stats
    st.subheader("📈 Platform Statistics")
    
    milestones = st.session_state.db.get_all_milestones()
    donations = st.session_state.db.donations.all() if st.session_state.db.db else []
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Projects", len(milestones))
    with col2:
        total_raised = sum(d['amount'] for d in donations) / 10**18
        st.metric("Total Raised", f"{total_raised:.4f} ETH")
    with col3:
        completed = len([m for m in milestones if m['status'] == 'RELEASED'])
        st.metric("Completed", completed)
    with col4:
        nfts = st.session_state.db.nfts.all() if st.session_state.db.db else []
        st.metric("NFTs Minted", len(nfts))
    
    st.markdown("---")
    st.success("🚀 **Get Started:** Choose your role in the sidebar and start making verifiable impact!")

# ============================================================================
# PAGE: CREATE PROJECT
# ============================================================================

def page_create_project():
    """Create project page"""
    st.header("🏗️ Create Charitable Project")
    
    if not RoleManager.require_role(['NGO', 'Oracle']):
        return
    
    with st.form("create_project"):
        st.subheader("Project Details")
        
        owner = st.text_input("NGO Name", value="Green Earth Foundation")
        description = st.text_area("Description", value="Plant 1000 trees in Amazon rainforest")
        
        goal_eth = st.number_input("Funding Goal (ETH)", min_value=0.01, value=2.5, step=0.1)
        
        st.subheader("📍 GPS Location")
        col_lat, col_lon = st.columns(2)
        with col_lat:
            latitude = st.number_input("Latitude", value=-3.4653, format="%.6f")
        with col_lon:
            longitude = st.number_input("Longitude", value=-62.2159, format="%.6f")
        
        submitted = st.form_submit_button("🚀 Create Milestone", type="primary", use_container_width=True)
        
        if submitted:
            goal_wei = int(goal_eth * 10**18)
            
            with st.spinner("Creating on blockchain..."):
                time.sleep(1)  # Simulate blockchain tx
                milestone_id = st.session_state.db.create_milestone(owner, goal_wei, description, latitude, longitude)
                
                Logger.log('INFO', f"Milestone created: {milestone_id}", {
                    'owner': owner, 'goal': goal_wei, 'lat': latitude, 'lon': longitude
                })
                
                st.success(f"✅ Milestone #{milestone_id} created successfully!")
                st.balloons()
                
                st.markdown(f"""
                <div class="success-box" style="background: #d4edda; padding: 1rem; border-radius: 10px; margin-top: 1rem;">
                    <h4>🎉 Project Created!</h4>
                    <p><strong>Milestone ID:</strong> {milestone_id}</p>
                    <p><strong>Share with donors:</strong> <code>#{milestone_id}</code></p>
                    <p><strong>Location:</strong> {latitude:.4f}, {longitude:.4f}</p>
                </div>
                """, unsafe_allow_html=True)

# ============================================================================
# PAGE: DONATE
# ============================================================================

def page_donate():
    """Donate page"""
    st.header("💰 Donate to Verified Projects")
    
    milestones = st.session_state.db.get_all_milestones()
    
    if not milestones:
        st.warning("No projects available yet. Create one first!")
        return
    
    # Filter
    filter_status = st.multiselect(
        "Filter by Status",
        ['PENDING', 'SUBMITTED', 'VERIFIED', 'RELEASED'],
        default=['PENDING', 'SUBMITTED']
    )
    
    filtered = [m for m in milestones if m['status'] in filter_status]
    
    st.markdown(f"**{len(filtered)} projects available**")
    st.markdown("---")
    
    for milestone in filtered:
        with st.expander(f"🎯 #{milestone['id']}: {milestone['description'][:60]}...", expanded=True):
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**NGO:** {milestone['owner']}")
                st.markdown(f"**Status:** `{milestone['status']}`")
                
                # NGO reputation
                profile = st.session_state.db.get_ngo_profile(milestone['owner'])
                st.markdown(f"**Reputation:** {'⭐' * (profile['reputation_score'] // 20)} ({profile['reputation_score']}/100)")
                
                progress = min(100, (milestone['raised'] / milestone['goal']) * 100) if milestone['goal'] > 0 else 0
                st.progress(progress / 100)
                st.caption(f"{progress:.1f}% funded")
            
            with col2:
                st.metric("Goal", f"{milestone['goal']/10**18:.2f} ETH")
                st.metric("Raised", f"{milestone['raised']/10**18:.4f} ETH")
            
            # Donation form
            with st.form(f"donate_{milestone['id']}"):
                col_name, col_amount = st.columns(2)
                
                with col_name:
                    donor_name = st.text_input("Your Name", value="Anonymous Donor", key=f"name_{milestone['id']}")
                
                with col_amount:
                    amount_eth = st.number_input("Amount (ETH)", min_value=0.001, value=0.1, step=0.01, key=f"amt_{milestone['id']}")
                
                impact_tokens = int(amount_eth * Config.IMPACT_TOKEN_REWARD_RATE)
                st.info(f"🪙 You'll earn **{impact_tokens} Impact Tokens** + NFT on verification!")
                
                if st.form_submit_button("💝 Donate Now", type="primary", use_container_width=True):
                    amount_wei = int(amount_eth * 10**18)
                    
                    with st.spinner("Processing donation..."):
                        time.sleep(1)
                        st.session_state.db.add_donation(milestone['id'], donor_name, amount_wei)
                        
                        Logger.log('INFO', f"Donation: {donor_name} → {amount_eth} ETH", {'milestone': milestone['id']})
                        
                        st.success(f"✅ Donated {amount_eth} ETH!")
                        st.info(f"🎁 Earned {impact_tokens} tokens!")
                        st.balloons()
                        time.sleep(2)
                        st.rerun()

# ============================================================================
# PAGE: SUBMIT EVIDENCE
# ============================================================================

def page_submit_evidence():
    """Submit evidence page with async upload"""
    st.header("📸 Submit Evidence with AI Pre-Check")
    
    if not RoleManager.require_role(['NGO']):
        return
    
    # Check rate limit
    if not SecurityValidator.check_rate_limit(st.session_state.user_id):
        st.error(f"⚠️ Rate limit: Max {Config.RATE_LIMIT_UPLOADS_PER_HOUR} uploads/hour")
        return
    
    milestones = st.session_state.db.get_all_milestones()
    pending = [m for m in milestones if m['status'] in ['PENDING', 'REJECTED']]
    
    if not pending:
        st.info("No milestones awaiting evidence")
        return
    
    milestone_id = st.selectbox(
        "Select Milestone",
        [m['id'] for m in pending],
        format_func=lambda x: f"#{x}: {next(m['description'] for m in pending if m['id']==x)[:50]}..."
    )
    
    selected = next(m for m in pending if m['id'] == milestone_id)
    
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📤 Upload Evidence")
        
        evidence_file = st.file_uploader("Evidence Photo", type=['jpg', 'jpeg', 'png'])
        
        if evidence_file:
            # Validate
            file_bytes = evidence_file.read()
            
            is_valid_mime, mime = SecurityValidator.validate_mime_type(file_bytes)
            is_valid_size, size_mb = SecurityValidator.validate_file_size(file_bytes)
            
            if not is_valid_mime:
                st.error(f"❌ Invalid file type: {mime}")
                return
            
            if not is_valid_size:
                st.error(f"❌ File too large: {size_mb:.2f}MB (max {Config.MAX_UPLOAD_SIZE_MB}MB)")
                return
            
            st.image(evidence_file, caption="Evidence Preview", use_container_width=True)
            st.caption(f"✅ Valid • {size_mb:.2f} MB")
        
        before_file = st.file_uploader("Before Photo (Optional)", type=['jpg', 'jpeg', 'png'])
        
        if before_file:
            st.image(before_file, caption="Before", width=200)
    
    with col2:
        st.subheader("📍 Location")
        st.metric("Expected Lat", f"{selected['latitude']:.6f}")
        st.metric("Expected Lon", f"{selected['longitude']:.6f}")
        st.caption(f"Must be within {Config.GPS_RADIUS_METERS}m")
    
    if evidence_file:
        col_check, col_upload = st.columns(2)
        
        with col_check:
            use_mock = st.checkbox("🎭 Mock AI (Fast)", value=False)
        
        with col_upload:
            if st.button("🚀 Upload & Verify", type="primary", use_container_width=True):
                evidence_bytes = evidence_file.read()
                before_bytes = before_file.read() if before_file else None
                
                # Record upload
                SecurityValidator.record_upload(st.session_state.user_id)
                
                # Upload to IPFS (async)
                with st.spinner("Uploading to IPFS..."):
                    progress = st.progress(0)
                    progress.progress(20)
                    
                    try:
                        # Try async upload
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        cid, ipfs_url = loop.run_until_complete(
                            EnhancedIPFSManager.upload_async(evidence_bytes, f"evidence_{milestone_id}.jpg")
                        )
                        loop.close()
                    except:
                        # Fallback to sync
                        cid, ipfs_url = EnhancedIPFSManager.upload_to_web3storage(
                            evidence_bytes, f"evidence_{milestone_id}.jpg"
                        )
                    
                    progress.progress(50)
                    st.success(f"✅ IPFS CID: `{cid[:20]}...`")
                
                # AI Verification
                with st.spinner("Running AI verification..."):
                    verifier = st.session_state.verifier
                    
                    try:
                        # Try async verification
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        result = loop.run_until_complete(
                            AsyncIOManager.verify_parallel_async(
                                verifier, evidence_bytes, before_bytes,
                                selected['latitude'] if selected['latitude'] != 0 else None,
                                selected['longitude'] if selected['longitude'] != 0 else None
                            )
                        )
                        loop.close()
                    except:
                        # Fallback to sync
                        result = verifier.run_parallel_verification(
                            evidence_bytes, before_bytes,
                            selected['latitude'] if selected['latitude'] != 0 else None,
                            selected['longitude'] if selected['longitude'] != 0 else None,
                            use_mock
                        )
                    
                    progress.progress(80)
                    
                    # Save
                    st.session_state.db.submit_evidence(milestone_id, cid, result)
                    progress.progress(100)
                    
                    Logger.log('INFO', f"Evidence submitted: M{milestone_id}", result)
                    
                    st.markdown("---")
                    st.subheader("🤖 AI Verification Report")
                    
                    # Visualizations
                    col_g1, col_g2 = st.columns(2)
                    
                    with col_g1:
                        fig = st.session_state.viz.create_confidence_gauge(result['confidence'])
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col_g2:
                        fig = st.session_state.viz.create_confidence_explanation_panel(result['checks'], result['confidence'])
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed checks
                    for check_name, check_data in result['checks'].items():
                        with st.expander(f"{check_name.replace('_', ' ').title()}", expanded=False):
                            if check_data['passed']:
                                st.success("✅ Passed")
                            else:
                                st.error("❌ Failed")
                            st.json(check_data)
                    
                    # Annotated image
                    if 'annotated_image' in result and result['annotated_image']:
                        st.subheader("🎯 AI Detection Overlay")
                        st.image(result['annotated_image'], use_container_width=True)
                    
                    st.info("✅ Evidence submitted! Go to 'Oracle' to approve/reject.")
                    time.sleep(2)
                    st.rerun()

# ============================================================================
# MAIN ROUTING
# ============================================================================

def main():
    st.set_page_config(
        page_title="CharityChain",
        page_icon="🌍",
        layout="wide"
    )
    
    apply_custom_css()
    
    # CRITICAL FIX: Always create fresh DatabaseManager
    # This ensures we read the current database file every time
    st.session_state.db = DatabaseManager()
    
    # Initialize other session state (only if not exists)
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
    st.markdown("<p style='text-align: center; font-size: 1.3rem;'>Transparent • Verifiable • Trustworthy</p>", 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/200x80/667eea/ffffff?text=CharityChain", use_container_width=True)
        add_dark_mode_toggle()
        RoleManager.login_page()
        st.markdown("---")
        
        page = st.radio("Navigate",
            ["Home", "Create", "Donate", "Evidence", "Analytics", "Map", "NFTs"],
            label_visibility="collapsed")
        
        st.markdown("---")
        st.metric("Projects", len(st.session_state.db.get_all_milestones()))
        
        # Manual demo data reload button with force reset
        if st.sidebar.button("🚀 Reload Demo Data", help="Force reset database with fresh demo projects"):
            st.session_state.demo_loaded = False  # Reset flag for fresh load
            if generate_demo_data(force=True):
                st.success("✅ Demo data reset and loaded!")
                st.session_state.demo_loaded = True
                st.rerun()
            else:
                st.error("❌ Failed to reload demo data – check logs")
    
    # Route
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
        st.header("🗺️ Project Map")
        m = create_project_map(st.session_state.db.get_all_milestones())
        if m and FOLIUM_AVAILABLE:
            folium_static(m, width=1200, height=600)
    elif page == "NFTs":
        st.header("🎨 My NFTs")
        show_nft_gallery(st.session_state.user_id, st.session_state.db)

if __name__ == "__main__":
    try:
        main()
        st.caption("Built with ❤️ | CharityChain v2.0")
    except Exception as e:
        st.error(f"Error: {e}")
        Logger.log('ERROR', str(e))
