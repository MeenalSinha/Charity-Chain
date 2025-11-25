"""
BULLETPROOF main() function for app_ui.py
Copy this ENTIRE function and replace your existing main()
"""

def main():
    st.set_page_config(
        page_title="CharityChain",
        page_icon="🌍",
        layout="wide"
    )
    
    apply_custom_css()
    
    # CRITICAL: Always create fresh DatabaseManager on every run
    # This ensures we always read the latest database file
    st.session_state.db = DatabaseManager()
    
    # Initialize other session state
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
    
    # Check if we need to load demo data
    milestones = st.session_state.db.get_all_milestones()
    
    if len(milestones) == 0 and "first_run_complete" not in st.session_state:
        # Database is empty and we haven't run demo generation yet
        Logger.log("INFO", "Database empty, generating demo data...")
        generate_demo_data(force=True)
        
        # Refresh database to see new data
        st.session_state.db = DatabaseManager()
        milestones = st.session_state.db.get_all_milestones()
        Logger.log("INFO", f"After generation: {len(milestones)} milestones")
        
        st.session_state.first_run_complete = True
    
    st.markdown('<h1 class="main-header">🌍 CharityChain</h1>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.3rem;'>Transparent • Verifiable • Trustworthy</p>", 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/200x80/667eea/ffffff?text=CharityChain", width=None)
        add_dark_mode_toggle()
        RoleManager.login_page()
        st.markdown("---")
        
        # Navigation
        st.markdown("### Navigate")
        page = st.radio(
            "Navigate",
            ["Home", "Create", "Donate", "Evidence", "Analytics", "Map", "NFTs"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Show current count (refresh on every run)
        current_milestones = st.session_state.db.get_all_milestones()
        st.metric("Projects", len(current_milestones))
        
        # Reload button
        if st.sidebar.button("🚀 Reload Demo Data", help="Force reset database with fresh demo projects"):
            generate_demo_data(force=True)
            st.session_state.db = DatabaseManager()  # Refresh
            st.success("✅ Demo data reset and loaded!")
            st.rerun()
    
    # Route to pages
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
        show_nft_gallery()


if __name__ == "__main__":
    main()
