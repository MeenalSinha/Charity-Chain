#!/usr/bin/env python3
"""
Force Streamlit to reload database by clearing session state
This script modifies app_ui.py to force refresh on startup
"""

import sys
from pathlib import Path

print("🔧 FIXING STREAMLIT SESSION STATE ISSUE")
print("=" * 60)

# Read app_ui.py
app_file = Path('app_ui.py')
if not app_file.exists():
    print("❌ app_ui.py not found!")
    sys.exit(1)

content = app_file.read_text()

# Check current auto-load section
if 'if "demo_loaded" not in st.session_state:' in content:
    print("✅ Found auto-load section")
    
    # Replace the auto-load section with a version that forces DB reload
    old_code = '''    # Auto-load demo data on first launch (for hackathon demo)
    if "demo_loaded" not in st.session_state:
        generate_demo_data(force=True)  # ← force reload on first run
        st.session_state.demo_loaded = True'''
    
    new_code = '''    # Auto-load demo data on first launch (for hackathon demo)
    if "demo_loaded" not in st.session_state:
        # Force refresh DatabaseManager to read existing data
        if "db" in st.session_state:
            del st.session_state.db
        init_session_state()  # Recreate db to read current file
        
        # Check if database already has data
        existing = st.session_state.db.get_all_milestones()
        if len(existing) == 0:
            # Only generate if truly empty
            generate_demo_data(force=True)
        
        st.session_state.demo_loaded = True'''
    
    if old_code in content:
        content = content.replace(old_code, new_code)
        app_file.write_text(content)
        print("✅ Updated app_ui.py with database refresh logic")
        print("\n📝 Changes made:")
        print("   • Forces DatabaseManager to reload from file")
        print("   • Checks for existing data before generating")
        print("   • Ensures session state sees current database")
        print("\n🚀 Now run:")
        print("   streamlit run app_ui.py")
    else:
        print("⚠️  Could not find exact code match")
        print("   The auto-load section may have been modified")
else:
    print("❌ Auto-load section not found in app_ui.py")

print("=" * 60)
