#!/usr/bin/env python3
"""
Comprehensive test to find why demo data isn't showing
"""
import sys
sys.path.insert(0, '.')

print("🔍 COMPREHENSIVE DEMO DATA DEBUG")
print("=" * 70)

# Test 1: Check if database file has data
print("\n1️⃣ Checking database file...")
from pathlib import Path
db_file = Path('data/charity.db')

if db_file.exists():
    print(f"   ✅ Database file exists: {db_file}")
    print(f"   Size: {db_file.stat().st_size} bytes")
    
    # Read it directly
    from tinydb import TinyDB
    db = TinyDB(str(db_file))
    milestones = db.table('milestones').all()
    donations = db.table('donations').all()
    db.close()
    
    print(f"   📊 Direct file read:")
    print(f"      Milestones: {len(milestones)}")
    print(f"      Donations: {len(donations)}")
    
    if len(milestones) > 0:
        print(f"   ✅ DATABASE HAS {len(milestones)} PROJECTS!")
        for m in milestones:
            print(f"      • {m['description'][:50]}...")
    else:
        print(f"   ❌ DATABASE IS EMPTY!")
else:
    print(f"   ❌ Database file doesn't exist!")

# Test 2: Check DatabaseManager
print("\n2️⃣ Checking DatabaseManager...")
from charity_tracker import DatabaseManager, DB_AVAILABLE

print(f"   DB_AVAILABLE: {DB_AVAILABLE}")

if DB_AVAILABLE:
    db_mgr = DatabaseManager()
    print(f"   ✅ DatabaseManager created")
    print(f"   db_mgr.db: {db_mgr.db}")
    print(f"   db_mgr.db is None: {db_mgr.db is None}")
    
    milestones = db_mgr.get_all_milestones()
    print(f"   📊 DatabaseManager.get_all_milestones(): {len(milestones)}")
    
    if len(milestones) > 0:
        print(f"   ✅ DatabaseManager CAN read data!")
    else:
        print(f"   ❌ DatabaseManager sees EMPTY database!")

# Test 3: Test with Streamlit session state simulation
print("\n3️⃣ Simulating Streamlit session state...")
try:
    import streamlit as st
    from charity_tracker import init_session_state, generate_demo_data
    
    # Clear session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    
    print("   📝 Running init_session_state()...")
    init_session_state()
    
    print(f"   ✅ Session state initialized")
    print(f"   'db' in session_state: {'db' in st.session_state}")
    
    if 'db' in st.session_state:
        milestones = st.session_state.db.get_all_milestones()
        print(f"   📊 st.session_state.db.get_all_milestones(): {len(milestones)}")
        
        if len(milestones) == 0:
            print("   ⚠️  Session state DB is empty, running generate_demo_data...")
            success = generate_demo_data(force=True)
            print(f"   generate_demo_data returned: {success}")
            
            # Check again
            milestones = st.session_state.db.get_all_milestones()
            print(f"   📊 After generate: {len(milestones)} milestones")
            
            if len(milestones) > 0:
                print(f"   ✅ SUCCESS! Data loaded!")
            else:
                print(f"   ❌ FAILED! Still empty after generate_demo_data!")
        else:
            print(f"   ✅ Session state DB already has data!")
            
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Check if issue is in get_all_milestones
print("\n4️⃣ Testing get_all_milestones method...")
from charity_tracker import DatabaseManager

db = DatabaseManager()
print(f"   db.db: {db.db}")
print(f"   type(db.db): {type(db.db)}")

if db.db:
    print(f"   db.milestones: {db.milestones}")
    direct_read = db.milestones.all()
    print(f"   db.milestones.all(): {len(direct_read)} records")
    
    via_method = db.get_all_milestones()
    print(f"   db.get_all_milestones(): {len(via_method)} records")
    
    if len(direct_read) != len(via_method):
        print(f"   ❌ MISMATCH! direct_read={len(direct_read)}, via_method={len(via_method)}")
    else:
        print(f"   ✅ Method working correctly")

# Test 5: Check folium
print("\n5️⃣ Checking Folium availability...")
from charity_tracker import FOLIUM_AVAILABLE
print(f"   FOLIUM_AVAILABLE: {FOLIUM_AVAILABLE}")

if not FOLIUM_AVAILABLE:
    print(f"   ⚠️  Folium not available - map won't render!")
    print(f"   Install with: pip install folium streamlit-folium")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

if db_file.exists():
    from tinydb import TinyDB
    db = TinyDB(str(db_file))
    m_count = len(db.table('milestones').all())
    d_count = len(db.table('donations').all())
    db.close()
    
    if m_count > 0:
        print(f"✅ Database file has {m_count} projects and {d_count} donations")
        print(f"✅ This data EXISTS and is readable")
        print(f"\n🔧 If Streamlit isn't showing it:")
        print(f"   1. Make sure Folium is installed: pip install folium streamlit-folium")
        print(f"   2. Check OTHER pages (Donate, Analytics) - they should show data")
        print(f"   3. If only Map is empty → Folium issue")
        print(f"   4. If ALL pages empty → Session state issue")
        print(f"\n📊 Try checking these pages:")
        print(f"   • Sidebar metric: Should show 'Projects: {m_count}'")
        print(f"   • Donate page: Should list {m_count} projects")
        print(f"   • Analytics page: Should show stats")
    else:
        print(f"❌ Database file exists but is EMPTY")
        print(f"\n🔧 Run: python emergency_force_load.py")
else:
    print(f"❌ No database file found")
    print(f"\n🔧 Run: python emergency_force_load.py")

print("=" * 70)
