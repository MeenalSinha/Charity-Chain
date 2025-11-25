#!/usr/bin/env python3
"""
Diagnostic script to test demo data generation
Run this BEFORE launching the Streamlit app
"""

import sys
from pathlib import Path

print("=" * 60)
print("CharityChain Demo Data Diagnostic")
print("=" * 60)

# Check imports
print("\n1. Checking imports...")
try:
    from charity_tracker import (
        DB_AVAILABLE, DatabaseManager, Logger, 
        generate_demo_data, STREAMLIT_AVAILABLE
    )
    print("   ✅ All imports successful")
except ImportError as e:
    print(f"   ❌ Import failed: {e}")
    sys.exit(1)

# Check TinyDB
print("\n2. Checking TinyDB...")
print(f"   DB_AVAILABLE: {DB_AVAILABLE}")
if not DB_AVAILABLE:
    print("   ❌ TinyDB not available - install with: pip install tinydb")
    sys.exit(1)

# Check database file
print("\n3. Checking database file...")
db_file = Path('data/charity.db')
if db_file.exists():
    print(f"   ⚠️  DB file exists: {db_file}")
    print(f"   Size: {db_file.stat().st_size} bytes")
else:
    print(f"   ✓ DB file doesn't exist (will be created)")

# Create a test database instance
print("\n4. Testing DatabaseManager...")
try:
    db = DatabaseManager()
    milestones = db.get_all_milestones()
    print(f"   ✅ DatabaseManager created")
    print(f"   Current milestones: {len(milestones)}")
except Exception as e:
    print(f"   ❌ Failed to create DatabaseManager: {e}")
    sys.exit(1)

# Test demo data generation WITHOUT Streamlit
print("\n5. Testing demo data generation...")
print("   NOTE: This requires Streamlit session state")
print("   We'll do a dry run to check the function exists")

# Check if function is callable
if callable(generate_demo_data):
    print("   ✅ generate_demo_data function exists")
else:
    print("   ❌ generate_demo_data is not callable")
    sys.exit(1)

# Manual insertion test
print("\n6. Manual demo data insertion test...")
print("   Inserting test project...")
try:
    test_id = db.create_milestone(
        "Test NGO",
        int(1.0 * 10**18),
        "Test project for diagnostics",
        40.7128,
        -74.0060
    )
    print(f"   ✅ Test milestone created: ID={test_id}")
    
    # Verify it was created
    milestones = db.get_all_milestones()
    print(f"   ✅ Verification: {len(milestones)} milestone(s) in DB")
    
    if len(milestones) > 0:
        print(f"   ✅ Last milestone: {milestones[-1]['description']}")
except Exception as e:
    print(f"   ❌ Failed to create milestone: {e}")
    import traceback
    traceback.print_exc()

# Check if we can read it back
print("\n7. Testing data persistence...")
try:
    db2 = DatabaseManager()  # Fresh instance
    milestones2 = db2.get_all_milestones()
    print(f"   ✅ Fresh instance sees {len(milestones2)} milestone(s)")
    if len(milestones2) == len(milestones):
        print("   ✅ Data persists across instances")
    else:
        print("   ⚠️  Data count mismatch - possible issue")
except Exception as e:
    print(f"   ❌ Failed to verify persistence: {e}")

# Summary
print("\n" + "=" * 60)
print("DIAGNOSTIC SUMMARY")
print("=" * 60)

if len(milestones) > 0:
    print("✅ Database is working!")
    print(f"   Found {len(milestones)} milestone(s)")
    print("\n💡 To clear and start fresh:")
    print("   1. Delete: data/charity.db")
    print("   2. Run: streamlit run app_ui.py")
    print("   3. Demo data should auto-load with force=True")
else:
    print("⚠️  No milestones found")
    print("\n🔧 Next steps:")
    print("   1. Delete data/charity.db if it exists")
    print("   2. Launch Streamlit app")
    print("   3. Check if generate_demo_data(force=True) runs")

print("\n" + "=" * 60)
