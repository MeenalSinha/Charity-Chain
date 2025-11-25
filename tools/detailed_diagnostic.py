#!/usr/bin/env python3
"""
Detailed diagnostic to find why data isn't saving
"""

print("🔍 Detailed Database Diagnostic")
print("=" * 60)

# Step 1: Check imports
print("\n1. Checking imports...")
try:
    from charity_tracker import DatabaseManager, DB_AVAILABLE, Config
    from tinydb import TinyDB
    print(f"   ✅ Imports successful")
    print(f"   DB_AVAILABLE: {DB_AVAILABLE}")
except ImportError as e:
    print(f"   ❌ Import error: {e}")
    exit(1)

if not DB_AVAILABLE:
    print("   ❌ TinyDB not available!")
    print("   Run: pip install tinydb")
    exit(1)

# Step 2: Check database file path
print("\n2. Checking database configuration...")
print(f"   DB_FILE path: {Config.DB_FILE}")
print(f"   DB_FILE exists: {Config.DB_FILE.exists()}")

# Step 3: Create DatabaseManager and inspect
print("\n3. Creating DatabaseManager...")
db = DatabaseManager()
print(f"   db object created: {db}")
print(f"   db.db object: {db.db}")
print(f"   db.db is None: {db.db is None}")

if db.db is None:
    print("   ❌ db.db is None - database not initialized!")
    exit(1)

# Step 4: Check table initialization
print("\n4. Checking tables...")
print(f"   milestones table: {db.milestones}")
print(f"   donations table: {db.donations}")

# Step 5: Try direct TinyDB test
print("\n5. Direct TinyDB test...")
try:
    test_db = TinyDB('data/test_direct.db')
    test_table = test_db.table('test')
    test_table.insert({'test': 'data', 'id': 1})
    result = test_table.all()
    print(f"   ✅ Direct TinyDB works: {len(result)} record(s)")
    test_db.close()
except Exception as e:
    print(f"   ❌ Direct TinyDB failed: {e}")

# Step 6: Try milestone creation with detailed logging
print("\n6. Testing milestone creation...")
print("   Calling create_milestone...")

try:
    # Get initial count
    initial = db.get_all_milestones()
    print(f"   Initial milestones: {len(initial)}")
    
    # Create a milestone
    milestone_id = db.create_milestone(
        "Test NGO",
        int(1.0 * 10**18),
        "Test milestone",
        40.7128,
        -74.0060
    )
    print(f"   ✅ create_milestone returned ID: {milestone_id}")
    
    # Check if it was actually inserted
    print("\n7. Verifying insertion...")
    all_milestones = db.get_all_milestones()
    print(f"   Milestones after insert: {len(all_milestones)}")
    
    if len(all_milestones) > len(initial):
        print("   ✅ Milestone was inserted!")
        print(f"   Data: {all_milestones[-1]}")
    else:
        print("   ❌ Milestone NOT inserted!")
        print("   Checking if db.db is set...")
        print(f"   db.db: {db.db}")
        print(f"   db.db type: {type(db.db)}")
        
        # Check the actual file
        from pathlib import Path
        db_file = Path('data/charity.db')
        if db_file.exists():
            print(f"   DB file exists, size: {db_file.stat().st_size} bytes")
            
            # Try reading it directly
            direct_db = TinyDB(str(db_file))
            direct_milestones = direct_db.table('milestones').all()
            print(f"   Direct read shows: {len(direct_milestones)} milestone(s)")
            direct_db.close()
        
except Exception as e:
    print(f"   ❌ Error during milestone creation: {e}")
    import traceback
    traceback.print_exc()

# Step 8: Check get_all_milestones implementation
print("\n8. Checking get_all_milestones method...")
try:
    result = db.get_all_milestones()
    print(f"   Result type: {type(result)}")
    print(f"   Result length: {len(result)}")
    print(f"   Result: {result}")
except Exception as e:
    print(f"   ❌ Error: {e}")

print("\n" + "=" * 60)
print("DIAGNOSIS COMPLETE")
print("=" * 60)
