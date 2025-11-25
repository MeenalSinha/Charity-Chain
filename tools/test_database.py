#!/usr/bin/env python3
"""
Direct test of DatabaseManager reading charity.db
"""
import sys
sys.path.insert(0, '.')

print("🔍 TESTING DatabaseManager")
print("=" * 60)

# Test 1: Check if file exists
from pathlib import Path
db_file = Path('data/charity.db')
print(f"\n1. Database file:")
print(f"   Path: {db_file}")
print(f"   Exists: {db_file.exists()}")
if db_file.exists():
    print(f"   Size: {db_file.stat().st_size} bytes")

# Test 2: Import and check DB_AVAILABLE
print(f"\n2. Checking imports:")
try:
    from charity_tracker import DatabaseManager, DB_AVAILABLE, Config
    print(f"   ✅ Imports successful")
    print(f"   DB_AVAILABLE: {DB_AVAILABLE}")
    print(f"   Config.DB_FILE: {Config.DB_FILE}")
except ImportError as e:
    print(f"   ❌ Import error: {e}")
    sys.exit(1)

# Test 3: Create DatabaseManager
print(f"\n3. Creating DatabaseManager:")
db = DatabaseManager()
print(f"   db created: {db}")
print(f"   db.db: {db.db}")
print(f"   db.db is None: {db.db is None}")

if db.db is None:
    print(f"   ❌ db.db is None!")
    print(f"   DB_AVAILABLE is: {DB_AVAILABLE}")
    print(f"   This means TinyDB didn't initialize!")
    sys.exit(1)

# Test 4: Check tables
print(f"\n4. Checking tables:")
print(f"   db.milestones: {db.milestones}")
print(f"   db.donations: {db.donations}")

# Test 5: Read milestones
print(f"\n5. Reading milestones:")
try:
    milestones = db.get_all_milestones()
    print(f"   get_all_milestones() returned: {type(milestones)}")
    print(f"   Length: {len(milestones)}")
    
    if len(milestones) == 0:
        print(f"   ❌ NO MILESTONES FOUND!")
        print(f"\n   Trying direct read:")
        direct = db.milestones.all()
        print(f"   db.milestones.all(): {len(direct)} items")
        
        if len(direct) > 0:
            print(f"   ✅ Direct read found {len(direct)} items!")
            print(f"   Problem is in get_all_milestones() method!")
        else:
            print(f"   ❌ Even direct read found nothing!")
            print(f"   TinyDB is not reading the file correctly!")
    else:
        print(f"   ✅ Found {len(milestones)} milestones!")
        for m in milestones[:2]:
            print(f"      • {m['description'][:50]}...")
            
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 6: Check database path
print(f"\n6. Verifying database path:")
print(f"   Config.DB_FILE: {Config.DB_FILE}")
print(f"   Absolute: {Config.DB_FILE.absolute()}")
print(f"   Exists: {Config.DB_FILE.exists()}")

# Test 7: Try opening with TinyDB directly
print(f"\n7. Direct TinyDB test:")
try:
    from tinydb import TinyDB
    test_db = TinyDB('data/charity.db')
    test_milestones = test_db.table('milestones').all()
    test_db.close()
    print(f"   Direct TinyDB read: {len(test_milestones)} milestones")
    
    if len(test_milestones) > 0:
        print(f"   ✅ TinyDB CAN read the file!")
        print(f"   Problem is in DatabaseManager!")
    else:
        print(f"   ❌ TinyDB also reads 0 milestones!")
        print(f"   File might be corrupted or wrong format!")
except Exception as e:
    print(f"   ❌ Error: {e}")

print(f"\n" + "=" * 60)
print(f"DIAGNOSIS:")
print(f"=" * 60)

if db.db and len(milestones) == 5:
    print(f"✅ Everything working! DatabaseManager reads 5 milestones.")
elif db.db and len(milestones) == 0:
    print(f"❌ DatabaseManager created but returns 0 milestones!")
    print(f"   Issue: TinyDB not reading file or file is empty")
    print(f"   Solution: Run emergency_force_load.py again")
else:
    print(f"❌ DatabaseManager not initialized!")
    print(f"   Issue: DB_AVAILABLE is False or TinyDB import failed")
    print(f"   Solution: pip install tinydb")
