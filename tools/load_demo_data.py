#!/usr/bin/env python3
"""
Standalone Demo Data Loader
Run this to populate the database before launching Streamlit
"""

import sys
from pathlib import Path

print("🚀 CharityChain - Standalone Demo Data Loader")
print("=" * 60)

# Import
try:
    from charity_tracker import DatabaseManager, Logger, Config
    print("✅ Imports successful")
except ImportError as e:
    print(f"❌ Import failed: {e}")
    sys.exit(1)

# Check/delete existing DB
db_file = Path('data/charity.db')
if db_file.exists():
    print(f"⚠️  Existing DB found: {db_file}")
    response = input("Delete and recreate? (y/n): ")
    if response.lower() == 'y':
        db_file.unlink()
        print("✅ Old DB deleted")
    else:
        print("❌ Keeping existing DB - may have conflicts")

# Create fresh database
print("\n📦 Creating DatabaseManager...")
db = DatabaseManager()

# Force reset
if db.db:
    print("🔄 Dropping all tables...")
    db.db.drop_tables()
    print("🔄 Reinitializing tables...")
    db._init_tables()

print("\n📝 Inserting demo data...")

demo_projects = [
    {
        'owner': 'Green Earth Foundation',
        'goal': int(2.5 * 10**18),
        'description': 'Plant 1000 trees in Amazon rainforest to combat deforestation',
        'latitude': -3.4653,
        'longitude': -62.2159
    },
    {
        'owner': 'Clean Water Initiative',
        'goal': int(5.0 * 10**18),
        'description': 'Build 10 wells in rural Kenya for 5000 people',
        'latitude': -1.2921,
        'longitude': 36.8219
    },
    {
        'owner': 'Education for All',
        'goal': int(3.0 * 10**18),
        'description': 'Provide books for 500 underprivileged children in Delhi',
        'latitude': 28.6139,
        'longitude': 77.2090
    },
    {
        'owner': 'Ocean Cleanup Project',
        'goal': int(4.0 * 10**18),
        'description': 'Remove 2 tons of plastic waste from Pacific coastal areas',
        'latitude': 21.3099,
        'longitude': -157.8581
    },
    {
        'owner': 'Medical Aid Africa',
        'goal': int(6.0 * 10**18),
        'description': 'Provide medical supplies for 3 rural clinics in Tanzania',
        'latitude': -6.7924,
        'longitude': 39.2083
    }
]

# Insert projects
for i, project in enumerate(demo_projects, 1):
    milestone_id = db.create_milestone(
        project['owner'],
        project['goal'],
        project['description'],
        project['latitude'],
        project['longitude']
    )
    print(f"   ✅ Project {i}: {project['owner']}")
    
    # Add donations to first 3 projects
    if i == 1:
        db.add_donation(milestone_id, "Alice Donor", int(0.5 * 10**18))
        db.add_donation(milestone_id, "Bob Supporter", int(0.3 * 10**18))
        db.add_donation(milestone_id, "Carol Benefactor", int(0.2 * 10**18))
        print(f"      └─ Added 3 donations (1.0 ETH)")
    elif i == 2:
        db.add_donation(milestone_id, "David Philanthropist", int(2.0 * 10**18))
        db.add_donation(milestone_id, "Eva Contributor", int(1.5 * 10**18))
        print(f"      └─ Added 2 donations (3.5 ETH)")
    elif i == 3:
        db.add_donation(milestone_id, "Frank Supporter", int(0.8 * 10**18))
        print(f"      └─ Added 1 donation (0.8 ETH)")

# Verify
print("\n✅ Verification:")
milestones = db.get_all_milestones()
donations = db.donations.all() if db.db else []
total_raised = sum(d['amount'] for d in donations) / 10**18

print(f"   Projects created: {len(milestones)}")
print(f"   Donations added: {len(donations)}")
print(f"   Total raised: {total_raised:.2f} ETH")

print("\n" + "=" * 60)
print("🎉 Demo data loaded successfully!")
print("\nYou can now run:")
print("   streamlit run app_ui.py")
print("\nExpected results:")
print("   ✅ Sidebar: 'Projects: 5'")
print("   ✅ Map: 5 markers")
print("   ✅ Donate: 5 projects with progress bars")
print("=" * 60)
