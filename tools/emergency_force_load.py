#!/usr/bin/env python3
"""
EMERGENCY FIX: Force database population
This bypasses all checks and directly inserts data
"""

from pathlib import Path
from tinydb import TinyDB
from datetime import datetime

print("🚨 EMERGENCY DATABASE POPULATION")
print("=" * 60)

# Delete old database
db_file = Path('data/charity.db')
if db_file.exists():
    print("🗑️  Deleting old database...")
    db_file.unlink()

# Create data directory if needed
Path('data').mkdir(exist_ok=True)

# Create fresh database directly
print("📦 Creating fresh database...")
db = TinyDB('data/charity.db')

# Drop and recreate tables
print("🔄 Initializing tables...")
db.drop_tables()
milestones_table = db.table('milestones')
donations_table = db.table('donations')
verifications_table = db.table('verifications')
ngo_profiles_table = db.table('ngo_profiles')
nfts_table = db.table('nfts')
analytics_table = db.table('analytics')

print("📝 Inserting 5 projects...")

# Project data
projects = [
    {
        'id': 1,
        'owner': 'Green Earth Foundation',
        'goal': int(2.5 * 10**18),
        'raised': int(1.0 * 10**18),
        'status': 'PENDING',
        'description': 'Plant 1000 trees in Amazon rainforest to combat deforestation',
        'latitude': -3.4653,
        'longitude': -62.2159,
        'created_at': datetime.now().isoformat(),
        'evidence_cid': None,
        'verification_score': None
    },
    {
        'id': 2,
        'owner': 'Clean Water Initiative',
        'goal': int(5.0 * 10**18),
        'raised': int(3.5 * 10**18),
        'status': 'PENDING',
        'description': 'Build 10 wells in rural Kenya for 5000 people',
        'latitude': -1.2921,
        'longitude': 36.8219,
        'created_at': datetime.now().isoformat(),
        'evidence_cid': None,
        'verification_score': None
    },
    {
        'id': 3,
        'owner': 'Education for All',
        'goal': int(3.0 * 10**18),
        'raised': int(0.8 * 10**18),
        'status': 'PENDING',
        'description': 'Provide books for 500 underprivileged children in Delhi',
        'latitude': 28.6139,
        'longitude': 77.2090,
        'created_at': datetime.now().isoformat(),
        'evidence_cid': None,
        'verification_score': None
    },
    {
        'id': 4,
        'owner': 'Ocean Cleanup Project',
        'goal': int(4.0 * 10**18),
        'raised': 0,
        'status': 'PENDING',
        'description': 'Remove 2 tons of plastic waste from Pacific coastal areas',
        'latitude': 21.3099,
        'longitude': -157.8581,
        'created_at': datetime.now().isoformat(),
        'evidence_cid': None,
        'verification_score': None
    },
    {
        'id': 5,
        'owner': 'Medical Aid Africa',
        'goal': int(6.0 * 10**18),
        'raised': 0,
        'status': 'PENDING',
        'description': 'Provide medical supplies for 3 rural clinics in Tanzania',
        'latitude': -6.7924,
        'longitude': 39.2083,
        'created_at': datetime.now().isoformat(),
        'evidence_cid': None,
        'verification_score': None
    }
]

# Insert projects
for project in projects:
    milestones_table.insert(project)
    print(f"   ✅ {project['owner']}")
    
    # Create NGO profile
    ngo_profiles_table.insert({
        'address': project['owner'],
        'reputation_score': 100,
        'completed_milestones': 0,
        'rejected_milestones': 0,
        'total_raised': 0
    })

print("\n💰 Inserting 6 donations...")

# Donation data
donations = [
    {'milestone_id': 1, 'donor': 'Alice Donor', 'amount': int(0.5 * 10**18), 'timestamp': datetime.now().isoformat()},
    {'milestone_id': 1, 'donor': 'Bob Supporter', 'amount': int(0.3 * 10**18), 'timestamp': datetime.now().isoformat()},
    {'milestone_id': 1, 'donor': 'Carol Benefactor', 'amount': int(0.2 * 10**18), 'timestamp': datetime.now().isoformat()},
    {'milestone_id': 2, 'donor': 'David Philanthropist', 'amount': int(2.0 * 10**18), 'timestamp': datetime.now().isoformat()},
    {'milestone_id': 2, 'donor': 'Eva Contributor', 'amount': int(1.5 * 10**18), 'timestamp': datetime.now().isoformat()},
    {'milestone_id': 3, 'donor': 'Frank Supporter', 'amount': int(0.8 * 10**18), 'timestamp': datetime.now().isoformat()},
]

for donation in donations:
    donations_table.insert(donation)
    print(f"   ✅ {donation['donor']} → Project #{donation['milestone_id']}")

# Close database
db.close()

print("\n✅ VERIFICATION:")
# Reopen and verify
verify_db = TinyDB('data/charity.db')
milestones = verify_db.table('milestones').all()
donations = verify_db.table('donations').all()
verify_db.close()

total_raised = sum(d['amount'] for d in donations) / 10**18

print(f"   Projects: {len(milestones)}")
print(f"   Donations: {len(donations)}")
print(f"   Total raised: {total_raised:.2f} ETH")

if len(milestones) == 5 and len(donations) == 6:
    print("\n" + "=" * 60)
    print("🎉 SUCCESS! Database populated!")
    print("=" * 60)
    print("\n✅ You can now run:")
    print("   streamlit run app_ui.py")
    print("\n✅ Expected results:")
    print("   • Sidebar: 'Projects: 5'")
    print("   • Map: 5 colored markers")
    print("   • Donate: 5 projects with progress bars")
    print("   • Analytics: Populated charts")
else:
    print("\n⚠️  Verification failed!")
    print(f"   Expected: 5 projects, 6 donations")
    print(f"   Got: {len(milestones)} projects, {len(donations)} donations")
