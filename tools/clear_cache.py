#!/usr/bin/env python3
"""
Clear Streamlit cache and force fresh start
"""
import shutil
from pathlib import Path

print("🧹 CLEARING STREAMLIT CACHE")
print("=" * 60)

# Clear .streamlit cache directory
cache_dir = Path('.streamlit/cache')
if cache_dir.exists():
    shutil.rmtree(cache_dir)
    print("✅ Deleted .streamlit/cache directory")
else:
    print("✓ No cache directory found")

# Clear __pycache__
pycache_dirs = list(Path('.').glob('**/__pycache__'))
for pdir in pycache_dirs:
    shutil.rmtree(pdir)
    print(f"✅ Deleted {pdir}")

if not pycache_dirs:
    print("✓ No __pycache__ found")

print("\n" + "=" * 60)
print("✅ CACHE CLEARED!")
print("\n🚀 Now run:")
print("   streamlit run app_ui.py")
print("\nℹ️  Streamlit will start fresh and load the 5 projects!")
print("=" * 60)
