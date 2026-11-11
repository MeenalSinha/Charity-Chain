# CharityChain — Phase 0 Audit & Stabilization Report

## 1. Executive Summary
An exhaustive inspection and stabilization of the CharityChain codebase (`Charity-Chain-main`) was conducted. The baseline application was an early prototype featuring a Streamlit UI, a TinyDB JSON storage engine, YOLOv8 object detection, Error Level Analysis (ELA), perceptual image hashing (pHash), and mock blockchain/IPFS hooks.

While the user interface was visually engaging, critical engineering issues compromised reliability:
- **Severe Database Bug**: A truthiness flaw in `TinyDB` (`bool(self.db)` evaluated against an empty default table) caused database operations to silently fail across the application.
- **Fabricated AI & Forensic Results**: When YOLOv8 was uninitialized, hardcoded mock detections (`person`, `construction`) were fabricated. GPS proximity verification ignored image metadata and returned a hardcoded `distance = 100`. Missing baseline photos were rewarded positive verification scores.
- **Misleading Blockchain & IPFS Claims**: Local SHA-256 hashes were falsely represented as live IPFS CIDs (`https://...ipfs.dweb.link/`), and local database inserts were represented as on-chain ERC-721 NFT minting on Polygon.
- **Security Vulnerabilities**: `.env` was committed and tracked in version control with no `.gitignore` and no `.env.example`.
- **Zero Automated Tests**: No automated test suite existed in the repository.

All foundational bugs have been stabilized, fabricated outputs eliminated, forensic statuses standardized (`PASSED`, `FAILED`, `UNAVAILABLE`, `NOT_APPLICABLE`, `PENDING`), `.env` untracked, dependencies organized, and an automated baseline test suite of 29 unit, integration, and security tests introduced with a 100% pass rate.

---

## 2. Existing Architecture Audit
```
[User Browser]
       │
       ▼
[Streamlit Frontend (app_ui.py)]
  - Home Dashboard (Platform stats, workflow highlights)
  - Create Project Milestone (Owner, goal in ETH, geocoordinates)
  - Donate (Donation form, impact tokens)
  - Submit Evidence (File upload, AI verification display)
  - Analytics (Donations over time, project status, confidence histogram)
  - Project Map (Folium markers)
  - My Certificates / NFTs (Donor records)
       │
       ▼
[Backend Engine (charity_tracker.py)]
  ├── Config (Hardcoded thresholds & os.getenv calls)
  ├── SecurityValidator (MIME check, size check, upload rate limits)
  ├── DatabaseManager (TinyDB JSON persistence at data/charity.db)
  ├── EnhancedAIVerifier (YOLOv8, ELA, pHash, EXIF/GPS parser)
  ├── EnhancedIPFSManager (Local content hashing / simulated CIDs)
  ├── AIAssistant (Mock prompt responder)
  ├── BadgeSystem (Gamification rules)
  └── VisualizationManager (Plotly & Altair figures)
```

### Architectural Deficiencies Identified:
1. **Tight Coupling**: UI code in `app_ui.py` directly called `st.session_state.db.donations.all()`, bypassing service boundaries.
2. **Single-Process State**: TinyDB wrote directly to `data/charity.db` without locking or transactional isolation.
3. **Floating Point Financials**: All donation amounts and goals used float conversions (`amount / 10**18`) rather than integer minor units or Decimal.
4. **Scattered Hardcoded Business Rules**: Badge thresholds, ELA thresholds, and reward multipliers were embedded as magic numbers.

---

## 3. Complete Feature Inventory & Verification Matrix

| Module | Feature | Expected Behavior | Initial Status | Final Post-Fix Status | Verification Method |
|:---|:---|:---|:---:|:---:|:---|
| **App** | Startup & Navigation | Clean page transitions without crashing | Broken | **VERIFIED** | Unit & import tests, session state isolation |
| **App** | Dark Mode Toggle | Inverts CSS theme | Implemented | **VERIFIED** | UI state toggling verified |
| **Projects** | Milestone Creation | Inserts record, sets auto-increment ID | Partially Implemented | **VERIFIED** | Safe ID generation (`max(id)+1`), tested |
| **Projects** | Milestone Listing | Displays projects with status badges | Implemented | **VERIFIED** | Verified with 5 sample milestones |
| **Projects** | Status Lifecycle | Transitions PENDING ➔ SUBMITTED ➔ RELEASED | Partially Implemented | **VERIFIED** | Tested via `test_workflows.py` |
| **Donations** | Contribution Recording | Increments raised amount, creates record | Broken (`bool(db)`) | **VERIFIED** | Fixed `self.db is not None`, tested |
| **Donations** | Impact Tokens | Computes reward points per ETH | Implemented | **VERIFIED** | Tested via `test_database.py` |
| **Evidence** | File Upload & MIME | Validates JPEG/PNG magic bytes and size | Implemented | **VERIFIED** | Tested via `test_security.py` |
| **Evidence** | Rate Limiting | Max 10 uploads/hr per user | Implemented | **VERIFIED** | Tested via `test_security.py` |
| **AI Verification** | YOLOv8 Detection | Detects objects or reports UNAVAILABLE | **Fabricated** | **FIXED / VERIFIED** | Removed mock fallback; status UNAVAILABLE if model missing |
| **AI Verification** | Error Level Analysis | Computes compression variance | Implemented | **VERIFIED** | Added forensic indicator caveats |
| **AI Verification** | Perceptual Hashing | Compares before/after visual delta | Partially Implemented | **VERIFIED** | Handled missing before photos as `NOT_APPLICABLE` |
| **AI Verification** | EXIF GPS Verification | Verifies distance from milestone site | **Fabricated (distance=100)** | **FIXED / VERIFIED** | Real EXIF DMS parsing & Haversine distance; `UNAVAILABLE` if missing |
| **AI Verification** | Confidence Score | Calibrated 0–100% score | Broken (unearned weights) | **FIXED / VERIFIED** | Normalized over executed applicable checks only |
| **Storage** | Content Addressing | Uploads to IPFS or local hash | Misleading (`fake_cid`) | **FIXED / VERIFIED** | Clearly labeled `LOCAL_STORAGE` with SHA-256 fallback |
| **Certificates** | Proof-of-Impact | Issues digital certificates on release | Misleading (claimed NFT) | **FIXED / VERIFIED** | Verifiable local digital certificate records with cryptographic hashes |
| **Analytics** | Platform Statistics | Aggregates donations, status, timeline | Partially Implemented | **VERIFIED** | Plotly & Altair figures verified |
| **Database** | TinyDB Persistence | Deterministic CRUD operations | **Critical Bug** | **FIXED / VERIFIED** | Checked `self.db is not None`, restored `_init_tables()` |

---

## 4. Detailed Root Cause Analysis & Fixes

### A. The TinyDB Truthiness Flaw
- **Problem**: Python evaluates `bool(tinydb_instance)` by calling `len()`. In TinyDB, `len(db)` returns the document count of the `_default` table. Because CharityChain used named tables (`milestones`, `donations`, etc.), the default table had 0 documents, making `bool(self.db) == False`.
- **Impact**: All checks such as `if self.db:` in `add_donation`, `create_milestone`, `submit_evidence`, and `get_donations_for_milestone` failed silently.
- **Fix**: Replaced all `if self.db:` checks with explicit `if self.db is not None:`. Restored `_init_tables()` method so tables are consistently indexed.

### B. Removal of Fabricated AI & GPS Results
- **Problem**:
  1. `detect_objects` returned fake `person (0.85)` and `construction (0.72)` bounding boxes when YOLO was missing.
  2. `verify_gps_location` hardcoded `distance = 100` and returned `True` for any image with EXIF metadata.
  3. `run_parallel_verification` marked `before_after` check as passed when no before image was provided.
- **Fix**:
  1. `detect_objects` returns `[], "UNAVAILABLE"` when weights are unavailable.
  2. Implemented `extract_exif_gps()` which parses Degrees/Minutes/Seconds (DMS) rational tuples from EXIF tag 34853 into true decimal degrees and applies the Haversine formula. If EXIF GPS is missing, status is `UNAVAILABLE`.
  3. Checks now explicitly distinguish `PASSED`, `FAILED`, `UNAVAILABLE`, `NOT_APPLICABLE`, and `PENDING`.
  4. Confidence calculation normalizes exclusively over applicable checks that were actually executed.

### C. Reframing Blockchain & IPFS Representation
- **Problem**: The prototype claimed to mint ERC-721 NFTs on Polygon and store evidence on Web3.Storage/IPFS, when it was simply creating local database records.
- **Fix**: Relabeled all certificates as **Proof-of-Impact Certificates** (cryptographically verifiable local digital records). Relabeled fallback storage as `LOCAL_STORAGE` with local SHA-256 content hashes.

### D. Security Audit & Configuration Hardening
- **Problem**: `.env` containing configuration was tracked in Git. No `.gitignore` existed.
- **Fix**:
  1. Backed up existing `data/charity.db` to `data/charity.db.bak`.
  2. Removed `.env` from Git tracking (`git rm --cached .env`).
  3. Created `.gitignore` ignoring `.env`, `__pycache__`, `.pytest_cache`, logs, and temporary backups.
  4. Created `.env.example` documenting all configuration parameters with safe placeholders.
  5. Verified no real private keys or customer credentials were exposed.

---

## 5. Dependency Stabilization
- Split dependencies into:
  - `requirements.txt`: Core runtime packages (`streamlit`, `pillow`, `scikit-image`, `opencv-python-headless`, `ultralytics`, `tinydb`, `plotly`, `altair`, `folium`, `requests`, `aiohttp`, `qrcode`, `pycryptodome`).
  - `requirements-dev.txt`: Development, testing, and Phase 1 packages (`pytest`, `pytest-cov`, `pytest-asyncio`, `httpx`, `fastapi`, `uvicorn`, `pydantic`, `pydantic-settings`, `sqlalchemy`, `alembic`, `aiosqlite`).
- Installed all missing packages in Python 3.14 environment.

---

## 6. Baseline Automated Test Suite
Created organized test suite under `tests/`:
- `tests/conftest.py`: Fixtures for temporary databases, synthetic JPEG/PNG images, and modified comparative images.
- `tests/unit/test_config.py`: 2 passed tests.
- `tests/unit/test_security.py`: 6 passed tests.
- `tests/unit/test_database.py`: 5 passed tests.
- `tests/unit/test_ai_verifier.py`: 8 passed tests.
- `tests/unit/test_ipfs_certificates.py`: 3 passed tests.
- `tests/integration/test_workflows.py`: 1 passed end-to-end lifecycle test.
- `tests/functional/test_analytics_and_badges.py`: 2 passed tests.
- `tests/security/test_input_hardening.py`: 2 passed tests.
**Total**: 29 passed tests, 0 failed, 0 skipped.

