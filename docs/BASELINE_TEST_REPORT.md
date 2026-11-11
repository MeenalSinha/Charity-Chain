# CharityChain — Baseline Test Report (Phase 0)

## 1. Test Suite Summary
- **Execution Command**: `pytest -v tests/`
- **Configuration**: `pytest.ini` (`testpaths = tests`, `filterwarnings = ignore::DeprecationWarning`)
- **Total Tests Collected**: 29
- **Passed**: 29
- **Failed**: 0
- **Skipped**: 0
- **Execution Time**: 4.04 seconds
- **Test Date**: 2026-11-11
- **Python Version**: Python 3.14.2 (Windows 11)

---

## 2. Test Execution Breakdown

### Functional & Analytics Tests (`tests/functional/`)
| Test Case | Description | Result | Duration |
|:---|:---|:---:|:---:|
| `test_analytics_and_badges.py::test_badge_progression_system` | Verifies dynamic unlocking of 'early_supporter' and 'top_donor' badges based on verified donations. | **PASSED** | 0.05s |
| `test_analytics_and_badges.py::test_visualization_figures` | Validates generation of confidence gauges and breakdown polar charts across multiple check statuses. | **PASSED** | 0.08s |

### Integration Tests (`tests/integration/`)
| Test Case | Description | Result | Duration |
|:---|:---|:---:|:---:|
| `test_workflows.py::test_full_project_milestone_lifecycle` | End-to-end integration: Milestone creation ➔ 2 donations ➔ Content hashing ➔ AI verification ➔ Evidence submission ➔ Reviewer RELEASE status ➔ Proof-of-Impact certificate generation. | **PASSED** | 0.45s |

### Security & Hardening Tests (`tests/security/`)
| Test Case | Description | Result | Duration |
|:---|:---|:---:|:---:|
| `test_input_hardening.py::test_corrupted_file_handling` | Asserts graceful failure handling when parsing corrupted/truncated image bytes across ELA, pHash, and EXIF modules. | **PASSED** | 0.04s |
| `test_input_hardening.py::test_mime_validator_rejects_executable_and_scripts` | Validates rejection of Windows PE executables and script payloads by magic bytes check. | **PASSED** | 0.01s |

### Unit Tests — AI & Computer Vision (`tests/unit/test_ai_verifier.py`)
| Test Case | Description | Result | Duration |
|:---|:---|:---:|:---:|
| `test_error_level_analysis` | Verifies ELA calculation and indicator threshold comparison on synthetic JPEG. | **PASSED** | 0.03s |
| `test_perceptual_hash_similarity` | Verifies 0 distance for identical images and positive distance for modified images. | **PASSED** | 0.04s |
| `test_gps_distance_haversine` | Computes accurate Haversine distance across geocoordinates (NYC to London ~ 5,570 km). | **PASSED** | 0.01s |
| `test_verify_gps_without_exif_returns_unavailable` | Asserts that missing EXIF returns status `UNAVAILABLE` rather than hardcoded mock distance. | **PASSED** | 0.01s |
| `test_dms_conversion` | Validates rational DMS degrees/minutes/seconds conversion to decimal degrees. | **PASSED** | 0.01s |
| `test_detect_objects_no_fake_fallback` | Validates that missing YOLO model weights never return fabricated detections. | **PASSED** | 0.01s |
| `test_parallel_verification_structured_statuses` | Validates structured status enumeration (`PASSED`, `FAILED`, `UNAVAILABLE`, `NOT_APPLICABLE`). | **PASSED** | 0.12s |
| `test_parallel_verification_not_applicable_checks` | Validates that omitted before images receive `NOT_APPLICABLE` and do not inflate confidence score. | **PASSED** | 0.06s |

### Unit Tests — Configuration (`tests/unit/test_config.py`)
| Test Case | Description | Result | Duration |
|:---|:---|:---:|:---:|
| `test_config_defaults` | Validates positive thresholds and MIME allowlists. | **PASSED** | 0.01s |
| `test_mask_sensitive` | Validates credential masking without leaking secret bytes. | **PASSED** | 0.01s |

### Unit Tests — Database (`tests/unit/test_database.py`)
| Test Case | Description | Result | Duration |
|:---|:---|:---:|:---:|
| `test_database_initialization` | Verifies deterministic initialization of tables on empty database. | **PASSED** | 0.02s |
| `test_create_and_retrieve_milestone` | Verifies collision-safe milestone ID generation and auto-creation of NGO reputation profile. | **PASSED** | 0.02s |
| `test_add_donations_and_raised_update` | Verifies atomic update of raised funds when donations are added. | **PASSED** | 0.02s |
| `test_evidence_submission_and_analytics` | Verifies persistence of verification reports and analytics summaries. | **PASSED** | 0.02s |
| `test_milestone_release_and_certificate_issuance` | Verifies issuance of verifiable digital certificates when a milestone is released. | **PASSED** | 0.03s |

### Unit Tests — Storage & Certificates (`tests/unit/test_ipfs_certificates.py`)
| Test Case | Description | Result | Duration |
|:---|:---|:---:|:---:|
| `test_ipfs_manager_local_storage_fallback` | Validates fallback to local SHA-256 content address when IPFS token is unconfigured. | **PASSED** | 0.01s |
| `test_qr_code_generation` | Validates PNG QR code generation from content identifier. | **PASSED** | 0.02s |
| `test_ai_assistant_unconfigured_mode` | Asserts that unconfigured assistant returns rule-based status with explicit notice. | **PASSED** | 0.01s |

### Unit Tests — Security (`tests/unit/test_security.py`)
| Test Case | Description | Result | Duration |
|:---|:---|:---:|:---:|
| `test_validate_mime_type_jpeg` | Verifies JPEG magic byte detection (`\xff\xd8`). | **PASSED** | 0.01s |
| `test_validate_mime_type_png` | Verifies PNG magic byte detection (`\x89PNG`). | **PASSED** | 0.01s |
| `test_validate_mime_type_invalid` | Rejects non-image bytes. | **PASSED** | 0.01s |
| `test_validate_file_size_ok` | Permits images under 10MB limit. | **PASSED** | 0.01s |
| `test_validate_file_size_exceeded` | Rejects files exceeding 10MB limit. | **PASSED** | 0.01s |
| `test_rate_limiting` | Validates user upload frequency limits within sliding 1-hour window. | **PASSED** | 0.02s |

---

## 3. Environment Limitations
1. **Live IPFS / Pinata Gateway**: Not connected in default local test suite; mocked by local content-addressing (SHA-256).
2. **Live PayPal API**: Scheduled for Phase 4.
3. **YOLO Model Inference**: Requires `yolov8n.pt` weight file in root; when running in minimal environments without weights, status gracefully degrades to `UNAVAILABLE` without test failure.

