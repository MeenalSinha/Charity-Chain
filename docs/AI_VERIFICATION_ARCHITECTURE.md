# AI Evidence Verification Architecture — CharityChain Phase 6

## 1. Overview and Design Principles

CharityChain Phase 6 introduces an authentic, multi-indicator AI Verification Engine designed to analyze photographic, geospatial, and document evidence submitted by NGOs to demonstrate real-world milestone completion.

Rather than relying on ungrounded generative predictions or synthetic confidence scores, CharityChain adopts a defense-in-depth, forensic architecture where multiple independent analyzers evaluate distinct dimensions of authenticity, physical progression, and domain relevance.

### Core Architectural Principles

1. **Advisory Role Only**: The AI Verification Engine acts strictly as a decision-support and forensic analysis tool for human reviewers. Under no circumstances does an AI verification score trigger automated fund disbursements, milestone approvals, or escrow releases (which are deferred to Phase 7 and Phase 8).
2. **Multi-Indicator Independence**: Every verification check executes as an isolated analyzer (Object Detection, ELA, pHash, SSIM, EXIF, GPS, PDF, and Grounded XAI) returning structured metrics, typed statuses (`PASSED`, `FLAGGED`, `FAILED`, `NOT_APPLICABLE`), and execution timings.
3. **Reproducibility & Pinned Weights**: All model artifacts (e.g. YOLOv8n weights) are pinned to immutable SHA-256 hashes (`f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`) and verified before loading.
4. **Honest Calibration Reporting**: Every verification report explicitly records `calibration_status = "UNVALIDATED"`. CharityChain refuses to claim calibrated statistical probabilities or true Bayes probabilities in the absence of an externally validated empirical test dataset.
5. **Cryptographic Chaining**: Every verification run is immutably appended to the evidence asset's cryptographic audit hash chain (`AI_VERIFIED` event).
6. **Dual Execution Engine (Decoupled Background & Synchronous)**: Fast in-memory synchronous execution for unit evaluation and durable asynchronous background job processing (`VerificationJob`) with automatic retries and state tracking.

---

## 2. Verification Pipeline Architecture

The overall verification flow is depicted below:

```
[Uploaded Evidence Asset (Storage Key)]
                  │
                  ▼
   ┌──────────────────────────────┐
   │    VerificationService       │
   │  - Validates RBAC & State    │
   │  - Enqueues VerificationJob  │
   └──────────────┬───────────────┘
                  │
                  ▼
   ┌──────────────────────────────┐
   │  MultiIndicatorAIProvider    │
   └──────────────┬───────────────┘
                  │
  ┌───────────────┼──────────────────────────────┐
  │               │                              │
  ▼               ▼                              ▼
┌──────────────┐┌──────────────┐┌──────────────┐┌──────────────┐
│ FileIntegrity││ YOLOv8 Model ││ ELA Analyzer ││ pHash / SSIM │
│ SHA-256 Match││ Relevance Map││ Resaved JPEG ││ Image Percept│
└──────────────┘└──────────────┘└──────────────┘└──────────────┘
  │               │                              │
  ▼               ▼                              ▼
┌──────────────┐┌──────────────┐┌──────────────┐┌──────────────┐
│ EXIF Header  ││ GPS Haversine││ PDF Bounded  ││ Grounded XAI │
│ Timestamp/Cam││ Dist to Site ││ Page Extractor││ Explainer    │
└──────────────┘└──────────────┘└──────────────┘└──────────────┘
                  │
                  ▼
   ┌──────────────────────────────┐
   │   Conservative Evaluator     │
   │  - Overall Assessment Enum   │
   │  - Weighted Confidence Calc  │
   │  - Diagnostic Artifact Gen   │
   └──────────────┬───────────────┘
                  │
                  ▼
   ┌──────────────────────────────┐
   │   Database Persistence       │
   │  - VerificationReport        │
   │  - VerificationCheckRecord[] │
   │  - EvidenceAuditEvent Chain  │
   └──────────────────────────────┘
```

---

## 3. Analyzer Modules Specification

### 3.1 File Integrity Check (`FILE_INTEGRITY`)
- **Purpose**: Verifies that the binary bytes retrieved from storage match the immutable cryptographic hash recorded in `evidence_files.file_hash`.
- **Implementation**: Computes streaming SHA-256 over raw input bytes.
- **Fail Condition**: Any byte alteration immediately yields `FAILED` and halts further processing.

### 3.2 Object Detection & Category Relevance (`OBJECT_DETECTION`)
- **Module**: `backend/app/integrations/ai/yolo_provider.py`
- **Model**: Ultralytics YOLOv8n (nano) with lazy model instantiation.
- **Weights Integrity**: Checks local or downloaded `yolov8n.pt` against SHA-256 hash `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`.
- **Relevance Mapping**: Maps project categories (`WATER_SANITATION`, `EDUCATION`, `ENVIRONMENT`, `HEALTHCARE`, `SHELTER_HOUSING`, `RENEWABLE_ENERGY`, `COMMUNITY_DEVELOPMENT`) to expected COCO classes.
- **Relevance Ratio**: Computes the ratio of category-relevant detections to total detections. Flags images where detections fail to match the charitable project domain.

### 3.3 Error Level Analysis (`ERROR_LEVEL_ANALYSIS`)
- **Module**: `backend/app/integrations/ai/ela_analyzer.py`
- **Methodology**: Saves original image at 90% quality, extracts absolute pixel difference, amplifies contrast by scale factor 10, and calculates mean and maximum difference per channel.
- **Format Handling**: Only applicable to JPEGs. PNG and PDF return `NOT_APPLICABLE` with honest limitation notes.
- **Threshold**: Mean error exceeding `settings.TAMPER_ELA_THRESHOLD` (default: 15.0) yields `FLAGGED`.

### 3.4 Perceptual Hashing & SSIM Progress Comparison (`PROGRESS_COMPARISON`)
- **Modules**: `phash_analyzer.py`, `ssim_analyzer.py`, `progress_comparator.py`
- **pHash**: 32x32 grayscale DCT hash producing 64-bit fingerprint; compares via Hamming distance.
- **SSIM**: Multichannel structural similarity index normalized to equal dimensions.
- **Duplicate Detection**: Hamming distance = 0 or SSIM >= 0.98 flags possible reuse of identical photographs without actual progression.
- **Progression Detection**: Hamming distance 2–25 and SSIM 0.40–0.85 indicates genuine physical site progress.

### 3.5 EXIF & GPS Geospatial Proximity (`EXIF_METADATA` & `GPS_GEOSPATIAL`)
- **Modules**: `exif_analyzer.py`, `gps_analyzer.py`
- **EXIF Extraction**: Extracts Camera Make/Model, Software, and Original Capture Date. Flags editing software tags (e.g. "Photoshop", "GIMP").
- **GPS Coordinates**: Converts Rational Degrees/Minutes/Seconds to signed decimal degrees.
- **Haversine Distance**: Calculates spherical geodesic distance between photo coordinates and milestone location. Distance outside `settings.GPS_RADIUS_METERS` (default: 1000m) flags the check.

### 3.6 Grounded XAI Explainer (`LLM_EXPLAINABILITY`)
- **Module**: `backend/app/integrations/ai/llm_explainer.py`
- **Safety**: Sanitizes inputs against prompt injection using pattern-matching guards.
- **Grounding**: Explanations synthesize strictly computed metrics without hallucinations or statistical exaggerations.
- **Deterministic Fallback**: Comprehensive rule-based synthesis if LLM provider is unavailable or disabled.

---

## 4. Security & Isolation Boundaries

1. **Storage Access**: Binary evidence bytes are accessed strictly via `StorageProviderInterface` with sanitized paths.
2. **Memory Bounds**: Large files are read in bounded buffers; PDF processing is capped at 10 pages and 50MB.
3. **Authorization Matrix**:
   - `PLATFORM_ADMIN` & `REVIEWER`: Full access to submit and review verification reports.
   - `ORGANIZATION_OWNER`, `ADMIN`, `MEMBER`: Access limited strictly to their own organization's evidence.
   - `DONOR` & Anonymous: Denied access to internal forensic diagnostic artifacts and unapproved evidence.

