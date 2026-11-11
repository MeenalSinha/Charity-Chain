# Verification Scoring, Assessment Logic & Calibration — CharityChain Phase 6

## 1. Conservative Scoring Methodology

CharityChain adheres to a principle of conservative, explainable scoring. Because AI models operate with inherent uncertainty and domain shifts, confidence scores must reflect observable evidence without artificial inflation or false precision.

### Overall Assessment States

The engine produces an `overall_assessment` corresponding to one of four strictly typed values (`OverallAssessmentStatus`):

| Assessment Status | Meaning | Conditions Required |
|-------------------|---------|---------------------|
| `CONSISTENT` | Strong multi-indicator evidence supports authentic milestone execution | File integrity intact, no critical forensic flags, relevant objects detected, and confidence $\ge 0.70$. |
| `INCONCLUSIVE` | Evidence provides partial confirmation but critical indicators are missing or borderline | Missing EXIF/GPS, low detection confidence, or unresolvable category ambiguity. |
| `FLAGGED` | Anomalies, discrepancies, or potential manipulation detected | ELA compression threshold exceeded, zero-progress duplicate baseline photo, or software editing tags present. |
| `FAILED` | Hard integrity failure or corrupted asset | Hash mismatch against database record, empty file, or non-decodable byte stream. |

---

## 2. Multi-Indicator Scoring Weights

Confidence score ($C \in [0.0, 1.0]$) is computed using weighted aggregation across active analyzers:

| Indicator / Check Type | Weight ($w_i$) | Evaluation Logic |
|------------------------|----------------|------------------|
| **File Integrity** | Mandatory | If failed, $C = 0.0$ and assessment = `FAILED`. |
| **Object Detection & Relevance** | 0.35 | Detection confidence $\times$ category relevance ratio. |
| **Error Level Analysis (ELA)** | 0.25 | $1.0 - \min(1.0, \frac{\text{mean\_diff}}{\text{threshold}})$. |
| **Progress Comparison** | 0.20 | $1.0$ if progress verified; $0.2$ if duplicate flagged; $0.7$ if not applicable (no baseline). |
| **Geospatial & EXIF Proximity** | 0.20 | $1.0$ if within radius; $0.4$ if distance exceeded; $0.6$ if no GPS available. |

When certain checks return `NOT_APPLICABLE` (e.g. PDF documents or images without baselines), weights are dynamically normalized over the subset of applicable indicators:

$$C = \frac{\sum_{i \in \text{Applicable}} w_i \cdot s_i}{\sum_{i \in \text{Applicable}} w_i}$$

---

## 3. Calibration Status & Empirical Transparency

### Mandatory `calibration_status`
Every `VerificationReport` and API response includes:
```json
{
  "confidence_score": 0.85,
  "calibration_status": "UNVALIDATED"
}
```

### Rationale
In machine learning, a confidence score of $0.85$ implies that among instances predicted with this score, $85\%$ are objectively positive. Establishing this requires:
1. A calibrated Platt scaling, isotonic regression, or temperature scaling layer.
2. A large, statistically representative holdout validation set across real-world charity project photographic evidence.

Because CharityChain does not yet maintain an externally validated multi-thousand photographic benchmark across global charity operations, claiming that confidence scores represent empirical probabilities would be misleading.

CharityChain explicitly documents and serializes `calibration_status = "UNVALIDATED"` across all API contracts and database records.

---

## 4. Human-in-the-Loop Safeguards

1. **Non-Executable Findings**: AI findings are purely advisory and cannot trigger database state changes in `Donation`, `PaymentCapture`, or `Milestone.status`.
2. **Reviewer Highlighting**: The UI presents flags, confidence scores, and diagnostic heatmaps to human reviewers with explicit recommendations for manual inspection.
3. **Audit Trail**: Every verification result is cryptographically signed and chained into `evidence_audit_events`.

