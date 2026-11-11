# Image Forensics & Tamper Detection — CharityChain Phase 6

## 1. Overview of Forensic Techniques

CharityChain's forensic subsystem evaluates incoming evidence assets to detect synthetic alterations, splicing, copy-paste tampering, format inconsistencies, and baseline reuse without genuine physical progress.

Forensic analysis is performed through three dedicated components:
1. **Error Level Analysis (ELA)**: Detects compression discrepancies across spatial regions in JPEG images.
2. **Perceptual Hashing (pHash)**: Evaluates low-frequency structural fingerprinting to detect identical photo re-submissions or benign transformations.
3. **Structural Similarity Index (SSIM)**: Quantifies luminance, contrast, and structural degradation between current evidence and baseline milestone photos.

---

## 2. Error Level Analysis (ELA)

### Physical Principle
JPEG is a lossy compression algorithm operating on 8x8 discrete cosine transform (DCT) blocks. Each successive re-save at a given quality level introduces less compression error until an equilibrium error rate is reached.

When an image is modified (e.g. inserting an object or text from another image), the spliced region possesses a different compression history and will display significantly higher error levels compared to unmodified surrounding regions when resaved at a known quality level (90%).

### Pipeline Algorithm
```python
1. Open source JPEG image.
2. Recompress image to temporary buffer at quality=90.
3. Compute absolute difference: diff = |original - resaved|.
4. Amplify error for visualization: amplified = diff * scale_factor (10).
5. Calculate statistical metrics:
   - mean_difference
   - max_difference
   - channel_means (R, G, B)
6. If mean_difference > settings.TAMPER_ELA_THRESHOLD:
   return FLAGGED with potential manipulation warning.
   Else:
   return PASSED.
```

### Format Constraints & Honesty
- **JPEG**: Full analysis and amplified heatmap generation.
- **PNG / WebP / PDF**: Lossless or non-DCT compression formats do not exhibit JPEG block error decay. The analyzer returns `NOT_APPLICABLE` with explicit explanation: *"Error Level Analysis is only applicable to lossy JPEG formats. Format PNG does not exhibit DCT compression discrepancies."*

---

## 3. Perceptual Hashing (pHash)

### Algorithm Details
1. Convert image to grayscale.
2. Downscale to 32x32 pixels to remove high-frequency noise and details.
3. Compute 2D Discrete Cosine Transform (DCT).
4. Retain top-left 8x8 low frequencies (excluding DC term at [0,0]).
5. Compute median of the 64 coefficients.
6. Generate 64-bit binary hash where bit = 1 if coefficient > median, else 0.

### Hamming Distance Thresholds
The Hamming distance measures the number of bit positions in which two hashes differ:

| Hamming Distance | Physical Interpretation | System Action |
|------------------|-------------------------|---------------|
| `0` | Identical or bit-level equivalent image | Flagged for possible duplicate/reuse |
| `1 – 10` | Benign transformations (resize, small crop, slight color shift) | Consistent with same photograph |
| `11 – 25` | Similar scene with minor physical alterations or different camera angle | Progress detection candidate |
| `> 25` | Distinct visual scenes or major structural modifications | Independent image |

---

## 4. Structural Similarity Index (SSIM)

The Structural Similarity Index (via `skimage.metrics.structural_similarity`) computes three components of visual perception:
- **Luminance ($l$)**: Compares mean pixel intensity.
- **Contrast ($c$)**: Compares standard deviation of pixel intensities.
- **Structure ($s$)**: Compares cross-correlation between normalized signals.

### Evaluation Criteria
$$\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$

- **$\text{SSIM} \ge 0.98$**: Exact or near-exact duplicates. Flagged when comparing a current milestone photo to a baseline.
- **$0.40 \le \text{SSIM} < 0.85$**: Demonstrates consistent site environment with visible construction, installation, or vegetation changes.
- **$\text{SSIM} < 0.30$**: Completely different view, framing, or site.

---

## 5. Diagnostic Artifact Generation

To assist human reviewers in Phase 7, the system automatically generates visual forensic artifacts:
- **ELA Amplified Heatmap**: PNG showing high-contrast compression discrepancies.
- **Side-by-Side Progression Composite**: Dual-panel PNG showing baseline vs. current photograph with superimposed pHash distance and SSIM metrics.
