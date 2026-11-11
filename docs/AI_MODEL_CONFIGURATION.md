# AI Model & Analyzer Configuration — CharityChain Phase 6

## 1. Model Configuration Reference

The AI Evidence Verification Engine is configured through environment variables managed by `backend/app/core/config.py`.

| Configuration Key | Type | Default Value | Purpose |
|-------------------|------|---------------|---------|
| `AI_VERIFICATION_ENABLED` | bool | `True` | Master toggle for the AI verification engine |
| `YOLO_MODEL_NAME` | str | `"yolov8n.pt"` | Pretrained YOLO model variant (nano recommended for edge/CPU) |
| `YOLO_WEIGHTS_PATH` | str | `""` | Optional local filesystem override for YOLO model weights |
| `YOLO_CONFIDENCE_THRESHOLD` | float | `0.25` | Minimum bounding box confidence for detected objects |
| `VERIFICATION_CONFIDENCE_THRESHOLD` | float | `0.70` | Pipeline-wide threshold for marking overall assessment `CONSISTENT` |
| `TAMPER_ELA_THRESHOLD` | float | `15.0` | Maximum acceptable mean difference error before flagging potential JPEG manipulation |
| `GPS_RADIUS_METERS` | float | `1000.0` | Radius tolerance in meters between photo GPS coordinates and milestone coordinates |
| `LLM_EXPLAINABILITY_ENABLED` | bool | `False` | Toggle for external LLM API calls; falls back to deterministic rule explainer |
| `LLM_API_KEY` | str | `""` | API key for LLM provider (when enabled) |
| `LLM_MODEL` | str | `"gemini-1.5-flash"` | Target LLM model for grounded explanations |

---

## 2. Pinned Weights & Supply-Chain Security

To protect against model poisoning, tampering, or supply-chain degradation, model weights are checked against cryptographic digests prior to execution:

- **Model**: `yolov8n.pt`
- **Expected SHA-256**: `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`
- **Verification Behavior**:
  - If a local file is provided via `YOLO_WEIGHTS_PATH`, its SHA-256 is checked against the pinned digest.
  - If the hash mismatches, initialization fails with `ModelLoadError` rather than executing untrusted binary weights.
  - During test execution, mock detectors or pre-verified test weights can be injected via `set_ai_provider(...)`.

---

## 3. Category Relevance Configuration

Project category object mappings are defined in `backend/app/integrations/ai/relevance_mapper.py`:

```python
PROJECT_CATEGORY_RELEVANCE_MAP: Dict[str, Set[str]] = {
    "WATER_SANITATION": {"bottle", "sink", "toilet", "cup", "bowl", "boat", "fire hydrant", "person"},
    "EDUCATION": {"book", "chair", "dining table", "backpack", "laptop", "keyboard", "mouse", "person", "cell phone", "clock"},
    "ENVIRONMENT": {"potted plant", "bird", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "person"},
    "HEALTHCARE": {"bed", "person", "bottle", "chair", "sink", "refrigerator"},
    "SHELTER_HOUSING": {"chair", "bed", "couch", "truck", "car", "person", "suitcase", "backpack", "bench"},
    "RENEWABLE_ENERGY": {"truck", "car", "person", "traffic light", "potted plant", "chair"},
    "COMMUNITY_DEVELOPMENT": {"person", "bicycle", "motorcycle", "car", "truck", "bus", "bench", "chair", "potted plant", "book"}
}
```

### Contextual Matching Logic
1. Category strings are normalized and sanitized of special characters.
2. Tokenized matching checks for keyword presence (e.g., `water` -> `WATER_SANITATION`).
3. Milestone title and project description are analyzed against `CATEGORY_ALIASES` to augment expected classes.
4. If no specific category matches, a general fallback set (`{"person", "car", "truck", "bench", "chair", "potted plant"}`) is applied.

---

## 4. Resource Allocation & Hardware Requirements

- **CPU**: Minimal 2 cores; YOLOv8n inference completes in ~120-250ms on standard x86_64 CPUs.
- **Memory**: 512MB RAM overhead for PyTorch and YOLO model caching.
- **Storage**: ~6.2MB for `yolov8n.pt` model weights. Diagnostic artifacts are stored as PNGs in `<storage_root>/diagnostics/`.
