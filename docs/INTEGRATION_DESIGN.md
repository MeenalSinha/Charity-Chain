# CharityChain — Integration Design Specification
**Phase 1 Deliverable — Port/Adapter Pattern, Provider Abstractions & External Contracts**

---

## 1. Integration Architecture & Design Principles

CharityChain integrates with external payment gateways, computer vision models, cloud storage, and transactional notification services using the **Ports and Adapters (Hexagonal)** architectural pattern:
- **Core Decoupling**: Domain business logic depends exclusively on abstract interfaces (Ports), never on vendor-specific client libraries.
- **Provider Swappability**: Adapters can be substituted via environment configuration (e.g. `LocalStorageProvider` -> `S3StorageProvider`, or `SandboxPayPalProvider` -> `LivePayPalProvider`) without altering domain code.
- **Strict Degradation & Reliability**: If an external provider is unconfigured, unreachable, or fails, the integration returns structured `UNAVAILABLE` statuses or raises standard exceptions. It never fabricates simulated responses.

---

## 2. Payment Gateway Integration (`PaymentProviderInterface`)

### 2.1 Interface Definition (`backend/app/integrations/payments/base.py`)
```python
class PaymentProviderInterface(ABC):
    @abstractmethod
    def create_order(self, amount: Money, currency: str, return_url: str, cancel_url: str, intent: str = "AUTHORIZE") -> Dict[str, Any]:
        """Creates an order at the payment gateway."""
        pass

    @abstractmethod
    def authorize_payment(self, order_id: str) -> Dict[str, Any]:
        """Authorizes funds against an approved order, holding them in escrow."""
        pass

    @abstractmethod
    def capture_payment(self, authorization_id: str, amount: Optional[Money] = None) -> Dict[str, Any]:
        """Captures previously authorized funds upon milestone completion."""
        pass

    @abstractmethod
    def refund_payment(self, capture_id: str, amount: Optional[Money] = None, reason: str = "") -> Dict[str, Any]:
        """Refunds captured funds back to the donor."""
        pass

    @abstractmethod
    def execute_disbursement(self, recipient_account: str, amount: Money, currency: str, note: str) -> Dict[str, Any]:
        """Transfers verified funds from platform escrow to the NGO's account."""
        pass

    @abstractmethod
    def verify_webhook_signature(self, headers: Dict[str, str], body: bytes) -> bool:
        """Verifies cryptographic authenticity of incoming webhook notifications."""
        pass
```

### 2.2 PayPal Gateway Adapter Details
- **Environments**: Configurable via `PAYPAL_MODE` (`sandbox` or `live`).
- **OAuth 2.0 Token Lifecycle**: Automatic token refresh with proactive caching (TTL 3600s).
- **Two-Phase Commit for Milestones**:
  1. *Donation*: Donor approves payment -> API calls `POST /v2/checkout/orders/{id}/authorize`. Funds are authorized and held on donor payment method.
  2. *Milestone Verified*: Reviewer approves milestone -> API calls `POST /v2/payments/authorizations/{id}/capture`. Funds settle into platform account.
  3. *Disbursement*: API triggers `POST /v1/payments/payouts` targeting NGO payout email.
- **Webhook Hardening**: Every incoming event verifies the PayPal-Transmission-Id, Cert-URL, and CRC32 signature.

---

## 3. AI & Computer Vision Integration (`AIProviderInterface`)

### 3.1 Interface Definition (`backend/app/integrations/ai/base.py`)
```python
class AIProviderInterface(ABC):
    @abstractmethod
    def detect_objects(self, image_bytes: bytes, target_labels: Optional[List[str]] = None) -> Dict[str, Any]:
        """Runs YOLO object detection. Returns detected classes, bboxes, and confidence."""
        pass

    @abstractmethod
    def analyze_tampering_ela(self, image_bytes: bytes) -> Dict[str, Any]:
        """Runs Error Level Analysis (ELA) for compression anomalies."""
        pass

    @abstractmethod
    def compute_perceptual_hash(self, image_bytes: bytes) -> str:
        """Computes perceptual hash (pHash) for image deduplication."""
        pass

    @abstractmethod
    def compare_images_ssim(self, image_bytes_a: bytes, image_bytes_b: bytes) -> float:
        """Computes Structural Similarity Index (SSIM) between before and after images."""
        pass

    @abstractmethod
    def verify_exif_gps(self, image_bytes: bytes, target_lat: float, target_lon: float, max_radius_meters: float) -> Dict[str, Any]:
        """Extracts EXIF GPS coordinates and computes Haversine distance to target site."""
        pass
```

### 3.2 Non-Fabrication Rules & Status Semantics
- **Model Unavailable**: If YOLO weights are missing from disk, `detect_objects` returns `{"status": "UNAVAILABLE", "detections": []}`. It never synthesizes fake bounding boxes.
- **GPS Unavailable**: If EXIF metadata contains no GPS tags, `verify_exif_gps` returns `{"status": "UNAVAILABLE", "distance_meters": None}`. It never injects a simulated 100-meter distance.
- **Normalized Confidence**: Overall report confidence is computed exclusively across executed checks (`PASSED` or `FAILED`). Unexecuted or `UNAVAILABLE` checks are excluded from the denominator.

---

## 4. Blob Storage Integration (`StorageProviderInterface`)

### 4.1 Interface Definition (`backend/app/integrations/storage/base.py`)
```python
class StorageProviderInterface(ABC):
    @abstractmethod
    def save_file(self, file_bytes: bytes, file_name: str, directory_prefix: str = "evidence") -> Tuple[str, str]:
        """Saves file bytes. Returns (storage_path, sha256_hash)."""
        pass

    @abstractmethod
    def get_file(self, storage_path: str) -> bytes:
        """Retrieves file bytes by path."""
        pass

    @abstractmethod
    def delete_file(self, storage_path: str) -> bool:
        """Deletes file by path."""
        pass

    @abstractmethod
    def get_download_url(self, storage_path: str, expiry_seconds: int = 3600) -> str:
        """Generates a secure, temporary pre-signed download URL."""
        pass
```

### 4.2 Adapters
- **`LocalStorageProvider`**: Writes evidence files to `data/evidence/<sha256[:2]>/<sha256>.<ext>` with directory sharding to prevent file system degradation.
- **`S3StorageProvider`**: Future adapter supporting AWS S3 or MinIO with server-side AES-256 encryption.

---

## 5. Notification Integration (`NotificationProviderInterface`)

### 5.1 Interface Definition (`backend/app/integrations/notifications/base.py`)
```python
class NotificationProviderInterface(ABC):
    @abstractmethod
    def send_email(self, recipient: str, subject: str, template_name: str, context: Dict[str, Any]) -> bool:
        """Sends a templated email notification."""
        pass

    @abstractmethod
    def send_webhook(self, url: str, event_name: str, payload: Dict[str, Any]) -> bool:
        """Dispatches an outbound webhook event with HMAC-SHA256 signature."""
        pass
```

### 5.2 Key Notification Traces
- **Donation Received**: Sent to donor with pending authorization details.
- **Evidence Review Requested**: Alert dispatched to platform reviewers.
- **Milestone Verified**: Sent to donors with link to downloadable Proof-of-Impact Certificate.
- **Disbursement Complete**: Sent to NGO confirming payout batch execution.

