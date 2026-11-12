# Cryptographic Architecture & Signature Specification

## 1. Asymmetric Cryptography: Ed25519 (RFC 8032)

CharityChain employs the **Edwards-curve Digital Signature Algorithm (EdDSA)** using the **Ed25519** curve as specified in **RFC 8032**.

### Why Ed25519?
- **High Performance**: Fast signature generation and verification without side-channel timing vulnerabilities.
- **Compact Keys**: 32-byte public keys and 64-byte signatures, optimal for embedding in QR codes and storage-constrained environments.
- **Deterministic**: Signatures are completely deterministic and do not rely on an external entropy source during signing, preventing private key leakage via weak random number generators.
- **Superior Security**: 128-bit security level, resistant to common elliptic curve attacks.

---

## 2. Deterministic JSON Canonicalization (RFC 8785)

Digital signatures require byte-for-byte reproducibility. Different JSON serializers produce disparate whitespace, Unicode encodings, and key orderings, which would invalidate valid digital signatures.

CharityChain implements the **RFC 8785 JSON Canonicalization Scheme (JCS)** in `backend/app/core/canonical_json.py`.

### Strict JCS Rules Applied:
1. **Key Sorting**: Object keys are sorted lexicographically by their UTF-16 code unit representation.
2. **Whitespace Stripping**: Zero whitespace is emitted outside string literals (`separators=(',', ':')`).
3. **Character Encoding**: Output is strictly encoded in UTF-8 without byte-order marks (BOM).
4. **Zero-Float Prohibition**: Floating-point numbers are strictly forbidden. Monetary amounts must be represented as integer minor units (cents) or formatted strings. Any float encountered during canonicalization raises `ValueError`.

### Determinism Verification Example

Given two syntactically different JSON dictionaries:
```python
dict_a = {"currency": "USD", "amount_cents": 5000, "project": "Clean Water"}
dict_b = {"project": "Clean Water", "currency": "USD", "amount_cents": 5000}

canonicalize_json(dict_a) == canonicalize_json(dict_b)
# Both strictly evaluate to:
# '{"amount_cents":5000,"currency":"USD","project":"Clean Water"}'
```

---

## 3. Cryptographic Signing & Verification Protocol

### A. Signing Pipeline
1. **Assemble Claims**: Structure the claims dictionary conforming to the standard certificate schema.
2. **Canonicalize**: Pass claims through `canonicalize_json(claims)` to produce canonical string $S$.
3. **Digest Calculation**: Compute SHA-256 hash $H = \text{SHA-256}(S\text{.encode('utf-8')})$.
4. **Asymmetric Signing**: Sign the canonical UTF-8 bytes using the active Ed25519 private key:
   $$\text{sig\_bytes} = \text{private\_key.sign}(S\text{.encode('utf-8')})$$
5. **Base64 Encoding**: Encode signature as standard Base64 string $\text{sig\_b64}$.
6. **Persistence**: Store $\text{canonical\_payload} = S$, $\text{payload\_hash} = H$, and $\text{signature} = \text{sig\_b64}$.

### B. Verification Pipeline
1. **Recompute Hash**: Compute $H' = \text{SHA-256}(\text{canonical\_payload}\text{.encode('utf-8')})$.
   - If $H' \neq \text{payload\_hash}$, emit `PAYLOAD_MODIFIED`.
2. **Resolve Public Key**: Retrieve issuer public key matching `signing_key_id` from the JWKS registry or database.
3. **Verify Signature**: Verify $\text{sig\_bytes}$ over canonical UTF-8 payload bytes using the public key:
   $$\text{public\_key.verify}(\text{sig\_bytes}, \text{canonical\_payload}\text{.encode('utf-8')})$$
   - If verification raises `InvalidSignature`, emit `INVALID_SIGNATURE`.
4. **Evaluate Status**: If signature and payload are valid, evaluate certificate status:
   - Status == `"REVOKED"` $\implies$ emit `REVOKED`.
   - Status == `"SUSPENDED"` $\implies$ emit `SUSPENDED`.
   - Status == `"ISSUED"` or `"ACTIVE"` $\implies$ emit `VALID`.

---

## 4. Completely Offline Verification Workflow

External verifiers and regulatory auditors can independently verify certificates without network connectivity or platform API calls using the cryptographic proof envelope.

### Verification Recipe:
1. Extract `canonical_payload`, `signature`, and `signing_key_id` from the certificate proof envelope or QR payload.
2. Retrieve the public key for `signing_key_id` from a pre-cached RFC 7517 JWKS file.
3. In any standard cryptographic environment (OpenSSL, Python `cryptography`, Node.js `crypto`, Go `crypto/ed25519`):

```bash
# Independent verification via OpenSSL CLI:
openssl dgst -sha256 -verify issuer_public_key.pem -signature signature.bin canonical_payload.json
```

```python
# Independent verification in pure Python:
import base64
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ed25519

public_key = serialization.load_pem_public_key(public_key_pem.encode("utf-8"))
signature_bytes = base64.b64decode(signature_b64)
payload_bytes = canonical_payload_str.encode("utf-8")

# Will raise InvalidSignature if tampered by even a single bit
public_key.verify(signature_bytes, payload_bytes)
print("Certificate signature verified mathematically!")
```

