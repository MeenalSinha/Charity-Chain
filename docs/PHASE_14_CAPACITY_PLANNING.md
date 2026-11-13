# Phase 14 — Capacity Planning

**Version:** 1.0  
**Date:** 2026-11-14

---

## 1. Traffic Projections

| Phase | Users | Requests/min | Donations/day | Storage/month |
|:---|:---:|:---:|:---:|:---:|
| **Launch (Month 1–3)** | ~100 | ~50 | ~20 | ~5 GB |
| **Growth (Month 4–12)** | ~1,000 | ~500 | ~200 | ~50 GB |
| **Scale (Year 2+)** | ~10,000 | ~5,000 | ~2,000 | ~500 GB |

---

## 2. Cloud Run Sizing

| Service | Launch | Growth | Scale |
|:---|:---:|:---:|:---:|
| **Backend min-instances** | 1 | 2 | 5 |
| **Backend max-instances** | 5 | 20 | 100 |
| **Backend memory** | 512Mi | 1Gi | 2Gi |
| **Backend CPU** | 1 | 2 | 4 |
| **Frontend min-instances** | 1 | 2 | 3 |
| **Frontend max-instances** | 5 | 10 | 50 |

---

## 3. Database Sizing

| Metric | Launch | Growth | Scale |
|:---|:---:|:---:|:---:|
| **Cloud SQL tier** | db-g1-small | db-custom-2-7680 | db-custom-4-15360 |
| **vCPUs** | 1 | 2 | 4 |
| **RAM** | 1.7 GB | 7.5 GB | 15 GB |
| **Storage** | 10 GB | 50 GB | 200 GB |
| **Connections** | 25 max | 100 max | 400 max |
| **Availability** | ZONAL | REGIONAL HA | REGIONAL HA |

**Connection pooling**: SQLAlchemy pool_size=5, max_overflow=10. With 2 Cloud Run instances at scale, max 30 connections per instance. For > 20 replicas, use PgBouncer.

---

## 4. Redis Sizing

| Metric | Launch | Growth | Scale |
|:---|:---:|:---:|:---:|
| **Tier** | BASIC | STANDARD_HA | STANDARD_HA |
| **Memory** | 1 GB | 2 GB | 5 GB |
| **Keys estimate** | ~10K | ~100K | ~1M |

Rate limiter uses ~200 bytes per key (user + endpoint + window). At 10,000 active users across 10 endpoints: ~20 MB — well within any tier.

---

## 5. Storage Sizing

| Data type | Growth rate | 7-year total |
|:---|:---:|:---:|
| **Evidence files** (avg 2 MB each, 200/month) | ~400 MB/month | ~35 GB |
| **Certificate PDFs** (avg 500 KB each, 50/month) | ~25 MB/month | ~2 GB |
| **Database backups** (compressed) | ~1 GB/month | ~90 GB |
| **Logs** (Cloud Logging) | ~10 GB/month | ~840 GB |

GCS Nearline/Coldline lifecycle rules reduce storage costs by ~70% after 90 days.

---

## 6. Cost Estimates (USD/month)

| Service | Launch | Growth | Scale |
|:---|:---:|:---:|:---:|
| Cloud Run (backend) | ~$10 | ~$80 | ~$800 |
| Cloud Run (frontend) | ~$5 | ~$40 | ~$400 |
| Cloud SQL | ~$25 | ~$150 | ~$600 |
| Memorystore Redis | ~$35 | ~$80 | ~$160 |
| Cloud Storage (GCS) | ~$2 | ~$15 | ~$80 |
| Cloud Load Balancer | ~$20 | ~$30 | ~$50 |
| Secret Manager | ~$1 | ~$2 | ~$5 |
| **Estimated Total** | **~$100** | **~$400** | **~$2,100** |

Estimates are approximations based on us-central1 pricing. Actual costs depend on traffic patterns.

---

## 7. Scale Trigger Points

| Metric | Current limit | Action at trigger |
|:---|:---:|:---|
| DB connections > 80% capacity | 80 connections | Add Cloud SQL replica, consider PgBouncer |
| Cloud Run P95 latency > 2s | — | Increase CPU allocation, check slow queries |
| Redis memory > 70% | 1.4 GB | Upgrade to larger tier |
| Evidence bucket > 1 TB | — | Review lifecycle policies, consider CDN |
| Backend requests > 10,000/min sustained | — | Evaluate GKE migration |

