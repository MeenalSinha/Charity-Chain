# Phase 14 — Deployment Architecture Decision

**Date:** 2026-11-14  
**Decision Status:** ACCEPTED  
**Chosen Platform:** Google Cloud Run (Serverless Containers)

---

## 1. Decision Context

CharityChain requires a production deployment architecture capable of:

- Hosting the FastAPI backend (Python 3.11)
- Hosting the Next.js frontend (Node.js 22)
- Connecting to PostgreSQL and Redis databases
- Processing AI evidence verification workloads
- Supporting 100–10,000 concurrent users during launch
- Meeting SOC 2 readiness criteria
- Operating within a startup cost envelope

---

## 2. Candidates Evaluated

| Factor | Google Cloud Run | GKE (Kubernetes) | AWS ECS Fargate | Fly.io |
|:---|:---:|:---:|:---:|:---:|
| **Operational complexity** | Low ✅ | High ❌ | Medium | Low ✅ |
| **Cost at low traffic** | Very low (scales to 0) ✅ | High (always-on nodes) ❌ | Low ✅ | Medium |
| **Cost at high traffic** | Competitive | Lower at scale | Competitive | Higher |
| **Auto-scaling** | Built-in ✅ | Manual/HPA | Built-in | Manual |
| **Managed TLS** | Built-in ✅ | Manual/cert-manager | Manual | Built-in |
| **Private VPC connectivity** | VPC Connector ✅ | Native ✅ | Native ✅ | Overlay |
| **Cloud SQL integration** | Native ✅ | Proxy sidecar | N/A | N/A |
| **Secret Manager** | Native ✅ | Workload Identity | AWS Secrets Mgr | Manual |
| **GCP-native observability** | ✅ | ✅ | ❌ | ❌ |
| **Multi-region HA** | Supported | Supported | Supported | Limited |
| **SOC 2 readiness** | ✅ GCP certifications | ✅ | ✅ | Partial |
| **Deployment automation** | Cloud Deploy / gcloud ✅ | Helm/Argo CD | CodePipeline | flyctl |
| **Team expertise required** | Minimal ✅ | Kubernetes expertise | AWS expertise | Minimal |

---

## 3. Decision: Google Cloud Run

**Rationale:**

1. **Operational simplicity**: Cloud Run eliminates cluster management, node pool sizing, and pod scheduling. The team can focus on application code rather than infrastructure.

2. **Scale-to-zero economics**: During development and low-traffic periods, Cloud Run services cost nothing when idle. Kubernetes clusters incur ~$75+/month for the smallest node pool.

3. **Native GCP integration**: Cloud SQL Auth Proxy, Memorystore Redis (via VPC Connector), Secret Manager, Cloud Monitoring, and Cloud Logging integrate without additional configuration.

4. **Blue/Green deployment**: Cloud Run's traffic splitting enables instant rollback by routing 100% traffic back to the previous revision without container replacement.

5. **SOC 2 alignment**: GCP holds ISO 27001, SOC 1/2/3, and PCI DSS certifications. Cloud Run inherits these certifications for infrastructure controls.

6. **Container portability**: The hardened Docker images developed for Cloud Run work identically on GKE, Kubernetes, or docker-compose. No vendor lock-in.

---

## 4. Architecture Overview

```
                         ┌─────────────────────────────────────────┐
                         │          Google Cloud Platform           │
                         │                                          │
  Internet ──► HTTPS ──► │  Cloud Load Balancer (Global HTTPS LB)  │
                         │            │                             │
                         │    ┌───────┴──────────┐                 │
                         │    │                  │                  │
                         │  Cloud Run          Cloud Run           │
                         │  Frontend           Backend             │
                         │  (nextjs)           (uvicorn)           │
                         │    │                  │                  │
                         │    └───────┬──────────┘                 │
                         │            │                             │
                         │       VPC Connector                      │
                         │            │                             │
                         │    ┌───────┴──────────────────┐         │
                         │    │   Private VPC Network    │         │
                         │    │                          │         │
                         │    │  Cloud SQL (PostgreSQL)  │         │
                         │    │  Memorystore (Redis)     │         │
                         │    │  Cloud Storage (GCS)     │         │
                         │    │  Secret Manager          │         │
                         │    └──────────────────────────┘         │
                         └─────────────────────────────────────────┘
```

---

## 5. Environment Strategy

| Environment | Purpose | Configuration |
|:---|:---|:---|
| **Local (dev)** | Development with SQLite | `docker-compose.yml` |
| **Staging** | Pre-production integration testing | `docker-compose.staging.yml` or Cloud Run staging |
| **Production** | Live user traffic | Cloud Run production + `docker-compose.prod.yml` |

---

## 6. Deployment Decision: Terraform + Cloud Run

- **IaC**: Terraform (GCS remote state)
- **Container Registry**: Google Container Registry (GCR)
- **CI/CD**: GitHub Actions
- **Traffic Management**: Cloud Run traffic splitting (Blue/Green)
- **Database**: Cloud SQL PostgreSQL 16 (Regional HA in production)
- **Cache/Rate-Limit**: Memorystore Redis (Standard HA in production)
- **Object Storage**: Cloud Storage (evidence files, backups)
- **Secrets**: Secret Manager
- **Monitoring**: Cloud Monitoring + Prometheus (self-hosted) + Grafana + Sentry

---

## 7. Decision Record

| Attribute | Value |
|:---|:---|
| Decision maker | CharityChain Engineering |
| Date | 2026-11-14 |
| Status | ACCEPTED |
| Primary platform | Google Cloud Run |
| Container registry | GCR (gcr.io) |
| IaC tool | Terraform |
| CI/CD | GitHub Actions |
| Revisit trigger | Traffic > 50,000 req/day sustained for 30 days (consider GKE migration) |

