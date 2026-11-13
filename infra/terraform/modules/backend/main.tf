################################################################################
# CharityChain — Terraform: Backend Module (Cloud Run)
# Phase 14: FastAPI Backend with VPC Access, Cloud SQL & Memorystore
################################################################################

variable "project_id"         { type = string }
variable "region"             { type = string }
variable "env"                { type = string }
variable "image"              { type = string }
variable "vpc_connector_name" { type = string }
variable "db_connection_name" { type = string }
variable "redis_host"         { type = string }
variable "redis_port"         { type = number }
variable "evidence_bucket"    { type = string }
variable "min_instances"      { type = number; default = 1 }
variable "max_instances"      { type = number; default = 10 }
variable "memory"             { type = string; default = "1Gi" }
variable "cpu"                { type = string; default = "1" }

# Service Account for Cloud Run Backend
resource "google_service_account" "backend_sa" {
  account_id   = "cc-backend-${var.env}"
  display_name = "CharityChain Backend ${var.env} Service Account"
  project      = var.project_id
}

# Grant Storage Admin on evidence bucket
resource "google_storage_bucket_iam_member" "evidence_access" {
  bucket = var.evidence_bucket
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.backend_sa.email}"
}

# Cloud Run Service (v2 API)
resource "google_cloud_run_v2_service" "backend" {
  name     = "charitychain-backend-${var.env}"
  location = var.region
  project  = var.project_id

  template {
    service_account = google_service_account.backend_sa.email

    scaling {
      min_instance_count = var.min_instances
      max_instance_count = var.max_instances
    }

    vpc_access {
      connector = var.vpc_connector_name
      egress    = "PRIVATE_RANGES_ONLY"
    }

    containers {
      image = var.image

      resources {
        limits = {
          memory = var.memory
          cpu    = var.cpu
        }
      }

      env {
        name  = "ENVIRONMENT"
        value = var.env
      }

      env {
        name  = "DEBUG"
        value = var.env == "production" ? "false" : "false"
      }

      env {
        name  = "STORAGE_PROVIDER"
        value = "s3"
      }

      env {
        name  = "S3_BUCKET_NAME"
        value = var.evidence_bucket
      }

      env {
        name  = "REDIS_HOST"
        value = var.redis_host
      }

      env {
        name  = "REDIS_PORT"
        value = tostring(var.redis_port)
      }

      env {
        name  = "RATE_LIMIT_REDIS_REQUIRED"
        value = var.env == "production" ? "true" : "false"
      }

      startup_probe {
        http_get {
          path = "/health"
          port = 8000
        }
        initial_delay_seconds = 10
        period_seconds        = 10
        failure_threshold     = 3
      }

      liveness_probe {
        http_get {
          path = "/health"
          port = 8000
        }
        period_seconds    = 15
        failure_threshold = 3
      }
    }
  }

  traffic {
    type    = "TRAFFIC_TARGET_ALLOCATION_TYPE_LATEST"
    percent = 100
  }
}

# Allow public invocation (guarded by load balancer / authentication)
resource "google_cloud_run_v2_service_iam_member" "public_invoker" {
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.backend.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# Serverless NEG for Global Load Balancer
resource "google_compute_region_network_endpoint_group" "backend_neg" {
  name                  = "cc-backend-neg-${var.env}"
  network_endpoint_type = "SERVERLESS"
  region                = var.region
  project               = var.project_id

  cloud_run {
    service = google_cloud_run_v2_service.backend.name
  }
}

output "service_name"          { value = google_cloud_run_v2_service.backend.name }
output "service_url"           { value = google_cloud_run_v2_service.backend.uri }
output "service_account_email" { value = google_service_account.backend_sa.email }
output "neg_name"              { value = google_compute_region_network_endpoint_group.backend_neg.name }
output "neg_id"                { value = google_compute_region_network_endpoint_group.backend_neg.id }

