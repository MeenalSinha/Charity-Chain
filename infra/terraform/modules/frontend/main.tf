################################################################################
# CharityChain — Terraform: Frontend Module (Cloud Run)
# Phase 14: Next.js App Router Containerized Frontend
################################################################################

variable "project_id"    { type = string }
variable "region"        { type = string }
variable "env"           { type = string }
variable "image"         { type = string }
variable "backend_url"   { type = string }
variable "min_instances" { type = number; default = 1 }
variable "max_instances" { type = number; default = 10 }
variable "memory"        { type = string; default = "512Mi" }
variable "cpu"           { type = string; default = "1" }

resource "google_service_account" "frontend_sa" {
  account_id   = "cc-frontend-${var.env}"
  display_name = "CharityChain Frontend ${var.env} Service Account"
  project      = var.project_id
}

resource "google_cloud_run_v2_service" "frontend" {
  name     = "charitychain-frontend-${var.env}"
  location = var.region
  project  = var.project_id

  template {
    service_account = google_service_account.frontend_sa.email

    scaling {
      min_instance_count = var.min_instances
      max_instance_count = var.max_instances
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
        name  = "NODE_ENV"
        value = "production"
      }

      env {
        name  = "BACKEND_URL"
        value = var.backend_url
      }

      startup_probe {
        http_get {
          path = "/api/health"
          port = 3000
        }
        initial_delay_seconds = 10
        period_seconds        = 10
        failure_threshold     = 3
      }

      liveness_probe {
        http_get {
          path = "/api/health"
          port = 3000
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

resource "google_cloud_run_v2_service_iam_member" "public_invoker" {
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.frontend.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# Serverless NEG for Global Load Balancer
resource "google_compute_region_network_endpoint_group" "frontend_neg" {
  name                  = "cc-frontend-neg-${var.env}"
  network_endpoint_type = "SERVERLESS"
  region                = var.region
  project               = var.project_id

  cloud_run {
    service = google_cloud_run_v2_service.frontend.name
  }
}

output "service_name" { value = google_cloud_run_v2_service.frontend.name }
output "service_url"  { value = google_cloud_run_v2_service.frontend.uri }
output "neg_name"     { value = google_compute_region_network_endpoint_group.frontend_neg.name }
output "neg_id"       { value = google_compute_region_network_endpoint_group.frontend_neg.id }

