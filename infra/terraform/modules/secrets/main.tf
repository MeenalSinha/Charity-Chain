################################################################################
# CharityChain — Terraform: Secrets Module (Secret Manager)
# Phase 14: Centralized encrypted secret storage & IAM bindings
################################################################################

variable "project_id"       { type = string }
variable "env"              { type = string }
variable "backend_sa_email" { type = string }

locals {
  secrets = toset([
    "secret-key",
    "jwt-secret-key",
    "encryption-key",
    "sentry-dsn",
    "paypal-client-id",
    "paypal-client-secret",
    "paypal-webhook-id",
    "smtp-password",
    "backup-encryption-key",
  ])
}

resource "google_secret_manager_secret" "app_secrets" {
  for_each  = local.secrets
  secret_id = "charitychain-${var.env}-${each.value}"
  project   = var.project_id

  replication {
    automatic = true
  }
}

# Grant Secret Accessor to Backend Service Account
resource "google_secret_manager_secret_iam_member" "backend_access" {
  for_each  = local.secrets
  project   = var.project_id
  secret_id = google_secret_manager_secret.app_secrets[each.value].secret_id
  role      = "roles/secretmanager.secretAccessor"
  member    = "serviceAccount:${var.backend_sa_email}"
}

output "secret_ids" {
  value = { for k, v in google_secret_manager_secret.app_secrets : k => v.id }
}

