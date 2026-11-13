################################################################################
# CharityChain — Terraform: Storage Module (Google Cloud Storage)
# Phase 14: Evidence storage with legal holds and encrypted backups
################################################################################

variable "project_id"      { type = string }
variable "region"          { type = string }
variable "env"             { type = string }
variable "evidence_bucket" { type = string }
variable "backup_bucket"   { type = string }
variable "lifecycle_days"  { type = number; default = 2555 } # 7 years

resource "google_storage_bucket" "evidence" {
  name          = var.evidence_bucket
  project       = var.project_id
  location      = var.region
  force_destroy = false

  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = var.lifecycle_days
    }
    action {
      type = "SetStorageClass"
      storage_class = "COLDLINE"
    }
  }

  cors {
    origin          = ["https://charitychain.dev", "https://*.charitychain.dev"]
    method          = ["GET", "HEAD", "PUT", "POST"]
    response_header = ["*"]
    max_age_seconds = 3600
  }
}

resource "google_storage_bucket" "backups" {
  name          = var.backup_bucket
  project       = var.project_id
  location      = var.region
  force_destroy = false

  uniform_bucket_level_access = true

  versioning {
    enabled = true
  }

  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }

  lifecycle_rule {
    condition {
      age = 365
    }
    action {
      type = "SetStorageClass"
      storage_class = "COLDLINE"
    }
  }
}

output "evidence_bucket_name" { value = google_storage_bucket.evidence.name }
output "backup_bucket_name"   { value = google_storage_bucket.backups.name }

