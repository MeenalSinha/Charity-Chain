################################################################################
# CharityChain — Terraform Backend Configuration
# Phase 14: GCS remote state with locking
################################################################################

terraform {
  required_version = ">= 1.6"

  backend "gcs" {
    bucket = "charitychain-terraform-state"
    prefix = "terraform/state"
  }

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }
}

