################################################################################
# CharityChain — Production Terraform Variables
################################################################################

variable "project_id" {
  description = "GCP Project ID"
  type        = string
}

variable "region" {
  description = "GCP Region"
  type        = string
  default     = "us-central1"
}

variable "domain" {
  description = "Production domain name"
  type        = string
  default     = "charitychain.dev"
}

variable "backend_image" {
  description = "Backend Docker image (gcr.io/PROJECT_ID/charitychain-backend:TAG)"
  type        = string
}

variable "frontend_image" {
  description = "Frontend Docker image (gcr.io/PROJECT_ID/charitychain-frontend:TAG)"
  type        = string
}

variable "alert_email" {
  description = "Email address for Cloud Monitoring alerts"
  type        = string
}

