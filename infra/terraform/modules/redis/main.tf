################################################################################
# CharityChain — Terraform: Redis Module (Google Cloud Memorystore)
# Phase 14: Rate limiting, distributed caching, session tracking
################################################################################

variable "project_id"        { type = string }
variable "region"            { type = string }
variable "env"               { type = string }
variable "network_self_link" { type = string }
variable "memory_size_gb"    { type = number; default = 1 }
variable "tier"              { type = string; default = "BASIC" }

resource "google_redis_instance" "cache" {
  name               = "charitychain-${var.env}-redis"
  project            = var.project_id
  region             = var.region
  tier               = var.tier
  memory_size_gb     = var.memory_size_gb
  authorized_network = var.network_self_link
  connect_mode       = "PRIVATE_SERVICE_ACCESS"

  redis_version      = "REDIS_7_0"
  display_name       = "CharityChain ${var.env} Redis"

  auth_enabled       = true
  transit_encryption_mode = "DISABLED"

  redis_configs = {
    maxmemory-policy = "allkeys-lru"
  }
}

# Store Redis auth string in Secret Manager
resource "google_secret_manager_secret" "redis_auth" {
  secret_id = "charitychain-${var.env}-redis-auth"
  project   = var.project_id
  replication { automatic {} }
}

resource "google_secret_manager_secret_version" "redis_auth" {
  secret      = google_secret_manager_secret.redis_auth.id
  secret_data = google_redis_instance.cache.auth_string
}

output "host"        { value = google_redis_instance.cache.host }
output "port"        { value = google_redis_instance.cache.port }
output "auth_string" { value = google_redis_instance.cache.auth_string; sensitive = true }

