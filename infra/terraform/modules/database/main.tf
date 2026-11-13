################################################################################
# CharityChain — Terraform: Database Module (Cloud SQL PostgreSQL 16)
# Phase 14: Private IP, regional HA, automated backups
################################################################################

variable "project_id"         { type = string }
variable "region"             { type = string }
variable "env"                { type = string }
variable "network_self_link"  { type = string }
variable "allocated_ip_range" { type = string }
variable "tier"               { type = string; default = "db-custom-2-7680" }
variable "availability_type"  { type = string; default = "ZONAL" }
variable "backup_enabled"     { type = bool;   default = true }
variable "backup_start_time"  { type = string; default = "03:00" }
variable "deletion_protection" { type = bool;  default = true }

resource "random_password" "db_password" {
  length  = 32
  special = false
}

resource "google_sql_database_instance" "postgres" {
  name             = "charitychain-${var.env}-db"
  project          = var.project_id
  region           = var.region
  database_version = "POSTGRES_16"
  deletion_protection = var.deletion_protection

  settings {
    tier              = var.tier
    availability_type = var.availability_type
    disk_autoresize   = true
    disk_size         = 50

    backup_configuration {
      enabled                        = var.backup_enabled
      point_in_time_recovery_enabled = true
      start_time                     = var.backup_start_time
      backup_retention_settings {
        retained_backups = 30
        retention_unit   = "COUNT"
      }
    }

    ip_configuration {
      ipv4_enabled    = false
      private_network = var.network_self_link
      allocated_ip_range = var.allocated_ip_range
      ssl_mode        = "ENCRYPTED_ONLY"
    }

    maintenance_window {
      day          = 7    # Sunday
      hour         = 2    # 02:00 UTC
      update_track = "stable"
    }

    database_flags {
      name  = "log_min_duration_statement"
      value = "1000"   # Log queries > 1 second
    }
    database_flags {
      name  = "log_connections"
      value = "on"
    }
    database_flags {
      name  = "log_disconnections"
      value = "on"
    }
  }
}

resource "google_sql_database" "charitychain" {
  name     = "charitychain"
  project  = var.project_id
  instance = google_sql_database_instance.postgres.name
}

resource "google_sql_user" "app_user" {
  name     = "charity_user"
  project  = var.project_id
  instance = google_sql_database_instance.postgres.name
  password = random_password.db_password.result
}

# Store DB password in Secret Manager
resource "google_secret_manager_secret" "db_password" {
  secret_id = "charitychain-${var.env}-db-password"
  project   = var.project_id
  replication { automatic {} }
}

resource "google_secret_manager_secret_version" "db_password" {
  secret      = google_secret_manager_secret.db_password.id
  secret_data = random_password.db_password.result
}

output "connection_name"  { value = google_sql_database_instance.postgres.connection_name }
output "private_ip"       { value = google_sql_database_instance.postgres.private_ip_address }
output "db_name"          { value = google_sql_database.charitychain.name }
output "db_user"          { value = google_sql_user.app_user.name }

