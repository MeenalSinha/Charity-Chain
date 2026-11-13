################################################################################
# CharityChain — Terraform: Monitoring Module (Cloud Monitoring)
# Phase 14: Automated alerting, uptime checks, SLI/SLO dashboards
################################################################################

variable "project_id"  { type = string }
variable "region"      { type = string }
variable "env"         { type = string }
variable "alert_email" { type = string }

# Notification Channel (Email)
resource "google_monitoring_notification_channel" "email" {
  display_name = "CharityChain ${var.env} Alert Email"
  type         = "email"
  project      = var.project_id

  labels = {
    email_address = var.alert_email
  }
}

# Alert Policy: High HTTP 5xx Error Rate (> 2% for 5 minutes)
resource "google_monitoring_alert_policy" "high_error_rate" {
  display_name = "CharityChain [${var.env}] High 5xx Error Rate"
  project      = var.project_id
  combiner     = "OR"

  conditions {
    display_name = "Cloud Run 5xx error rate > 2%"

    condition_threshold {
      filter          = "resource.type = \"cloud_run_revision\" AND metric.type = \"run.googleapis.com/request_count\" AND metric.labels.response_code_class = \"5xx\""
      duration        = "300s"
      comparison      = "COMPARISON_GT"
      threshold_value = 0.02

      aggregations {
        alignment_period     = "60s"
        per_series_aligner   = "ALIGN_RATE"
        cross_series_reducer = "REDUCE_SUM"
      }
    }
  }

  notification_channels = [google_monitoring_notification_channel.email.name]

  alert_strategy {
    auto_close = "1800s" # 30 min
  }
}

# Alert Policy: Cloud SQL High CPU Utilization (> 85% for 10 minutes)
resource "google_monitoring_alert_policy" "high_db_cpu" {
  display_name = "CharityChain [${var.env}] High Database CPU"
  project      = var.project_id
  combiner     = "OR"

  conditions {
    display_name = "Cloud SQL CPU utilization > 85%"

    condition_threshold {
      filter          = "resource.type = \"cloud_sql_database\" AND metric.type = \"cloudsql.googleapis.com/database/cpu/utilization\""
      duration        = "600s"
      comparison      = "COMPARISON_GT"
      threshold_value = 0.85

      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_MEAN"
      }
    }
  }

  notification_channels = [google_monitoring_notification_channel.email.name]
}

# Alert Policy: Cloud SQL Low Disk Space (< 15% free)
resource "google_monitoring_alert_policy" "low_db_disk" {
  display_name = "CharityChain [${var.env}] Low Database Disk Space"
  project      = var.project_id
  combiner     = "OR"

  conditions {
    display_name = "Cloud SQL disk utilization > 85%"

    condition_threshold {
      filter          = "resource.type = \"cloud_sql_database\" AND metric.type = \"cloudsql.googleapis.com/database/disk/utilization\""
      duration        = "300s"
      comparison      = "COMPARISON_GT"
      threshold_value = 0.85

      aggregations {
        alignment_period   = "60s"
        per_series_aligner = "ALIGN_MEAN"
      }
    }
  }

  notification_channels = [google_monitoring_notification_channel.email.name]
}

output "notification_channel_id" { value = google_monitoring_notification_channel.email.id }

