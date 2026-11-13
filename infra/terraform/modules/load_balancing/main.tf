################################################################################
# CharityChain — Terraform: Load Balancing Module (Cloud Load Balancing)
# Phase 14: Global External HTTPS LB with Cloud Armor & SSL Termination
################################################################################

variable "project_id"   { type = string }
variable "region"       { type = string }
variable "env"          { type = string }
variable "domain"       { type = string }
variable "backend_neg"  { type = string }
variable "frontend_neg" { type = string }

# Global Static IP
resource "google_compute_global_address" "ip" {
  name    = "charitychain-${var.env}-lb-ip"
  project = var.project_id
}

# Managed SSL Certificate
resource "google_compute_managed_ssl_certificate" "cert" {
  name    = "charitychain-${var.env}-ssl-cert"
  project = var.project_id

  managed {
    domains = [
      var.domain,
      "www.${var.domain}",
      "api.${var.domain}",
    ]
  }
}

# Backend Service for API
resource "google_compute_backend_service" "api_backend" {
  name                  = "cc-api-backend-${var.env}"
  project               = var.project_id
  protocol              = "HTTPS"
  load_balancing_scheme = "EXTERNAL_MANAGED"

  backend {
    group = "projects/${var.project_id}/regions/${var.region}/networkEndpointGroups/${var.backend_neg}"
  }

  security_policy = google_compute_security_policy.edge_security.id
}

# Backend Service for Web Frontend
resource "google_compute_backend_service" "web_backend" {
  name                  = "cc-web-backend-${var.env}"
  project               = var.project_id
  protocol              = "HTTPS"
  load_balancing_scheme = "EXTERNAL_MANAGED"

  backend {
    group = "projects/${var.project_id}/regions/${var.region}/networkEndpointGroups/${var.frontend_neg}"
  }

  security_policy = google_compute_security_policy.edge_security.id
}

# URL Map (Host and path-based routing)
resource "google_compute_url_map" "url_map" {
  name            = "charitychain-${var.env}-url-map"
  project         = var.project_id
  default_service = google_compute_backend_service.web_backend.id

  host_rule {
    hosts        = ["api.${var.domain}"]
    path_matcher = "api-paths"
  }

  host_rule {
    hosts        = [var.domain, "www.${var.domain}"]
    path_matcher = "web-paths"
  }

  path_matcher {
    name            = "api-paths"
    default_service = google_compute_backend_service.api_backend.id
  }

  path_matcher {
    name            = "web-paths"
    default_service = google_compute_backend_service.web_backend.id

    path_rule {
      paths   = ["/api/*"]
      service = google_compute_backend_service.api_backend.id
    }
  }
}

# HTTPS Target Proxy
resource "google_compute_target_https_proxy" "https_proxy" {
  name             = "charitychain-${var.env}-https-proxy"
  project          = var.project_id
  url_map          = google_compute_url_map.url_map.id
  ssl_certificates = [google_compute_managed_ssl_certificate.cert.id]
}

# Global Forwarding Rule
resource "google_compute_global_forwarding_rule" "https_forwarding" {
  name                  = "charitychain-${var.env}-https-rule"
  project               = var.project_id
  target                = google_compute_target_https_proxy.https_proxy.id
  port_range            = "443"
  ip_address            = google_compute_global_address.ip.address
  load_balancing_scheme = "EXTERNAL_MANAGED"
}

# Cloud Armor Edge Security Policy (WAF / DDoS protection)
resource "google_compute_security_policy" "edge_security" {
  name        = "charitychain-${var.env}-cloud-armor"
  project     = var.project_id
  description = "Edge WAF security policy for CharityChain"

  # Default allow
  rule {
    action   = "allow"
    priority = "2147483647"
    match {
      versioned_expr = "SRC_IPS_V1"
      config {
        src_ip_ranges = ["*"]
      }
    }
    description = "Default allow rule"
  }

  # Rate limiting rule: max 200 requests per minute per IP
  rule {
    action   = "rate_based_ban"
    priority = "1000"
    match {
      versioned_expr = "SRC_IPS_V1"
      config {
        src_ip_ranges = ["*"]
      }
    }
    rate_limit_options {
      conform_action = "allow"
      exceed_action  = "deny(429)"
      rate_limit_threshold {
        count        = 200
        interval_sec = 60
      }
      ban_duration_sec = 300
    }
    description = "Rate limiting at edge"
  }
}

output "lb_ip_address" { value = google_compute_global_address.ip.address }

