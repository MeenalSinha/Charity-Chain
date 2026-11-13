################################################################################
# CharityChain — Terraform: Networking Module
# Phase 14: VPC, Subnet, VPC Connector, Private IP range for Cloud SQL
################################################################################

variable "project_id"  { type = string }
variable "region"      { type = string }
variable "env"         { type = string }

resource "google_compute_network" "vpc" {
  name                    = "charitychain-${var.env}-vpc"
  project                 = var.project_id
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "app_subnet" {
  name          = "charitychain-${var.env}-app-subnet"
  project       = var.project_id
  region        = var.region
  network       = google_compute_network.vpc.id
  ip_cidr_range = "10.0.1.0/24"

  private_ip_google_access = true
}

# Private IP range for Cloud SQL (peered with Google Services)
resource "google_compute_global_address" "private_ip_range" {
  name          = "charitychain-${var.env}-db-ip-range"
  project       = var.project_id
  purpose       = "VPC_PEERING"
  address_type  = "INTERNAL"
  prefix_length = 20
  network       = google_compute_network.vpc.id
}

resource "google_service_networking_connection" "vpc_peering" {
  network                 = google_compute_network.vpc.id
  service                 = "servicenetworking.googleapis.com"
  reserved_peering_ranges = [google_compute_global_address.private_ip_range.name]
}

# VPC Connector for Cloud Run → VPC connectivity
resource "google_vpc_access_connector" "connector" {
  name          = "cc-${var.env}-connector"
  project       = var.project_id
  region        = var.region
  network       = google_compute_network.vpc.name
  ip_cidr_range = "10.8.0.0/28"
  min_throughput = 200
  max_throughput = 1000
}

output "vpc_self_link"      { value = google_compute_network.vpc.self_link }
output "db_ip_range_name"   { value = google_compute_global_address.private_ip_range.name }
output "vpc_connector_name" { value = google_vpc_access_connector.connector.name }
output "subnet_self_link"   { value = google_compute_subnetwork.app_subnet.self_link }

