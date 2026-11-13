################################################################################
# CharityChain — Terraform: Production Environment
# Phase 14: Google Cloud Run deployment
################################################################################

terraform {
  required_version = ">= 1.6"

  backend "gcs" {
    bucket = "charitychain-terraform-state"
    prefix = "terraform/state/production"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
}

# --- Enable Required APIs ---
resource "google_project_service" "apis" {
  for_each = toset([
    "run.googleapis.com",
    "sqladmin.googleapis.com",
    "redis.googleapis.com",
    "secretmanager.googleapis.com",
    "storage.googleapis.com",
    "containerregistry.googleapis.com",
    "vpcaccess.googleapis.com",
    "cloudresourcemanager.googleapis.com",
    "monitoring.googleapis.com",
    "logging.googleapis.com",
    "cloudtrace.googleapis.com",
    "servicenetworking.googleapis.com",
  ])
  service            = each.value
  disable_on_destroy = false
}

# --- Networking ---
module "networking" {
  source     = "../../modules/networking"
  project_id = var.project_id
  region     = var.region
  env        = "production"
  depends_on = [google_project_service.apis]
}

# --- Cloud SQL PostgreSQL ---
module "database" {
  source              = "../../modules/database"
  project_id          = var.project_id
  region              = var.region
  env                 = "production"
  network_self_link   = module.networking.vpc_self_link
  allocated_ip_range  = module.networking.db_ip_range_name
  tier                = "db-custom-2-7680"   # 2 vCPU, 7.5 GB RAM
  availability_type   = "REGIONAL"            # High availability
  backup_enabled      = true
  backup_start_time   = "03:00"
  deletion_protection = true
  depends_on          = [module.networking]
}

# --- Redis (Memorystore) ---
module "redis" {
  source            = "../../modules/redis"
  project_id        = var.project_id
  region            = var.region
  env               = "production"
  network_self_link = module.networking.vpc_self_link
  memory_size_gb    = 2
  tier              = "STANDARD_HA"           # High availability
  depends_on        = [module.networking]
}

# --- GCS Storage Buckets ---
module "storage" {
  source         = "../../modules/storage"
  project_id     = var.project_id
  region         = var.region
  env            = "production"
  evidence_bucket = "charitychain-evidence-prod-${var.project_id}"
  backup_bucket  = "charitychain-backups-prod-${var.project_id}"
  lifecycle_days = 2555  # 7 years retention for evidence
}

# --- Secret Manager ---
module "secrets" {
  source     = "../../modules/secrets"
  project_id = var.project_id
  env        = "production"
  backend_sa_email = module.backend.service_account_email
}

# --- Cloud Run Backend ---
module "backend" {
  source              = "../../modules/backend"
  project_id          = var.project_id
  region              = var.region
  env                 = "production"
  image               = var.backend_image
  vpc_connector_name  = module.networking.vpc_connector_name
  db_connection_name  = module.database.connection_name
  redis_host          = module.redis.host
  redis_port          = module.redis.port
  evidence_bucket     = module.storage.evidence_bucket_name
  min_instances       = 2
  max_instances       = 20
  memory              = "1Gi"
  cpu                 = "2"
  depends_on          = [module.database, module.redis, module.secrets]
}

# --- Cloud Run Frontend ---
module "frontend" {
  source             = "../../modules/frontend"
  project_id         = var.project_id
  region             = var.region
  env                = "production"
  image              = var.frontend_image
  backend_url        = module.backend.service_url
  min_instances      = 2
  max_instances      = 10
  memory             = "512Mi"
  cpu                = "1"
  depends_on         = [module.backend]
}

# --- Load Balancer ---
module "load_balancing" {
  source           = "../../modules/load_balancing"
  project_id       = var.project_id
  region           = var.region
  env              = "production"
  domain           = var.domain
  backend_neg      = module.backend.neg_name
  frontend_neg     = module.frontend.neg_name
  depends_on       = [module.backend, module.frontend]
}

# --- Cloud Monitoring ---
module "monitoring" {
  source     = "../../modules/monitoring"
  project_id = var.project_id
  region     = var.region
  env        = "production"
  alert_email = var.alert_email
}

