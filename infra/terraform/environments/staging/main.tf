################################################################################
# CharityChain — Terraform: Staging Environment
# Phase 14: Google Cloud Run staging deployment (scaled-down, sandbox)
################################################################################

terraform {
  required_version = ">= 1.6"

  backend "gcs" {
    bucket = "charitychain-terraform-state"
    prefix = "terraform/state/staging"
  }
}

provider "google" {
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
  env        = "staging"
  depends_on = [google_project_service.apis]
}

# --- Cloud SQL PostgreSQL (Single Zone, db-f1-micro for cost) ---
module "database" {
  source              = "../../modules/database"
  project_id          = var.project_id
  region              = var.region
  env                 = "staging"
  network_self_link   = module.networking.vpc_self_link
  allocated_ip_range  = module.networking.db_ip_range_name
  tier                = "db-custom-1-3840"
  availability_type   = "ZONAL"
  backup_enabled      = true
  deletion_protection = false
  depends_on          = [module.networking]
}

# --- Redis (Basic Tier) ---
module "redis" {
  source            = "../../modules/redis"
  project_id        = var.project_id
  region            = var.region
  env               = "staging"
  network_self_link = module.networking.vpc_self_link
  memory_size_gb    = 1
  tier              = "BASIC"
  depends_on        = [module.networking]
}

# --- Storage ---
module "storage" {
  source          = "../../modules/storage"
  project_id      = var.project_id
  region          = var.region
  env             = "staging"
  evidence_bucket = "charitychain-evidence-staging-${var.project_id}"
  backup_bucket   = "charitychain-backups-staging-${var.project_id}"
  lifecycle_days  = 90
}

# --- Secrets ---
module "secrets" {
  source           = "../../modules/secrets"
  project_id       = var.project_id
  env              = "staging"
  backend_sa_email = module.backend.service_account_email
}

# --- Cloud Run Backend ---
module "backend" {
  source             = "../../modules/backend"
  project_id         = var.project_id
  region             = var.region
  env                = "staging"
  image              = var.backend_image
  vpc_connector_name = module.networking.vpc_connector_name
  db_connection_name = module.database.connection_name
  redis_host         = module.redis.host
  redis_port         = module.redis.port
  evidence_bucket    = module.storage.evidence_bucket_name
  min_instances      = 1
  max_instances      = 3
  memory             = "512Mi"
  cpu                = "1"
  depends_on         = [module.database, module.redis, module.secrets]
}

# --- Cloud Run Frontend ---
module "frontend" {
  source        = "../../modules/frontend"
  project_id    = var.project_id
  region        = var.region
  env           = "staging"
  image         = var.frontend_image
  backend_url   = module.backend.service_url
  min_instances = 1
  max_instances = 3
  memory        = "512Mi"
  cpu           = "1"
  depends_on    = [module.backend]
}

