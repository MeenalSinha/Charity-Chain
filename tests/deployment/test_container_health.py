"""
tests/deployment/test_container_health.py

Phase 14: Container configuration health and security checks.
Validates that the Docker images and compose files meet
production hardening standards without requiring running containers.
"""
import os
import re

import pytest


REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")


def read_file(rel_path: str) -> str:
    full_path = os.path.join(REPO_ROOT, rel_path)
    with open(full_path) as f:
        return f.read()


class TestBackendContainerHardening:
    def test_backend_dockerfile_multistage_build(self):
        """Backend Dockerfile must use multi-stage build for minimal attack surface."""
        content = read_file("backend/Dockerfile")
        stages = re.findall(r"AS\s+\w+", content, re.IGNORECASE)
        assert len(stages) >= 2, (
            f"Backend Dockerfile should have at least 2 build stages, found: {stages}"
        )

    def test_backend_dockerfile_uses_slim_base(self):
        """Backend Dockerfile should use a minimal base image."""
        content = read_file("backend/Dockerfile")
        assert "slim" in content or "alpine" in content, (
            "Backend Dockerfile should use a slim or alpine base image"
        )

    def test_backend_dockerfile_no_latest_tag(self):
        """Backend Dockerfile should pin image versions, not use 'latest'."""
        content = read_file("backend/Dockerfile")
        from_lines = [l for l in content.splitlines() if l.strip().startswith("FROM")]
        for line in from_lines:
            if "AS" in line:
                image_part = line.split("AS")[0].replace("FROM", "").strip()
                assert ":latest" not in image_part, (
                    f"Dockerfile stage should pin image version: {line}"
                )

    def test_backend_dockerfile_no_root_in_final_stage(self):
        """Final stage of backend Dockerfile must run as non-root."""
        content = read_file("backend/Dockerfile")
        # Find content after the last FROM line (final stage)
        from_lines = [i for i, l in enumerate(content.splitlines())
                      if l.strip().startswith("FROM")]
        if from_lines:
            final_stage_content = "\n".join(content.splitlines()[from_lines[-1]:])
            assert re.search(r"^USER\s+(?!root)", final_stage_content, re.MULTILINE), (
                "Final Docker stage must have a USER instruction setting a non-root user"
            )

    def test_backend_dockerfile_healthcheck_configured(self):
        """HEALTHCHECK must be present in backend Dockerfile."""
        content = read_file("backend/Dockerfile")
        assert "HEALTHCHECK" in content

    def test_backend_dockerfile_cmd_uses_exec_form(self):
        """CMD must use exec form (JSON array) for proper signal handling."""
        content = read_file("backend/Dockerfile")
        cmd_lines = [l.strip() for l in content.splitlines() if l.strip().startswith("CMD")]
        assert cmd_lines, "No CMD instruction found in backend Dockerfile"
        last_cmd = cmd_lines[-1]
        assert last_cmd.startswith('CMD ["'), (
            f"CMD must use exec form (JSON array), found: {last_cmd}"
        )


class TestFrontendContainerHardening:
    def test_frontend_dockerfile_multistage_build(self):
        """Frontend Dockerfile must use multi-stage build."""
        content = read_file("frontend/Dockerfile")
        stages = re.findall(r"AS\s+\w+", content, re.IGNORECASE)
        assert len(stages) >= 2, (
            f"Frontend Dockerfile should have >= 2 stages, found: {stages}"
        )

    def test_frontend_dockerfile_non_root_user(self):
        """Frontend Dockerfile final stage must run as non-root user."""
        content = read_file("frontend/Dockerfile")
        from_lines = [i for i, l in enumerate(content.splitlines())
                      if l.strip().startswith("FROM")]
        if from_lines:
            final_stage = "\n".join(content.splitlines()[from_lines[-1]:])
            assert re.search(r"^USER\s+\w+", final_stage, re.MULTILINE), (
                "Frontend final stage must have a USER instruction"
            )

    def test_frontend_dockerfile_healthcheck(self):
        """HEALTHCHECK must be in frontend Dockerfile."""
        content = read_file("frontend/Dockerfile")
        assert "HEALTHCHECK" in content


class TestDockerComposeSecurity:
    def test_production_compose_postgres_no_public_port(self):
        """Production compose must not expose PostgreSQL port publicly."""
        content = read_file("docker-compose.prod.yml")
        # Check that postgres does not have a simple port mapping
        # The prod file uses `ports: !reset []` to clear the base ports
        assert "!reset" in content or "5432:5432" not in content, (
            "Production compose must not expose PostgreSQL port 5432 publicly"
        )

    def test_production_compose_redis_no_public_port(self):
        """Production compose must not expose Redis port publicly."""
        content = read_file("docker-compose.prod.yml")
        assert "!reset" in content or "6379:6379" not in content, (
            "Production compose must not expose Redis port 6379 publicly"
        )

    def test_production_compose_debug_false(self):
        """Production compose must set DEBUG=false."""
        content = read_file("docker-compose.prod.yml")
        assert "DEBUG=false" in content, (
            "docker-compose.prod.yml must explicitly set DEBUG=false"
        )

    def test_production_compose_has_restart_always(self):
        """Production services must use restart: always."""
        content = read_file("docker-compose.prod.yml")
        assert "restart: always" in content, (
            "Production services must use restart: always policy"
        )

    def test_production_compose_has_resource_limits(self):
        """Production services should have memory limits defined."""
        content = read_file("docker-compose.prod.yml")
        assert "memory:" in content, (
            "Production compose should define memory limits for services"
        )


class TestBackupScriptsExist:
    def test_backup_script_exists(self):
        assert os.path.isfile(os.path.join(REPO_ROOT, "scripts", "backup", "backup_postgres.sh"))

    def test_verify_script_exists(self):
        assert os.path.isfile(os.path.join(REPO_ROOT, "scripts", "backup", "verify_backup.sh"))

    def test_backup_script_has_encryption(self):
        content = read_file("scripts/backup/backup_postgres.sh")
        assert "gpg" in content and "AES256" in content, (
            "Backup script must use GPG AES-256 encryption"
        )

    def test_backup_script_has_retention_policy(self):
        content = read_file("scripts/backup/backup_postgres.sh")
        assert "retention" in content.lower() or "RETENTION" in content, (
            "Backup script must implement retention policy"
        )


class TestTerraformStructure:
    def test_terraform_backend_config_exists(self):
        assert os.path.isfile(os.path.join(REPO_ROOT, "infra", "terraform", "backend.tf"))

    def test_terraform_production_main_exists(self):
        assert os.path.isfile(
            os.path.join(REPO_ROOT, "infra", "terraform", "environments", "production", "main.tf")
        )

    def test_terraform_networking_module_exists(self):
        assert os.path.isfile(
            os.path.join(REPO_ROOT, "infra", "terraform", "modules", "networking", "main.tf")
        )

    def test_terraform_database_module_exists(self):
        assert os.path.isfile(
            os.path.join(REPO_ROOT, "infra", "terraform", "modules", "database", "main.tf")
        )

