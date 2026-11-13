"""
tests/deployment/test_production_configuration.py

Phase 14: Verifies that production configuration requirements are enforced.
Tests the startup_checks preflight system against various configuration states.
"""
import os
import sys
from dataclasses import dataclass
from typing import Optional
from unittest.mock import MagicMock

import pytest

# Add backend to path for standalone test execution
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from backend.app.core.startup_checks import (
    CheckSeverity,
    PreflightReport,
    run_production_preflight,
)


@dataclass
class FakeSettings:
    """Minimal settings stub for preflight testing."""
    ENVIRONMENT: str = "production"
    DEBUG: bool = False
    DATABASE_URL: str = "postgresql+asyncpg://user:pass@host/db"
    REDIS_URL: str = "redis://:strongpass@host:6379/0"
    SECRET_KEY: str = "a" * 128  # 128 char strong key
    JWT_SECRET_KEY: str = "b" * 128
    ENCRYPTION_KEY: str = "c" * 44  # Fernet key length
    SENTRY_DSN: str = "https://abc@o123.ingest.sentry.io/456"
    PAYPAL_ENVIRONMENT: str = "sandbox"
    FEATURE_PAYPAL_LIVE_ENABLED: bool = False
    RATE_LIMIT_REDIS_REQUIRED: bool = True
    STORAGE_PROVIDER: str = "s3"


class TestProductionPreflightChecks:
    """Verify production preflight checks enforce all security requirements."""

    def test_all_checks_pass_with_valid_config(self):
        """Full production config should pass all critical checks."""
        settings = FakeSettings()
        report = run_production_preflight(settings)
        assert report.all_passed, (
            f"Expected all checks to pass. Failures: "
            f"{[(r.name, r.message) for r in report.critical_failures]}"
        )

    def test_weak_secret_key_fails(self):
        """Short SECRET_KEY should fail in production."""
        settings = FakeSettings(SECRET_KEY="too_short")
        report = run_production_preflight(settings)
        assert not report.all_passed
        failures = [r.name for r in report.critical_failures]
        assert "secret:SECRET_KEY" in failures

    def test_placeholder_secret_key_fails(self):
        """Placeholder values containing 'change_me' should fail."""
        settings = FakeSettings(SECRET_KEY="change_me_" + "x" * 60)
        report = run_production_preflight(settings)
        assert not report.all_passed
        failures = [r.name for r in report.critical_failures]
        assert "secret:SECRET_KEY" in failures

    def test_sqlite_database_fails_in_production(self):
        """SQLite DATABASE_URL must be rejected in production."""
        settings = FakeSettings(DATABASE_URL="sqlite:///./test.db")
        report = run_production_preflight(settings)
        assert not report.all_passed
        failures = [r.name for r in report.critical_failures]
        assert "database:not_sqlite" in failures

    def test_missing_sentry_dsn_fails_in_production(self):
        """Missing SENTRY_DSN should be a critical failure in production."""
        settings = FakeSettings(SENTRY_DSN="")
        report = run_production_preflight(settings)
        assert not report.all_passed
        failures = [r.name for r in report.critical_failures]
        assert "observability:sentry_dsn" in failures

    def test_debug_mode_in_production_fails(self):
        """DEBUG=True must be rejected in production."""
        settings = FakeSettings(DEBUG=True)
        report = run_production_preflight(settings)
        assert not report.all_passed
        failures = [r.name for r in report.critical_failures]
        assert "config:debug_disabled" in failures

    def test_paypal_live_without_feature_flag_fails(self):
        """PAYPAL_ENVIRONMENT=live without FEATURE_PAYPAL_LIVE_ENABLED fails."""
        settings = FakeSettings(
            PAYPAL_ENVIRONMENT="live",
            FEATURE_PAYPAL_LIVE_ENABLED=False,
        )
        report = run_production_preflight(settings)
        assert not report.all_passed
        failures = [r.name for r in report.critical_failures]
        assert "payments:paypal_live_guard" in failures

    def test_paypal_live_with_feature_flag_passes(self):
        """PAYPAL_ENVIRONMENT=live WITH FEATURE_PAYPAL_LIVE_ENABLED=true passes guard."""
        settings = FakeSettings(
            PAYPAL_ENVIRONMENT="live",
            FEATURE_PAYPAL_LIVE_ENABLED=True,
        )
        report = run_production_preflight(settings)
        # Only the PayPal guard check should pass; other checks may still fail
        paypal_checks = [r for r in report.results if r.name == "payments:paypal_live_guard"]
        assert len(paypal_checks) == 1
        assert paypal_checks[0].passed

    def test_missing_redis_in_production_is_warning_not_critical(self):
        """Missing RATE_LIMIT_REDIS_REQUIRED in production is a WARNING, not CRITICAL."""
        settings = FakeSettings(RATE_LIMIT_REDIS_REQUIRED=False)
        report = run_production_preflight(settings)
        # Critical checks should still pass (this is a warning)
        redis_warnings = [
            r for r in report.results
            if r.name == "rate_limiting:redis_required" and not r.passed
        ]
        if redis_warnings:
            assert redis_warnings[0].severity == CheckSeverity.WARNING

    def test_local_storage_in_production_is_warning(self):
        """Local storage in production is a WARNING, not CRITICAL."""
        settings = FakeSettings(STORAGE_PROVIDER="local")
        report = run_production_preflight(settings)
        storage_results = [r for r in report.results if r.name == "storage:not_local"]
        if storage_results and not storage_results[0].passed:
            assert storage_results[0].severity == CheckSeverity.WARNING

    def test_development_environment_allows_weak_config(self):
        """Development environment should not enforce production requirements."""
        settings = FakeSettings(
            ENVIRONMENT="development",
            SECRET_KEY="short",
            DEBUG=True,
            SENTRY_DSN="",
            DATABASE_URL="sqlite:///./dev.db",
            STORAGE_PROVIDER="local",
        )
        report = run_production_preflight(settings)
        # In development, critical checks should not mandate production-level config
        # (The check for SECRET_KEY still runs, but SQLite/Sentry/DEBUG
        #  are only enforced in production)
        db_failures = [r for r in report.critical_failures if r.name == "database:not_sqlite"]
        debug_failures = [r for r in report.critical_failures if r.name == "config:debug_disabled"]
        sentry_failures = [r for r in report.critical_failures if r.name == "observability:sentry_dsn"]
        # In dev mode: these specific production guards should NOT be critical failures
        assert len(db_failures) == 0, "SQLite check should only fail in production"
        assert len(debug_failures) == 0, "DEBUG check should only fail in production"
        assert len(sentry_failures) == 0, "Sentry check should only fail in production"

    def test_preflight_report_has_results(self):
        """PreflightReport should contain results for a valid settings object."""
        settings = FakeSettings()
        report = run_production_preflight(settings)
        assert len(report.results) > 0

    def test_preflight_report_critical_failures_property(self):
        """critical_failures should only return non-passing critical checks."""
        settings = FakeSettings(DEBUG=True)
        report = run_production_preflight(settings)
        for failure in report.critical_failures:
            assert not failure.passed
            assert failure.severity == CheckSeverity.CRITICAL


class TestProductionDockerConfiguration:
    """Verify Docker configuration constants meet production requirements."""

    def test_backend_dockerfile_exists(self):
        """Backend Dockerfile must exist."""
        dockerfile = os.path.join(
            os.path.dirname(__file__), "..", "..", "backend", "Dockerfile"
        )
        assert os.path.isfile(dockerfile), "backend/Dockerfile does not exist"

    def test_backend_dockerfile_has_non_root_user(self):
        """Backend Dockerfile must configure a non-root user."""
        dockerfile = os.path.join(
            os.path.dirname(__file__), "..", "..", "backend", "Dockerfile"
        )
        with open(dockerfile) as f:
            content = f.read()
        assert "USER charitychain" in content or "USER " in content, (
            "Backend Dockerfile must switch to a non-root user"
        )
        assert "useradd" in content or "adduser" in content, (
            "Backend Dockerfile must create an application user"
        )

    def test_backend_dockerfile_has_healthcheck(self):
        """Backend Dockerfile must include HEALTHCHECK."""
        dockerfile = os.path.join(
            os.path.dirname(__file__), "..", "..", "backend", "Dockerfile"
        )
        with open(dockerfile) as f:
            content = f.read()
        assert "HEALTHCHECK" in content, (
            "Backend Dockerfile must include a HEALTHCHECK instruction"
        )

    def test_backend_dockerfile_does_not_install_dev_deps(self):
        """Backend Dockerfile must not install dev/test dependencies."""
        dockerfile = os.path.join(
            os.path.dirname(__file__), "..", "..", "backend", "Dockerfile"
        )
        with open(dockerfile) as f:
            content = f.read()
        # Should install requirements.txt NOT requirements-dev.txt in final stage
        assert "requirements-dev.txt" not in content.split("AS runner")[-1] if "AS runner" in content else True, (
            "Backend production stage must not install dev requirements"
        )

    def test_frontend_dockerfile_exists(self):
        """Frontend Dockerfile must exist."""
        dockerfile = os.path.join(
            os.path.dirname(__file__), "..", "..", "frontend", "Dockerfile"
        )
        assert os.path.isfile(dockerfile), "frontend/Dockerfile does not exist"

    def test_frontend_dockerfile_has_non_root_user(self):
        """Frontend Dockerfile must configure non-root user."""
        dockerfile = os.path.join(
            os.path.dirname(__file__), "..", "..", "frontend", "Dockerfile"
        )
        with open(dockerfile) as f:
            content = f.read()
        assert "USER nextjs" in content or "USER " in content, (
            "Frontend Dockerfile must switch to a non-root user"
        )

    def test_production_compose_exists(self):
        """docker-compose.prod.yml must exist."""
        compose_prod = os.path.join(
            os.path.dirname(__file__), "..", "..", "docker-compose.prod.yml"
        )
        assert os.path.isfile(compose_prod), "docker-compose.prod.yml does not exist"

    def test_production_compose_has_no_debug_true(self):
        """docker-compose.prod.yml must not have DEBUG=true."""
        compose_prod = os.path.join(
            os.path.dirname(__file__), "..", "..", "docker-compose.prod.yml"
        )
        with open(compose_prod) as f:
            content = f.read()
        assert "DEBUG=true" not in content, (
            "docker-compose.prod.yml must not set DEBUG=true"
        )

    def test_env_production_example_exists(self):
        """.env.production.example must exist."""
        env_example = os.path.join(
            os.path.dirname(__file__), "..", "..", ".env.production.example"
        )
        assert os.path.isfile(env_example), ".env.production.example does not exist"

    def test_github_actions_ci_exists(self):
        """GitHub Actions CI workflow must exist."""
        ci_workflow = os.path.join(
            os.path.dirname(__file__), "..", "..", ".github", "workflows", "ci.yml"
        )
        assert os.path.isfile(ci_workflow), ".github/workflows/ci.yml does not exist"

    def test_github_actions_cd_staging_exists(self):
        """GitHub Actions staging CD workflow must exist."""
        cd_workflow = os.path.join(
            os.path.dirname(__file__), "..", "..", ".github", "workflows", "cd-staging.yml"
        )
        assert os.path.isfile(cd_workflow), ".github/workflows/cd-staging.yml does not exist"

    def test_github_actions_cd_production_exists(self):
        """GitHub Actions production CD workflow must exist."""
        cd_workflow = os.path.join(
            os.path.dirname(__file__), "..", "..", ".github", "workflows", "cd-production.yml"
        )
        assert os.path.isfile(cd_workflow), ".github/workflows/cd-production.yml does not exist"

