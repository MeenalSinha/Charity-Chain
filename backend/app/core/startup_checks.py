"""
backend/app/core/startup_checks.py

Production preflight checks executed at application startup.
Phase 14: Validates all required secrets, connections, and configuration
before the application begins accepting traffic.

If any check fails in PRODUCTION environment, the application refuses to start.
"""
from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import List

logger = logging.getLogger(__name__)


class CheckSeverity(str, Enum):
    CRITICAL = "CRITICAL"   # Fail-fast: application must not start
    WARNING = "WARNING"     # Log warning but allow startup
    INFO = "INFO"           # Informational only


@dataclass
class CheckResult:
    name: str
    passed: bool
    severity: CheckSeverity
    message: str
    remediation: str = ""


@dataclass
class PreflightReport:
    results: List[CheckResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results if r.severity == CheckSeverity.CRITICAL)

    @property
    def critical_failures(self) -> List[CheckResult]:
        return [r for r in self.results if not r.passed and r.severity == CheckSeverity.CRITICAL]

    @property
    def warnings(self) -> List[CheckResult]:
        return [r for r in self.results if not r.passed and r.severity == CheckSeverity.WARNING]


def _check_secret_strength(name: str, value: str, min_length: int = 32) -> CheckResult:
    """Verify a secret meets minimum entropy requirements."""
    if not value:
        return CheckResult(
            name=f"secret:{name}",
            passed=False,
            severity=CheckSeverity.CRITICAL,
            message=f"{name} is not set",
            remediation=f"Set {name} to a cryptographically random value of at least {min_length} characters",
        )
    if len(value) < min_length:
        return CheckResult(
            name=f"secret:{name}",
            passed=False,
            severity=CheckSeverity.CRITICAL,
            message=f"{name} is too short ({len(value)} chars, minimum {min_length})",
            remediation=f"Regenerate {name} with at least {min_length} random characters",
        )
    # Detect placeholder/default values
    weak_patterns = [
        "change_me", "replace", "example", "default", "secret", "password",
        "00000000", "xxxxxxxx", "staging-secret", "staging-jwt",
    ]
    lower_val = value.lower()
    for pattern in weak_patterns:
        if pattern in lower_val:
            return CheckResult(
                name=f"secret:{name}",
                passed=False,
                severity=CheckSeverity.CRITICAL,
                message=f"{name} appears to be a placeholder (contains '{pattern}')",
                remediation=f"Replace {name} with a genuinely random secret",
            )
    return CheckResult(
        name=f"secret:{name}",
        passed=True,
        severity=CheckSeverity.CRITICAL,
        message=f"{name} is set and meets minimum length",
    )


def run_production_preflight(settings) -> PreflightReport:
    """
    Run all production preflight checks.

    Args:
        settings: The application settings object (backend.app.core.config.Settings)

    Returns:
        PreflightReport with all check results
    """
    report = PreflightReport()
    is_production = getattr(settings, "ENVIRONMENT", "development") == "production"

    # ----------------------------------------------------------------
    # Check 1: SECRET_KEY strength
    # ----------------------------------------------------------------
    report.results.append(
        _check_secret_strength("SECRET_KEY", getattr(settings, "SECRET_KEY", ""), min_length=64)
    )

    # ----------------------------------------------------------------
    # Check 2: JWT_SECRET_KEY strength
    # ----------------------------------------------------------------
    report.results.append(
        _check_secret_strength("JWT_SECRET_KEY", getattr(settings, "JWT_SECRET_KEY", ""), min_length=64)
    )

    # ----------------------------------------------------------------
    # Check 3: ENCRYPTION_KEY (Fernet key)
    # ----------------------------------------------------------------
    encryption_key = getattr(settings, "ENCRYPTION_KEY", "")
    if is_production:
        report.results.append(
            _check_secret_strength("ENCRYPTION_KEY", encryption_key, min_length=44)
        )

    # ----------------------------------------------------------------
    # Check 4: DATABASE_URL set and not SQLite in production
    # ----------------------------------------------------------------
    db_url = str(getattr(settings, "DATABASE_URL", ""))
    if is_production and "sqlite" in db_url.lower():
        report.results.append(CheckResult(
            name="database:not_sqlite",
            passed=False,
            severity=CheckSeverity.CRITICAL,
            message="SQLite database detected in production environment",
            remediation="Set DATABASE_URL to a PostgreSQL connection string",
        ))
    elif db_url:
        report.results.append(CheckResult(
            name="database:url_set",
            passed=True,
            severity=CheckSeverity.CRITICAL,
            message="DATABASE_URL is configured",
        ))
    else:
        report.results.append(CheckResult(
            name="database:url_set",
            passed=False,
            severity=CheckSeverity.CRITICAL,
            message="DATABASE_URL is not set",
            remediation="Set DATABASE_URL to a PostgreSQL connection string",
        ))

    # ----------------------------------------------------------------
    # Check 5: Sentry DSN required in production
    # ----------------------------------------------------------------
    sentry_dsn = getattr(settings, "SENTRY_DSN", "")
    if is_production:
        if not sentry_dsn:
            report.results.append(CheckResult(
                name="observability:sentry_dsn",
                passed=False,
                severity=CheckSeverity.CRITICAL,
                message="SENTRY_DSN is not set in production",
                remediation="Configure SENTRY_DSN with a valid Sentry project DSN",
            ))
        else:
            report.results.append(CheckResult(
                name="observability:sentry_dsn",
                passed=True,
                severity=CheckSeverity.CRITICAL,
                message="SENTRY_DSN is configured",
            ))

    # ----------------------------------------------------------------
    # Check 6: PayPal live mode safety guard
    # ----------------------------------------------------------------
    paypal_env = getattr(settings, "PAYPAL_ENVIRONMENT", "sandbox")
    feature_live = getattr(settings, "FEATURE_PAYPAL_LIVE_ENABLED", False)
    if paypal_env == "live" and not feature_live:
        report.results.append(CheckResult(
            name="payments:paypal_live_guard",
            passed=False,
            severity=CheckSeverity.CRITICAL,
            message="PAYPAL_ENVIRONMENT=live but FEATURE_PAYPAL_LIVE_ENABLED is not explicitly set to true",
            remediation="Set FEATURE_PAYPAL_LIVE_ENABLED=true only after explicit authorization",
        ))
    else:
        report.results.append(CheckResult(
            name="payments:paypal_live_guard",
            passed=True,
            severity=CheckSeverity.CRITICAL,
            message=f"PayPal environment: {paypal_env} (live guard: {feature_live})",
        ))

    # ----------------------------------------------------------------
    # Check 7: DEBUG must be false in production
    # ----------------------------------------------------------------
    debug_mode = getattr(settings, "DEBUG", False)
    if is_production and debug_mode:
        report.results.append(CheckResult(
            name="config:debug_disabled",
            passed=False,
            severity=CheckSeverity.CRITICAL,
            message="DEBUG=true in production environment",
            remediation="Set DEBUG=false in production",
        ))
    else:
        report.results.append(CheckResult(
            name="config:debug_disabled",
            passed=True,
            severity=CheckSeverity.CRITICAL,
            message=f"DEBUG={debug_mode} (environment={getattr(settings, 'ENVIRONMENT', 'development')})",
        ))

    # ----------------------------------------------------------------
    # Check 8: Redis required for rate limiting in production
    # ----------------------------------------------------------------
    redis_required = getattr(settings, "RATE_LIMIT_REDIS_REQUIRED", False)
    redis_url = str(getattr(settings, "REDIS_URL", ""))
    if is_production and not redis_required:
        report.results.append(CheckResult(
            name="rate_limiting:redis_required",
            passed=False,
            severity=CheckSeverity.WARNING,
            message="RATE_LIMIT_REDIS_REQUIRED is not set — in-memory fallback will not work with multiple replicas",
            remediation="Set RATE_LIMIT_REDIS_REQUIRED=true and ensure Redis is available",
        ))
    else:
        report.results.append(CheckResult(
            name="rate_limiting:redis_required",
            passed=True,
            severity=CheckSeverity.WARNING,
            message=f"Redis rate limiting: required={redis_required}",
        ))

    # ----------------------------------------------------------------
    # Check 9: Local storage in production warning
    # ----------------------------------------------------------------
    storage_provider = getattr(settings, "STORAGE_PROVIDER", "local")
    if is_production and storage_provider == "local":
        report.results.append(CheckResult(
            name="storage:not_local",
            passed=False,
            severity=CheckSeverity.WARNING,
            message="Local filesystem storage detected in production — data will be lost on container restart",
            remediation="Set STORAGE_PROVIDER=s3 and configure S3_BUCKET_NAME and credentials",
        ))
    else:
        report.results.append(CheckResult(
            name="storage:not_local",
            passed=True,
            severity=CheckSeverity.WARNING,
            message=f"Storage provider: {storage_provider}",
        ))

    return report


def enforce_preflight(settings) -> None:
    """
    Run preflight checks and abort if critical failures exist in production.

    Call this from the application lifespan startup handler.
    """
    report = run_production_preflight(settings)
    is_production = getattr(settings, "ENVIRONMENT", "development") == "production"

    # Log all results
    for result in report.results:
        level = logging.INFO if result.passed else (
            logging.CRITICAL if result.severity == CheckSeverity.CRITICAL else logging.WARNING
        )
        logger.log(level, "[PREFLIGHT] %s: %s — %s",
                   result.name,
                   "PASS" if result.passed else "FAIL",
                   result.message)
        if not result.passed and result.remediation:
            logger.log(level, "[PREFLIGHT] Remediation: %s", result.remediation)

    # In production, abort on any critical failure
    if is_production and report.critical_failures:
        logger.critical(
            "[PREFLIGHT] %d critical check(s) failed — refusing to start in production",
            len(report.critical_failures)
        )
        for failure in report.critical_failures:
            logger.critical("[PREFLIGHT] CRITICAL FAILURE: %s — %s", failure.name, failure.message)
        sys.exit(1)

    if report.warnings:
        logger.warning(
            "[PREFLIGHT] %d warning(s) — application started but configuration should be reviewed",
            len(report.warnings)
        )

    logger.info("[PREFLIGHT] All critical checks passed (%d total checks)", len(report.results))

