"""
logging.py - Structured Application Logging & Secret Sanitization
================================================================
Provides contextual JSON or console logging with automatic credential masking.
"""

import os
import sys
import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from contextvars import ContextVar

# Context variables for tracing requests across async boundaries
correlation_id_ctx: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)
user_id_ctx: ContextVar[Optional[str]] = ContextVar("user_id", default=None)

SENSITIVE_KEYS = {
    "password", "secret", "token", "private_key", "authorization",
    "client_secret", "access_token", "refresh_token", "cvv", "card"
}


def sanitize_data(data: Any) -> Any:
    """Recursively masks sensitive values from log payloads."""
    if isinstance(data, dict):
        sanitized = {}
        for k, v in data.items():
            if any(s in k.lower() for s in SENSITIVE_KEYS):
                sanitized[k] = "***REDACTED***"
            else:
                sanitized[k] = sanitize_data(v)
        return sanitized
    elif isinstance(data, list):
        return [sanitize_data(item) for item in data]
    return data


class StructuredJsonFormatter(logging.Formatter):
    """Outputs standardized JSON log lines for observability pipelines."""
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": correlation_id_ctx.get(),
            "user_id": user_id_ctx.get(),
            "module": record.module,
            "line": record.lineno
        }
        if hasattr(record, "extra_data") and isinstance(record.extra_data, dict):
            log_entry["data"] = sanitize_data(record.extra_data)
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_entry)


def setup_logging(log_level: str = "INFO", log_format: str = "json") -> logging.Logger:
    """Initializes root logger with appropriate formatter."""
    root_logger = logging.getLogger("charitychain")
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    root_logger.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    if log_format.lower() == "json":
        handler.setFormatter(StructuredJsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(
            "[%(asctime)s] [%(levelname)s] [%(name)s] [%(module)s:%(lineno)d] %(message)s"
        ))
    
    root_logger.addHandler(handler)
    return root_logger


logger = setup_logging()


def get_logger(name: str = "charitychain") -> logging.Logger:
    """Helper to return named logger under the charitychain root hierarchy."""
    if name == "charitychain" or name.startswith("charitychain."):
        return logging.getLogger(name)
    return logging.getLogger(f"charitychain.{name}")


