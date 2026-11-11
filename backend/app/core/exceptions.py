"""
exceptions.py - Application Custom Exception Hierarchy
======================================================
Defines domain and infrastructure exceptions with standard error codes and HTTP mappings.
"""

from typing import Optional, Dict, Any


class CharityChainException(Exception):
    """Base exception for all CharityChain application errors."""
    def __init__(self, message: str, code: str = "INTERNAL_SERVER_ERROR", status_code: int = 500, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code
        self.details = details or {}


class EntityNotFoundError(CharityChainException):
    def __init__(self, entity_name: str, entity_id: Any):
        super().__init__(
            message=f"{entity_name} with identifier '{entity_id}' not found.",
            code="ENTITY_NOT_FOUND",
            status_code=404,
            details={"entity": entity_name, "id": str(entity_id)}
        )


class InvalidStateTransitionError(CharityChainException):
    def __init__(self, message: str):
        super().__init__(
            message=message,
            code="INVALID_STATE_TRANSITION",
            status_code=400
        )


class UnauthorizedTransitionError(CharityChainException):
    def __init__(self, message: str):
        super().__init__(
            message=message,
            code="UNAUTHORIZED_TRANSITION",
            status_code=403
        )


class AuthenticationError(CharityChainException):
    def __init__(self, message: str = "Invalid credentials or expired session."):
        super().__init__(
            message=message,
            code="AUTHENTICATION_FAILED",
            status_code=401
        )


class PermissionDeniedError(CharityChainException):
    def __init__(self, message: str = "Insufficient permissions to perform this operation."):
        super().__init__(
            message=message,
            code="PERMISSION_DENIED",
            status_code=403
        )


class FinancialValidationError(CharityChainException):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            code="FINANCIAL_VALIDATION_ERROR",
            status_code=422,
            details=details
        )


class EvidenceValidationError(CharityChainException):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            code="EVIDENCE_VALIDATION_ERROR",
            status_code=400,
            details=details
        )


class ProviderUnavailableError(CharityChainException):
    def __init__(self, provider_name: str, reason: str):
        super().__init__(
            message=f"External provider '{provider_name}' is currently unavailable: {reason}",
            code="PROVIDER_UNAVAILABLE",
            status_code=503,
            details={"provider": provider_name, "reason": reason}
        )


class ConfigurationError(CharityChainException):
    def __init__(self, message: str):
        super().__init__(
            message=message,
            code="CONFIGURATION_ERROR",
            status_code=500
        )

