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
    def __init__(self, entity_name: str, entity_id: Any = None):
        if entity_id is not None:
            msg = f"{entity_name} with identifier '{entity_id}' not found."
            ent_name = entity_name
            ent_id = str(entity_id)
        else:
            msg = entity_name
            ent_name = "Resource"
            ent_id = "unknown"
        super().__init__(
            message=msg,
            code="ENTITY_NOT_FOUND",
            status_code=404,
            details={"entity": ent_name, "id": ent_id}
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
    def __init__(self, message: str = "Invalid credentials or expired session.", code: str = "AUTHENTICATION_FAILED", status_code: int = 401, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            code=code,
            status_code=status_code,
            details=details
        )


class PermissionDeniedError(CharityChainException):
    def __init__(self, message: str = "Insufficient permissions to perform this operation.", code: str = "PERMISSION_DENIED", status_code: int = 403, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            code=code,
            status_code=status_code,
            details=details
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


class ConflictException(CharityChainException):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(
            message=message,
            code="CONFLICT",
            status_code=409,
            details=details
        )


class ValidationException(CharityChainException):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, status_code: int = 422):
        super().__init__(
            message=message,
            code="VALIDATION_ERROR",
            status_code=status_code,
            details=details
        )


class StorageException(CharityChainException):
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, status_code: int = 500):
        super().__init__(
            message=message,
            code="STORAGE_ERROR",
            status_code=status_code,
            details=details
        )


# Standard Aliases
AuthenticationException = AuthenticationError
ForbiddenException = PermissionDeniedError
NotFoundException = EntityNotFoundError
InvalidStateTransitionException = InvalidStateTransitionError
UnauthorizedTransitionException = UnauthorizedTransitionError
InternalServerException = CharityChainException

from backend.app.integrations.payments.exceptions import (
    CurrencyMismatchException, PaymentException, InvalidPaymentAmountException,
    PaymentCaptureFailedException, PaymentRefundFailedException,
    PaymentOrderNotFoundException, PaymentProviderException,
    PaymentAuthenticationException, WebhookVerificationFailedException
)


