"""
base.py - Abstract Notification Provider Interface
==================================================
Contract for transactional notification delivery (Email, SMS, In-App).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional


class NotificationProviderInterface(ABC):
    """Abstract interface decoupling notification mechanisms from domain workflows."""

    @abstractmethod
    async def send_email(
        self,
        recipient_email: str,
        subject: str,
        html_body: str,
        text_body: Optional[str] = None
    ) -> bool:
        """Dispatches transactional email."""
        pass

    @abstractmethod
    async def send_in_app_notification(
        self,
        user_id: str,
        title: str,
        message: str,
        data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Stores in-app alert for user."""
        pass

