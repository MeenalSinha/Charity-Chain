"""
base.py - Abstract AI & Forensic Provider Interface
===================================================
Contract for multi-modal AI analysis (e.g. YOLO, Gemini 1.5 Pro, OpenAI GPT-4o).
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List


class AIProviderInterface(ABC):
    """Abstract interface decoupling AI and computer vision models from service logic."""

    @abstractmethod
    async def analyze_evidence(
        self,
        image_bytes: bytes,
        baseline_bytes: Optional[bytes] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Executes multi-modal forensic evaluation.
        Returns structured findings dictionary with individual check statuses.
        """
        pass

    @abstractmethod
    async def generate_findings_summary(
        self,
        verification_report: Dict[str, Any],
        project_details: Dict[str, Any]
    ) -> str:
        """Generates an explainable natural language summary for human reviewers."""
        pass

    @abstractmethod
    async def explain_verification_outcome(
        self,
        checks: Dict[str, Any],
        confidence_score: float
    ) -> Dict[str, Any]:
        """Provides transparency and explainability factors for review audits."""
        pass

