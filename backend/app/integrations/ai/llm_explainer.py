"""
llm_explainer.py - Grounded Explainable AI (XAI) Summary Generator
===================================================================
Produces explainable natural language findings for human reviewers.
Strictly grounded in deterministic computer-vision and forensic measurements.
Enforces prompt-injection defenses and structured output schemas.
Falls back safely to rule-based synthesis if external LLM APIs are offline or disabled.
"""

import re
import json
from typing import Dict, Any, List, Optional
from backend.app.core.config import settings
from backend.app.core.constants import VerificationCheckStatus
from backend.app.core.logging import logger
from backend.app.integrations.ai.schemas import LlmExplanationResult, SingleCheckResult


class GroundedLlmExplainer:
    """Generates grounded explainability summaries from technical check records."""

    def __init__(self, provider: Optional[str] = None, model_name: Optional[str] = None):
        self.provider = provider or settings.LLM_PROVIDER or "none"
        self.model_name = model_name or settings.LLM_MODEL_NAME

    @staticmethod
    def _sanitize_untrusted_text(text: str) -> str:
        """Sanitizes user-supplied text to prevent prompt injection."""
        if not text:
            return ""
        # Strip potential system instruction hijacking tokens
        clean = re.sub(r"(ignore previous instructions|system prompt|you are now|override|assistant:)", "[FILTERED]", text, flags=re.IGNORECASE)
        return clean[:500]

    def generate_explanation(
        self,
        checks: List[SingleCheckResult],
        overall_assessment: str,
        project_title: str,
        milestone_title: str
    ) -> LlmExplanationResult:
        """
        Synthesizes technical forensic results into an explainable reviewer brief.
        Uses OpenAI/Gemini if configured with valid API keys; otherwise uses deterministic synthesis.
        """
        sanitized_proj = self._sanitize_untrusted_text(project_title)
        sanitized_ms = self._sanitize_untrusted_text(milestone_title)

        # If LLM provider is enabled and configured with OpenAI
        if self.provider == "openai" and settings.OPENAI_API_KEY:
            try:
                import openai
                client = openai.OpenAI(api_key=settings.OPENAI_API_KEY, timeout=settings.LLM_TIMEOUT_SECONDS)
                prompt_payload = {
                    "project": sanitized_proj,
                    "milestone": sanitized_ms,
                    "overall_assessment": overall_assessment,
                    "checks": [
                        {
                            "type": c.check_type,
                            "status": c.status.value if hasattr(c.status, "value") else str(c.status),
                            "passed": c.passed,
                            "message": c.message
                        }
                        for c in checks
                    ]
                }
                system_instruction = (
                    "You are the CharityChain Verification Explainer. Your duty is to explain "
                    "the provided technical forensic checks to a human grant reviewer in clear, objective language. "
                    "CRITICAL GROUNDING RULES:\n"
                    "1. Only discuss facts provided in the JSON payload.\n"
                    "2. NEVER invent object detections, GPS distances, or forensic measurements.\n"
                    "3. Respond strictly in JSON format with keys: 'summary_text', 'key_findings', 'anomalies_flagged', 'recommended_reviewer_questions'."
                )
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_instruction},
                        {"role": "user", "content": json.dumps(prompt_payload)}
                    ],
                    response_format={"type": "json_object"},
                    max_tokens=600,
                    temperature=0.1
                )
                content_str = response.choices[0].message.content or "{}"
                data = json.loads(content_str)
                return LlmExplanationResult(
                    status=VerificationCheckStatus.PASSED,
                    passed=True,
                    provider_name="openai",
                    model_name=self.model_name,
                    summary_text=data.get("summary_text", "Explanation generated."),
                    key_findings=data.get("key_findings", []),
                    anomalies_flagged=data.get("anomalies_flagged", []),
                    recommended_reviewer_questions=data.get("recommended_reviewer_questions", []),
                    prompt_injection_detected=False,
                    message="LLM explanation successfully generated."
                )
            except Exception as e:
                logger.warning(f"OpenAI LLM explanation call failed: {e}. Falling back to deterministic synthesis.")

        # Deterministic Grounded Synthesis (Safe Fallback / Default)
        return self._generate_deterministic_explanation(checks, overall_assessment, sanitized_proj, sanitized_ms)

    def _generate_deterministic_explanation(
        self,
        checks: List[SingleCheckResult],
        overall_assessment: str,
        project_title: str,
        milestone_title: str
    ) -> LlmExplanationResult:
        """Deterministic rule-based explainability synthesis guaranteeing 100% grounding and reliability."""
        passed_checks = [c for c in checks if c.status == VerificationCheckStatus.PASSED]
        flagged_checks = [c for c in checks if c.status == VerificationCheckStatus.FLAGGED]
        unavailable_checks = [c for c in checks if c.status == VerificationCheckStatus.UNAVAILABLE]
        not_applicable_checks = [c for c in checks if c.status == VerificationCheckStatus.NOT_APPLICABLE]

        findings: List[str] = []
        anomalies: List[str] = []
        questions: List[str] = []

        for c in checks:
            if c.status == VerificationCheckStatus.PASSED:
                findings.append(f"{c.check_type}: {c.message}")
            elif c.status == VerificationCheckStatus.FLAGGED:
                anomalies.append(f"{c.check_type}: {c.message}")
                if "ELA" in c.check_type:
                    questions.append("Examine whether image compression variance is caused by re-saving or intentional localized manipulation.")
                if "GPS" in c.check_type:
                    questions.append("Verify if evidence was recorded off-site or if project location coordinates require correction.")
                if "PROGRESS" in c.check_type:
                    questions.append("Inspect whether the submitted image represents genuine new work or an unedited duplicate of baseline evidence.")
            elif c.status == VerificationCheckStatus.UNAVAILABLE:
                findings.append(f"{c.check_type} omitted: {c.message}")
                if "GPS" in c.check_type or "EXIF" in c.check_type:
                    questions.append("Request original un-scrubbed camera files if EXIF GPS verification is necessary for this milestone.")

        summary_parts = [
            f"Verification assessment for '{milestone_title}' under project '{project_title}' concluded as {overall_assessment}.",
            f"{len(passed_checks)} check(s) passed, {len(flagged_checks)} anomaly indicator(s) flagged, and {len(unavailable_checks)} check(s) were unavailable."
        ]
        if anomalies:
            summary_parts.append(f"Primary investigation areas: {'; '.join(anomalies)}.")
        else:
            summary_parts.append("No technical tampering or visual inconsistencies were flagged by executed forensic modules.")

        summary_text = " ".join(summary_parts)

        return LlmExplanationResult(
            status=VerificationCheckStatus.PASSED,
            passed=True,
            provider_name="deterministic_rules",
            model_name="charitychain_xai_v1",
            summary_text=summary_text,
            key_findings=findings[:6],
            anomalies_flagged=anomalies,
            recommended_reviewer_questions=questions or ["Confirm that visual evidence matches submitted expenditure narrative."],
            prompt_injection_detected=False,
            message="Deterministic grounded explanation generated successfully."
        )

