import re
from typing import Tuple
from src.schemas.guardrail import GuardrailCheckResult, GuardrailAction

PII_PATTERNS = {
    "EMAIL": (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "[REDACTED_EMAIL]"),
    "SSN": (r"\b\d{3}-\d{2}-\d{4}\b", "[REDACTED_SSN]"),
    "CREDIT_CARD": (r"\b(?:4\d{12}(?:\d{3})?|5[1-5]\d{14}|3[47]\d{13})\b", "[REDACTED_CREDIT_CARD]"),
    "PHONE": (r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "[REDACTED_PHONE]"),
}


def mask_pii(text: str) -> Tuple[str, GuardrailCheckResult]:
    masked_text = text
    matches_found = []

    for key, (pattern, replacement) in PII_PATTERNS.items():
        found = re.findall(pattern, masked_text, re.IGNORECASE)
        if found:
            matches_found.extend([f"{key}: '{item}'" for item in found])
            masked_text = re.sub(pattern, replacement, masked_text, flags=re.IGNORECASE)

    if matches_found:
        result = GuardrailCheckResult(
            passed=False,
            score=0.5,
            reason=f"PII detected and redacted: {', '.join(matches_found)}",
            action=GuardrailAction.MODIFY
        )
    else:
        result = GuardrailCheckResult(
            passed=True,
            score=1.0,
            reason="No PII detected.",
            action=GuardrailAction.ALLOW
        )

    return masked_text, result
