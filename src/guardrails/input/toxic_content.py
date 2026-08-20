import re
from src.schemas.guardrail import GuardrailCheckResult, GuardrailAction

TOXIC_CONTENT_PATTERNS = [
    r"\b(?:kill|harm|destroy)\s+(?:yourself|yourselves)\b",
    r"how\s+to\s+(?:make|build|synthesize|create)\s+(?:a\s+)?(?:bomb|explosive|chemical\s+weapon)",
    r"\b(?:hate\s+speech|violent\s+threat)\b",
    r"\b(?:suicide|self-harm)\s+(?:instructions|guide)\b",
]

TOXIC_CONTENT_REGEX = re.compile("|".join(TOXIC_CONTENT_PATTERNS), re.IGNORECASE)


def check_toxic_content(text: str) -> GuardrailCheckResult:
    match = TOXIC_CONTENT_REGEX.search(text)
    if match:
        return GuardrailCheckResult(
            passed=False,
            score=0.0,
            reason=f"Toxic or policy-violating content detected: '{match.group(0)}'",
            action=GuardrailAction.BLOCK
        )

    return GuardrailCheckResult(
        passed=True,
        score=1.0,
        reason="No toxic content detected.",
        action=GuardrailAction.ALLOW
    )
