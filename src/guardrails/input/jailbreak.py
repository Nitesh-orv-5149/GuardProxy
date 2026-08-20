import re
from src.schemas.guardrail import GuardrailCheckResult, GuardrailAction

JAILBREAK_PATTERNS = [
    r"\b(?:DAN|do\s+anything\s+now)\b",
    r"\bjailbreak(?:ed|ing)?\b",
    r"pretend\s+(?:you\s+have|to\s+have)\s+no\s+(?:rules|restrictions|filters|safety)",
    r"ignore\s+(?:all\s+)?(?:ethical|moral|safety)\s+(?:guidelines|constraints|policies)",
    r"hypothetical\s+scenario\s+where\s+(?:there\s+are\s+no|you\s+have\s+no)\s+(?:rules|laws|restrictions)",
    r"act\s+as\s+(?:an?\s+)?unfiltered\s+(?:ai|assistant|model)",
]

JAILBREAK_REGEX = re.compile("|".join(JAILBREAK_PATTERNS), re.IGNORECASE)


def check_jailbreak(text: str) -> GuardrailCheckResult:
    match = JAILBREAK_REGEX.search(text)
    if match:
        return GuardrailCheckResult(
            passed=False,
            score=0.0,
            reason=f"Jailbreak pattern detected: '{match.group(0)}'",
            action=GuardrailAction.BLOCK
        )

    return GuardrailCheckResult(
        passed=True,
        score=1.0,
        reason="No jailbreak pattern detected.",
        action=GuardrailAction.ALLOW
    )
