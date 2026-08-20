import re
from src.schemas.guardrail import GuardrailCheckResult, GuardrailAction

PROMPT_INJECTION_PATTERNS = [
    r"(?:ignore|disregard|forget|override|bypass)\s+(?:all\s+)?(?:previous|prior|above|system)\s+(?:instructions|directives|prompts|rules)",
    r"you\s+are\s+now\s+(?:a|an)?\s*(?:unfiltered|developer\s+mode|DAN|system\s+admin|root)",
    r"new\s+(?:system\s+)?(?:rule|instruction|directive|prompt)\s*:",
    r"(?:print|output|display|show|reveal)\s+(?:your|the)\s+(?:system\s+prompt|initial\s+instructions|developer\s+prompt)",
    r"(?:<\|im_start\|>system|\[system\]|\[system\s+prompt\]|<system>)",
]

PROMPT_INJECTION_REGEX = re.compile("|".join(PROMPT_INJECTION_PATTERNS), re.IGNORECASE)


def check_prompt_injection(text: str) -> GuardrailCheckResult:
    match = PROMPT_INJECTION_REGEX.search(text)
    if match:
        return GuardrailCheckResult(
            passed=False,
            score=0.0,
            reason=f"Prompt injection attempt detected: '{match.group(0)}'",
            action=GuardrailAction.BLOCK
        )

    return GuardrailCheckResult(
        passed=True,
        score=1.0,
        reason="No prompt injection detected.",
        action=GuardrailAction.ALLOW
    )