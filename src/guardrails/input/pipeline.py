import asyncio

from src.schemas.guardrail import (
    GuardrailAction,
    GuardrailCheckResult,
    TotalInputGuardrailResult,
)
from .prompt_injection import check_prompt_injection
from .jailbreak import check_jailbreak
from .toxic_content import check_toxic_content
from .pii_masker import mask_pii

# Priority order used to pick which BLOCK reason surfaces when multiple
# checks fail at once (checks run concurrently, so no check short-circuits
# the others).
BLOCK_PRIORITY = ["prompt_injection", "jailbreak", "toxic_content"]


async def run_input_guardrails(text: str) -> TotalInputGuardrailResult:
    (
        injection_result,
        jailbreak_result,
        toxic_result,
        (masked_text, pii_result),
    ) = await asyncio.gather(
        asyncio.to_thread(check_prompt_injection, text),
        asyncio.to_thread(check_jailbreak, text),
        asyncio.to_thread(check_toxic_content, text),
        asyncio.to_thread(mask_pii, text),
    )

    results_by_name = {
        "prompt_injection": injection_result,
        "jailbreak": jailbreak_result,
        "toxic_content": toxic_result,
    }

    blocking = [
        results_by_name[name]
        for name in BLOCK_PRIORITY
        if results_by_name[name].action == GuardrailAction.BLOCK
    ]

    if blocking:
        winner: GuardrailCheckResult = blocking[0]
        passed = False
        score = winner.score
        action = GuardrailAction.BLOCK
        reason = winner.reason
    elif pii_result.action == GuardrailAction.MODIFY:
        passed = True
        score = pii_result.score
        action = GuardrailAction.MODIFY
        reason = pii_result.reason
    else:
        passed = True
        score = 1.0
        action = GuardrailAction.ALLOW
        reason = "No guardrail violations detected."

    return TotalInputGuardrailResult(
        text=text,
        passed=passed,
        score=score,
        prompt_injection_result=injection_result,
        jailbreak_result=jailbreak_result,
        pii_result=pii_result,
        toxic_content_result=toxic_result,
        action=action,
        reason=reason,
        modified_content=masked_text,
    )
