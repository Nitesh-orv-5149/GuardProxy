import asyncio
import time

from src.schemas.guardrail import (
    GuardrailAction,
    GuardrailCheckResult,
    TotalInputGuardrailResult,
)
from .prompt_injection import check_prompt_injection
from .jailbreak import check_jailbreak
from .toxic_content import check_toxic_content
from .pii_masker import mask_pii
from .ml_classifier import check_ml_classifier

# Priority order used to pick which BLOCK reason surfaces when multiple
# checks fail at once (checks run concurrently, so no check short-circuits
# the others). The regex checks are exact-pattern matches and keep priority
# over the ML classifier, which is an additional net rather than a
# replacement for them.
BLOCK_PRIORITY = ["prompt_injection", "jailbreak", "toxic_content", "ml_classifier"]


def _timed(check, text: str) -> GuardrailCheckResult:
    start = time.perf_counter()
    result = check(text)
    result.latency_ms = (time.perf_counter() - start) * 1000
    return result


def _timed_mask_pii(text: str) -> tuple[str, GuardrailCheckResult]:
    start = time.perf_counter()
    masked_text, result = mask_pii(text)
    result.latency_ms = (time.perf_counter() - start) * 1000
    return masked_text, result


async def run_input_guardrails(text: str) -> TotalInputGuardrailResult:
    start = time.perf_counter()
    (
        injection_result,
        jailbreak_result,
        toxic_result,
        ml_result,
        (masked_text, pii_result),
    ) = await asyncio.gather(
        asyncio.to_thread(_timed, check_prompt_injection, text),
        asyncio.to_thread(_timed, check_jailbreak, text),
        asyncio.to_thread(_timed, check_toxic_content, text),
        asyncio.to_thread(_timed, check_ml_classifier, text),
        asyncio.to_thread(_timed_mask_pii, text),
    )
    latency_ms = (time.perf_counter() - start) * 1000
    regex_latency_ms = max(
        r.latency_ms for r in (injection_result, jailbreak_result, toxic_result, pii_result)
    )

    results_by_name = {
        "prompt_injection": injection_result,
        "jailbreak": jailbreak_result,
        "toxic_content": toxic_result,
        "ml_classifier": ml_result,
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
        ml_classifier_result=ml_result,
        regex_latency_ms=regex_latency_ms,
        latency_ms=latency_ms,
        action=action,
        reason=reason,
        modified_content=masked_text,
    )
