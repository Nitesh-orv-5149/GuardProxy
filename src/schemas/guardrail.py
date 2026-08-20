from enum import Enum
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field

class InputGuardrailRequestPayload(BaseModel):
    text: str = Field(..., description="Content or prompt text to evaluate")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional execution context or user metadata"
    )

class GuardrailAction(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"
    MODIFY = "modify"

class GuardrailCheckResult(BaseModel):
    passed: bool = Field(..., description="Whether the guardrail check passed")
    score: float = Field(
        default=1.0,
        description="Safety score (1.0 = safe, 0.0 = dangerous)"
    )
    reason: str = Field(..., description="Reason for the score")
    action: GuardrailAction = Field(
        default=GuardrailAction.ALLOW,
        description="Action to take: allow, block, or modify"
    )

class TotalInputGuardrailResult(BaseModel):
    text: str = Field(..., description="Content or prompt text to evaluate")
    passed: bool = Field(..., description="Whether the guardrail check passed")
    score: float = Field(
        default=1.0,
        description="Overall safety score (1.0 = safe, 0.0 = dangerous)"
    )
    prompt_injection_result: GuardrailCheckResult = Field(..., description="Prompt injection guardrail result")
    jailbreak_result: GuardrailCheckResult = Field(..., description="Jailbreak guardrail result")
    pii_result: GuardrailCheckResult = Field(..., description="PII detection/masking guardrail result")
    toxic_content_result: GuardrailCheckResult = Field(..., description="Toxic content guardrail result")
    action: GuardrailAction = Field(
        default=GuardrailAction.ALLOW,
        description="Action to take: allow, block, or modify"
    )
    reason: Optional[str] = Field(
        default=None,
        description="Explanation if the content failed or flagged a rule"
    )
    modified_content: Optional[str] = Field(
        default=None,
        description="Sanitized or modified content if action is modify"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional evaluation metadata or diagnostic details"
    )
