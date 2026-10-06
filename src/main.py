import asyncio
import time
from typing import Literal, Optional

from fastapi import FastAPI, Response
from pydantic import BaseModel, model_validator
import httpx
from src.config import settings
from src.guardrails import conversation
from src.guardrails.input import mask_pii, run_input_guardrails
from src.guardrails.input.model_registry import get_registry
from src.schemas.guardrail import GuardrailAction

app = FastAPI(title="Guardrail Proxy Framework")

TARGET_LLM_URL = settings.API_ENDPOINT
MODEL_NAME = settings.MODEL_NAME


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    """Either a single `prompt` (legacy) or a chat `messages` list ending in a
    user turn. `conversation_id` turns on the conversation monitor."""
    prompt: Optional[str] = None
    messages: Optional[list[Message]] = None
    conversation_id: Optional[str] = None

    @model_validator(mode="after")
    def _one_input(self):
        if (self.prompt is None) == (self.messages is None):
            raise ValueError("send exactly one of `prompt` or `messages`")
        if self.messages is not None and (not self.messages or self.messages[-1].role != "user"):
            raise ValueError("`messages` must end with a user message")
        return self

    @property
    def user_text(self) -> str:
        return self.prompt if self.prompt is not None else self.messages[-1].content


async def _timed_window(texts: list[str]) -> tuple[tuple[float, str], float]:
    start = time.perf_counter()
    window = await asyncio.to_thread(conversation.score_window, texts)
    return window, (time.perf_counter() - start) * 1000

@app.on_event("startup")
async def startup_event():
    app.state.client = httpx.AsyncClient(base_url=TARGET_LLM_URL, timeout=60.0)
    # Load the ML model now so the first request doesn't pay the ~1s joblib load.
    get_registry().get()

@app.on_event("shutdown")
async def shutdown_event():
    await app.state.client.aclose()

def validate_output_guardrails(response_content: bytes):
    pass

@app.get("/")
async def health_check():
    return {"status": "ok"}

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    # 1. Input guardrails on the latest user turn; the conversation window
    #    (proxy-held history + this turn) is scored concurrently.
    text = request.user_text
    if request.conversation_id:
        window_texts = conversation.window_texts(request.conversation_id, text)
        guardrail_result, (window, window_ms) = await asyncio.gather(
            run_input_guardrails(text), _timed_window(window_texts)
        )
        risk, head = conversation.turn_risk(guardrail_result)
        conversation_risk = conversation.record_turn(
            request.conversation_id, guardrail_result.modified_content, risk, head, window, window_ms
        )
    else:
        guardrail_result = await run_input_guardrails(text)
        conversation_risk = None

    ml_result = guardrail_result.ml_classifier_result
    ml_classifier_info = {
        "action": ml_result.action.value,
        "reason": ml_result.reason,
        "score": ml_result.score,
        "latency_ms": round(ml_result.latency_ms, 2),
        **ml_result.details,
    }
    regex_info = {"latency_ms": round(guardrail_result.regex_latency_ms, 2)}
    guardrail_latency_ms = round(guardrail_result.latency_ms, 2)

    if guardrail_result.action == GuardrailAction.BLOCK:
        return {
            "blocked": True,
            "reason": guardrail_result.reason,
            "response": "",
            "regex": regex_info,
            "ml_classifier": ml_classifier_info,
            "guardrail_latency_ms": guardrail_latency_ms,
            "conversation_risk": conversation_risk,
        }

    if settings.DRY_RUN:
        return {
            "blocked": False,
            "reason": guardrail_result.reason,
            "response": "[LLM call skipped]",
            "regex": regex_info,
            "ml_classifier": ml_classifier_info,
            "guardrail_latency_ms": guardrail_latency_ms,
            "conversation_risk": conversation_risk,
        }

    # 2. Forward to Ollama /api/chat: the checked turn as PII-masked, earlier
    #    user turns PII-masked too, system/assistant turns as sent.
    history = request.messages[:-1] if request.messages else []
    upstream_messages = [
        {"role": m.role, "content": mask_pii(m.content)[0] if m.role == "user" else m.content}
        for m in history
    ] + [{"role": "user", "content": guardrail_result.modified_content}]
    client: httpx.AsyncClient = app.state.client
    upstream_response = await client.post(
        "/api/chat",
        json={"model": MODEL_NAME, "messages": upstream_messages, "stream": False},
    )

    # 3. Run Output Guardrails on response
    validate_output_guardrails(upstream_response.content)

    # Parse response JSON and extract clean fields
    try:
        data = upstream_response.json()
        clean_response = {
            "response": (data.get("message") or {}).get("content", ""),
            "model": data.get("model", MODEL_NAME),
            "regex": regex_info,
            "ml_classifier": ml_classifier_info,
            "guardrail_latency_ms": guardrail_latency_ms,
            "conversation_risk": conversation_risk,
        }
        return clean_response
    except Exception:
        return Response(
            content=upstream_response.content,
            status_code=upstream_response.status_code,
            media_type="application/json"
        )
