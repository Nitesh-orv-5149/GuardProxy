import os
from fastapi import FastAPI, Response
from pydantic import BaseModel
import httpx
from src.config import settings
from src.guardrails.input import run_input_guardrails
from src.guardrails.input.model_registry import get_registry
from src.schemas.guardrail import GuardrailAction

app = FastAPI(title="Guardrail Proxy Framework")

TARGET_LLM_URL = settings.API_ENDPOINT
MODEL_NAME = settings.MODEL_NAME

class ChatRequest(BaseModel):
    prompt: str

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
    # 1. Run Input Guardrails on prompt
    guardrail_result = await run_input_guardrails(request.prompt)

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
        }

    if settings.DRY_RUN:
        return {
            "blocked": False,
            "reason": guardrail_result.reason,
            "response": "[LLM call skipped]",
            "regex": regex_info,
            "ml_classifier": ml_classifier_info,
            "guardrail_latency_ms": guardrail_latency_ms,
        }

    # 2. Forward (PII-masked) prompt to Ollama /api/generate using default MODEL_NAME
    client: httpx.AsyncClient = app.state.client
    upstream_response = await client.post(
        "/api/generate",
        json={
            "model": MODEL_NAME,
            "prompt": guardrail_result.modified_content,
            "stream": False
        }
    )

    # 3. Run Output Guardrails on response
    validate_output_guardrails(upstream_response.content)

    # Parse response JSON and extract clean fields
    try:
        data = upstream_response.json()
        clean_response = {
            "response": data.get("response", ""),
            "model": data.get("model", MODEL_NAME),
            "regex": regex_info,
            "ml_classifier": ml_classifier_info,
            "guardrail_latency_ms": guardrail_latency_ms,
        }
        return clean_response
    except Exception:
        return Response(
            content=upstream_response.content,
            status_code=upstream_response.status_code,
            media_type="application/json"
        )
