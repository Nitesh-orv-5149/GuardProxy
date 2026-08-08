from fastapi import FastAPI, Request, Response
import httpx

app = FastAPI(title="Guardrail Proxy Framework")

TARGET_LLM_URL = "https://api.openai.com"

@app.on_event("startup")
async def startup_event():
    app.state.client = httpx.AsyncClient(base_url=TARGET_LLM_URL, timeout=60.0)

@app.on_event("shutdown")
async def shutdown_event():
    await app.state.client.aclose()

def validate_input_guardrails(body_bytes: bytes):
    pass

def validate_output_guardrails(response_content: bytes):
    pass

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_handler(request: Request, path: str):
    body = await request.body()

    validate_input_guardrails(body)

    headers = dict(request.headers)
    headers.pop("host", None) 

    client: httpx.AsyncClient = request.app.state.client
    upstream_response = await client.request(
        method=request.method,
        url=f"/{path}",
        headers=headers,
        params=dict(request.query_params),
        content=body,
    )

    validate_output_guardrails(upstream_response.content)

    response_headers = dict(upstream_response.headers)
    response_headers.pop("content-encoding", None)
    response_headers.pop("content-length", None)

    return Response(
        content=upstream_response.content,
        status_code=upstream_response.status_code,
        headers=response_headers,
    )
