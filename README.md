# GuardProxy

## What GuardProxy Does

**GuardProxy** is a lightweight FastAPI-based proxy framework designed to sit between client applications and Large Language Model (LLM) providers (such as openai, gemini, claude models). It intercepts prompt requests, provides hooks for validating input and output safety guardrails, and forwards requests to the upstream LLM backend.

## Documentation

Further documentation lives in [`docs/`](docs/):
- [Input Guardrails](docs/INPUT_GUARDRAILS.md) — how the input guardrail pipeline works
- [Test Prompts](docs/TEST_PROMPTS.md) — example blockable and allowable prompts for manual testing
- [Features & Roadmap](docs/TODO.md) — current capabilities and planned work

## How to Configure and Run

### Configuration

GuardProxy is configured using environment variables specified in a `.env` file located at the project root:

- `API_ENDPOINT`: The base URL of the upstream target LLM service.
- `MODEL_NAME`: The default model name to specify in upstream generation requests.
- `DRY_RUN`: When `true`, skips forwarding the prompt to the upstream LLM entirely. Useful for testing guardrails without an LLM backend running. Non-blocked requests return `{"blocked": false, "reason": ..., "response": "[LLM call skipped]"}` instead of a real completion.

Example `.env` file:
```env
API_ENDPOINT=http://host.docker.internal:11434
MODEL_NAME=llama3.1:8b
DRY_RUN=false
```

### Running the Application

#### Option 1: Docker Compose (Recommended)

To build and start the service with Docker Compose:

```bash
docker compose up --build
```

The API will be available at `http://localhost:8000`.

#### Option 2: Local Python Environment

1. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Start the development server using Uvicorn:
   ```bash
   uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
   ```

### Usage Example

Send a prompt request to the `/chat` endpoint:

```bash
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{"prompt": "Hello, world!"}'
```
