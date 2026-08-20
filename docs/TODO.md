# Features & Roadmap

#### Core & API Features:
- [x] FastAPI proxy backend core with `/chat` API endpoint
- [x] Forwarding prompt requests to upstream LLM backend
- [x] Environment variable configuration for target API endpoint and model name (`.env`)
- [x] Dockerfile and Docker Compose containerization setup
- [x] Architecture hooks for input and output guardrail processing
- [x] `DRY_RUN` mode to test guardrails without a live upstream LLM
- [ ] Response streaming support (`stream: true`)
- [ ] Dynamic model overrides per request payload
- [ ] Multi-provider API routing (OpenAI, Gemini, Claude, Ollama)
- [ ] Rate limiting, logging, and API authentication

#### Input Guardrails:
- [x] Prompt injection detection
- [x] Jailbreak & adversarial prompt detection
- [x] PII detection / redaction (masking, non-blocking)
- [x] Content policy violation filtering (toxic content)

#### Output Guardrails:
- [ ] Content safety & toxicity filtering
- [ ] Data leakage prevention (credentials, secrets, PII)
- [ ] Behavioral guardrails
- [ ] Response schema validation

#### Performance Improvements:
- [x] Asynchronous parallel guardrail execution pipeline
- [ ] Guardrail evaluation caching (In-memory / Redis prompt-response caching)
- [ ] Low-latency regex engines & lightweight local classifier models
- [ ] Connection pooling and HTTP keep-alive optimization for upstream LLM requests

#### Testing & Quality:
- [ ] Complete unit and integration test coverage
- [ ] Automated CI/CD build and test pipeline
