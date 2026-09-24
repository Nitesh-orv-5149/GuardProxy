# Features & Roadmap

#### Core & API Features:
- [x] FastAPI proxy backend core with `/chat` API endpoint
- [x] Forwarding prompt requests to upstream LLM backend
- [x] Environment variable configuration for target API endpoint and model name (`.env`)
- [x] Dockerfile and Docker Compose containerization setup
- [x] Input guardrail hook (`run_input_guardrails` in `/chat`)
- [ ] Output guardrail hook (`validate_output_guardrails` in `src/main.py` is an empty stub; `src/guardrails/output/*` are placeholders)
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
- [x] Low-latency regex engines & lightweight local classifier models (see [`docs/TRAINER.md`](TRAINER.md))
- [ ] Connection pooling and HTTP keep-alive optimization for upstream LLM requests

#### ML Classifier (input, see [`docs/TRAINER.md`](TRAINER.md)):
- [x] Three binary TF-IDF (word + char n-gram) + LogisticRegression heads trained via `python -m src.trainer.train`
- [x] Hot-swap model shipping via a `pointer.json` + versioned artifacts (no docker rebuild/restart)
- [x] Quality-gated shipping (per-head F1 on held-out split + block accuracy on a hand-written eval set)
- [x] Public dataset sources: `deepset/prompt-injections`, `jackhhao/jailbreak-classification`, `lmsys/toxic-chat`
- [ ] Improve the toxic head (F1 ≈ 0.63) and short/polite injection recall — see "Known limitations" in [`docs/TRAINER.md`](TRAINER.md)
- [ ] Replace `lmsys/toxic-chat` (CC-BY-NC-4.0) before any commercial use
- [ ] Training data sourced from live `/chat` traffic logs (needs request logging + a labeling workflow first)
- [ ] Scheduled/cron retraining trigger (manual CLI only for now)
- [ ] Output-guardrail ML classifier (input-only for now)
- [ ] Transformer-based upgrade path (kept out for now to protect inference latency)
#### Testing & Quality:
- [ ] Complete unit and integration test coverage
- [ ] Automated CI/CD build and test pipeline
