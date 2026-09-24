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
- [x] TF-IDF + LogisticRegression classifier trained via `python -m src.trainer.train`
- [x] Hot-swap model shipping via a `pointer.json` + versioned artifacts (no docker rebuild/restart)
- [x] Quality-gated shipping (won't ship a retrain below a minimum macro-F1)
- [x] Public dataset source (`BudEcosystem/guardrail-training-data` via HuggingFace `datasets`, prototype-scale sample)
- [ ] Training data sourced from live `/chat` traffic logs (needs request logging + a labeling workflow first)
- [ ] Scheduled/cron retraining trigger (manual CLI only for now)
- [ ] Output-guardrail ML classifier (input-only for now)
- [ ] Transformer-based upgrade path (kept out for now to protect inference latency)
- [ ] Improve shipped model quality: the current model (`v20260822-143004`) scores benign prompts as non-`safe` at ~0.4–0.48 confidence (passing only because they're under the 0.5 threshold), false-positives on "kill a python process", and misses paraphrased injection / DAN-style jailbreaks. See "Known limitations" in [`docs/TRAINER.md`](TRAINER.md).

#### Testing & Quality:
- [ ] Complete unit and integration test coverage
- [ ] Automated CI/CD build and test pipeline
