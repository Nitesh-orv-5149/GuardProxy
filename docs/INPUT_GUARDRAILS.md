# Input Guardrails

The `/chat` endpoint runs every incoming prompt through the input guardrail
pipeline (`src/guardrails/input/pipeline.py::run_input_guardrails`) before it
is forwarded to the upstream LLM. If the pipeline blocks the prompt, the LLM
is never called.

## Checks

Four regex checks plus one ML classifier run concurrently via
`asyncio.gather` (each wrapped in `asyncio.to_thread`), so total guardrail
latency is roughly the slowest single check rather than the sum of all of
them.

| Check | Module | Action on match |
|---|---|---|
| Prompt injection | `prompt_injection.py` | `BLOCK` |
| Jailbreak / adversarial prompts | `jailbreak.py` | `BLOCK` |
| Toxic / policy-violating content | `toxic_content.py` | `BLOCK` |
| ML classifier (TF-IDF + LogisticRegression) | `ml_classifier.py` | `BLOCK` if non-`safe` with confidence ≥ `ML_CLASSIFIER_THRESHOLD`; `ALLOW` if no model is trained (see [TRAINER.md](TRAINER.md)) |
| PII (email, SSN, credit card, phone) | `pii_masker.py` | `MODIFY` (redacts, does not block) |

There used to be a separate `sensitive_data` check, but it used the exact
same patterns as `pii_masker`, so it was removed — PII is now consistently
masked instead of blocked.

## Decision logic

1. All five checks run against the original prompt text at the same time.
2. If any of the four blocking checks fire, the request is **blocked**.
   When more than one fires simultaneously, the surfaced reason/score is
   picked by priority order:
   `prompt_injection > jailbreak > toxic_content > ml_classifier`
   (`BLOCK_PRIORITY` in `pipeline.py`).
3. If nothing blocks but PII was found, the request is **allowed** with
   `action = MODIFY`, and the masked text (not the original) is what gets
   forwarded to the LLM.
4. If nothing matches, the request is **allowed** with `action = ALLOW`
   and the original text is forwarded unchanged.

The result of each check is returned in full in `TotalInputGuardrailResult`
(`src/schemas/guardrail.py`) regardless of which one caused the final
decision, so nothing is silently dropped.

## API behavior

- **Blocked**: `POST /chat` returns HTTP 200 with
  `{"blocked": true, "reason": "...", "response": "", "ml_classifier": {...}}`.
  The upstream LLM is never called.
- **Allowed**: the (possibly PII-masked) prompt is forwarded to the LLM as
  normal, and its response is returned as `{"response", "model", "ml_classifier"}`.
- Every response carries `ml_classifier: {action, reason, score}` so the
  classifier's verdict is visible even when it didn't decide the request.
- **`DRY_RUN=true`**: skips the LLM call for allowed prompts too, returning
  `{"blocked": false, "reason": "...", "response": "[LLM call skipped]"}`.
  Useful for exercising the guardrail logic without an LLM backend running
  (see the main [README](../README.md#configuration)).

## Testing

Unit tests for the pipeline live in `tests/unit/test_input_rules.py`;
end-to-end behavior (including `DRY_RUN`) is covered in
`tests/integration/test_proxy_flow.py`.
