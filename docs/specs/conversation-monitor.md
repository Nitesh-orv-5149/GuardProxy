# Conversation Monitor — multi-turn breach detection

**Status:** draft · **Owner:** Nitesh Kumar · **Date:** 2026-10-05 · **v1 response:** flag only

Every GuardProxy check judges one prompt in isolation. The Conversation
Monitor tracks each conversation across turns and **flags** (never blocks,
in v1) when an attacker builds a breach gradually over many messages.

## Problem

`/chat` is stateless: `ChatRequest` takes one `prompt` (`src/main.py:15-16`),
nothing survives between requests, and every input check sees only the
latest message. The best-known jailbreaks exploit exactly that gap. Each
turn looks benign; the conversation doesn't.

| Attack | How it beats a per-message filter |
|---|---|
| Crescendo ([arXiv 2404.01833](https://arxiv.org/abs/2404.01833), Microsoft) | Escalates slowly, building on the model's own replies. |
| Many-shot jailbreaking ([Anthropic](https://www.anthropic.com/research/many-shot-jailbreaking)) | Fills the context with fake dialogue turns. |
| Deceptive Delight ([Unit 42](https://unit42.paloaltonetworks.com/jailbreak-llms-through-camouflage-distraction/)) | Hides an unsafe topic among benign ones; 65% ASR within 3 turns. |
| ActorAttack ([arXiv 2410.10700](https://arxiv.org/abs/2410.10700)) | Reaches the goal via a chain of innocuous related questions. |
| Echo Chamber ([NeuralTrust, arXiv 2601.05742](https://arxiv.org/abs/2601.05742)) | Context poisoning the model then amplifies; >90% ASR. |
| Payload splitting | An injection split across turns, so no single message matches. |
| Indirect injection ([arXiv 2302.12173](https://arxiv.org/abs/2302.12173)) | Payload arrives via tool output, not the user. |

Most guards share the gap. Lakera Guard screens only the latest interaction
([docs](https://docs.lakera.ai)), and Azure Prompt Shields is stateless
([docs](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/concepts/jailbreak-detection)).
The stateful options are hand-written rules:
[NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails) dialog rails
(Colang state machines) and [Invariant Labs](https://github.com/invariantlabs-ai/invariant)
trace rules (acquired by Snyk in 2025). OpenAI's
[o3 system card](https://openai.com/index/o3-o4-mini-system-card/) relies on
human monitoring to catch adversarial retries. Without this feature,
GuardProxy only stops single-shot attacks. This is also the first step
toward "Behavioral guardrails" (`docs/TODO.md:25`).

## Goals

1. Catch gradual attacks that per-turn checks miss: **≥70% of generated
   multi-turn attacks flagged within 5 turns**.
2. Stay quiet on long benign chats: **≤3% false-flag rate**.
3. Stay cheap: **≤15 ms p95 added per turn** on CPU.
4. Settle the riskiest assumption with data. Do per-turn scores carry
   signal on Crescendo-style attacks whose individual turns are benign?
5. Lay groundwork that P1/P2 and training on live traffic
   (`docs/TODO.md:41`) need: a messages API, a store and structured logs.

## Non-goals (v1)

| Non-goal | Why |
|---|---|
| Blocking on conversation risk | False positives pile up over long benign chats. Flag first; block once the false-flag rate is measured. |
| Output guardrails / leak checks | `src/guardrails/output/*` are stubs. They land in P1 with canaries. |
| Learned trajectory model | Needs the v1 eval harness and labelled data first (P2). |
| LLM judge on every turn | Too slow and costly on a CPU proxy. P2, flagged conversations only. |
| Tool calls, streaming, persistent store | None exist in `/chat` today. Redis goes behind the store interface later. |

## User stories

- As an **operator**, I want a flag when risk builds up over many turns,
  so that I can review an attacker who slipped every single-message check.
- As an **operator**, I want v1 flags never to block, so that a false
  positive costs me a log line, not a user.
- As an **operator**, I want each flag to say why (ledger or window,
  which head, which turns), so that I can triage it in a minute.
- As an **app developer**, I want to send `messages[]` plus a
  `conversation_id` while old `prompt`-only clients keep working, so that
  I don't have to rewrite my client.
- As a **security reviewer**, I want risk computed from history the proxy
  stored itself, so that an attacker can't reset it by editing or dropping
  earlier turns client-side.

## Requirements

### P0: v1, must ship

**P0-1. Messages API.** `ChatRequest` (`src/main.py:15-16`) gains
`messages: [{role: system|user|assistant, content}]` and
`conversation_id`. `prompt` stays and is treated as one user message.
Upstream moves from `/api/generate` to Ollama `/api/chat`
(`src/main.py:73-80`).
- [ ] A `prompt`-only request returns today's response shape. `/api/chat`'s
  `message.content` maps to `response`.
- [ ] `run_input_guardrails` (`src/main.py:38`) runs on the latest user
  message, and its PII-masked text is what gets forwarded.
- [ ] A request with both `prompt` and `messages`, or whose last message
  isn't `user`, returns 422.

**P0-2. Server-side conversation store.** A new
`src/guardrails/conversation/store.py` holds an in-memory dict with a TTL
and a size cap, behind a two-method interface (`get`, `append`) so Redis
can drop in. Nothing is stored today (`src/core/middleware.py` and
`src/utils/logger.py` are empty).
- [ ] **Proxy-held history is authoritative.** If a client resends
  `messages[]` with earlier turns edited or dropped, the ledger and window
  don't change.
- [ ] Only PII-masked text is stored. Memory stays bounded under a
  10k-conversation load test.
- [ ] Documented limit: run a single uvicorn worker, since each process
  has its own store.

**P0-3. Per-turn risk from the existing classifier.** `risk_t` is the max
head probability from the ML check that already runs in
`src/guardrails/input/pipeline.py:45-51`. It is 1.0 if any regex check
blocked. The turn needs no extra inference.
- [ ] Turns blocked per-turn are still recorded in the ledger.

**P0-4. Cumulative risk ledger.** `R_t = α·R_{t−1} + β·risk_t`, with
α < 1 so old turns decay (Temporal Context Awareness,
[arXiv 2503.15560](https://arxiv.org/abs/2503.15560)). α, β and the
threshold live in `src/config.py` Settings.
- [ ] Placeholder defaults: `CONV_ALPHA=0.8`, `CONV_BETA=0.5`,
  `CONV_FLAG_THRESHOLD=0.6`. With these, five straight turns at risk 0.4
  flag on turn 5. A single 0.9 spike (R = 0.45) does not flag.
- [ ] TCA warns that 10–15% weight miscalibration causes false positives
  and false negatives, so the eval reports a sensitivity sweep.

**P0-5. Cross-turn window rescoring.** The last `CONV_WINDOW_TURNS`
(default 4) user turns are concatenated and scored via
`segment_probabilities` (`src/guardrails/input/ml_classifier.py:11`). The
window runs inside the existing `asyncio.gather` (`pipeline.py:45-51`), so
it costs wall time only when it is the slowest check.
- [ ] Score the window as **whole text only**: the per-sentence max would
  just re-surface the strongest single turn. The TF-IDF path needs a
  whole-text flag; transformer `.score(text)` already scores whole text.
- [ ] Flag when `window_score ≥ threshold` **and** no single turn in the
  window crossed it alone.
- [ ] Unit test: a 3-turn split payload flags, and its fragments scored
  one by one don't.

**P0-6. FLAG action, response field, alert.** Add `FLAG` to
`GuardrailAction` (`src/schemas/guardrail.py:12-15`). `/chat` responses
gain `conversation_risk`, and each flag writes one structured JSON log
line via `src/utils/logger.py`.

```json
"conversation_risk": {
  "conversation_id": "c-123", "turn": 6, "score": 0.67, "threshold": 0.6,
  "window_score": 0.41, "flagged": true, "first_flagged_turn": 5,
  "reason": "ledger: sustained jailbreak-head risk, turns 2-6",
  "history": [{"turn": 1, "risk": 0.04}, "..."], "latency_ms": 3.1
}
```
- [ ] FLAG never sets `blocked: true` and never skips the LLM call.
- [ ] The log line has ids, scores, head and reason. It has no message
  content by default.
- [ ] `history` is capped at 20 turns, and the monitor runs under
  `DRY_RUN` too.

**P0-7. Latency budget.** The monitor adds ≤15 ms p95 per turn on CPU
(store, ledger and the window's share of the gather), reported in
`conversation_risk.latency_ms`.
- [ ] Re-measure when the DeBERTa-v3-small ONNX model ships. If the window
  breaks the budget, cap the window's length before dropping it.

### P1: v1.1

| Item | Notes |
|---|---|
| Block-then-rephrase detection | Embedding similarity between a new turn and earlier **blocked** turns in the same conversation. |
| Canary token + output leak check | Inject a random token into the system prompt and flag any response containing it (the [Rebuff](https://github.com/protectai/rebuff) pattern, archived 2025). Covers `docs/TODO.md:24`. |
| Real output guardrails | Replace the `pass` in `validate_output_guardrails` (`src/main.py:28-29`). Crescendo feeds on the model's replies, so output scores should feed the ledger. |

### P2: v2, design for it but don't build it

| Item | Notes |
|---|---|
| Learned trajectory model | DeepContext-style BERT+GRU over turn embeddings ([arXiv 2602.16935](https://arxiv.org/abs/2602.16935)): F1 0.84 vs PromptGuard2's 0.67, 19 ms/turn on a T4, detects ~4.2 turns in. |
| LLM-judge cascade | Judge only flagged conversations, like the Constitutional Classifiers++ probe→classifier cascade ([arXiv 2601.04603](https://arxiv.org/abs/2601.04603)). LlamaFirewall AlignmentCheck ([arXiv 2505.03574](https://arxiv.org/abs/2505.03574)) cut AgentDojo ASR from 17.6% to 2.89%, but needs a 70B-class judge. |
| Tool-call flow rules | Invariant-style trace rules, once `/chat` carries tool calls. |
| Agent benchmarks | [AgentDojo](https://github.com/ethz-spylab/agentdojo) (MIT, 629 security cases), [InjecAgent](https://github.com/uiuc-kang-lab/InjecAgent) (MIT, 1,054 cases). |

**Architectural insurance:** keep per-turn records as dicts, with room for
embeddings and output scores. Feed the ledger a single `risk_t`, so any
scorer can plug in.

## Success metrics & eval plan

These are **initial targets to calibrate**, not commitments.

| Metric | Measured as | v1 target | Stretch |
|---|---|---|---|
| Detection rate | % of generated attack conversations flagged | ≥70% within 5 turns | ≥85% |
| Turns-to-detection | median `first_flagged_turn` | ≤4 | ≤3 |
| False-flag rate | % of benign multi-turn conversations ever flagged | ≤3% | ≤1% |
| Added latency | p95 `conversation_risk.latency_ms`, CPU | ≤15 ms | ≤5 ms |
| Lift over per-turn | detection minus % of attacks with any per-turn block | > 0 | — |

- **Attacks:** run [PyRIT](https://github.com/Azure/PyRIT)'s Crescendo
  orchestrator and promptfoo's Crescendo, GOAT and Hydra
  [strategies](https://www.promptfoo.dev/docs/red-team/strategies/multi-turn/)
  against a staging GuardProxy endpoint with live Ollama.
- **Benign:** [oasst2](https://huggingface.co/datasets/OpenAssistant/oasst2)
  (Apache-2.0) threads with ≥4 user turns, replayed under `DRY_RUN`. This
  needs no LLM, because v1 scores only user turns.
- **Riskiest assumption first (2-day spike):** replay ~50 Crescendo
  conversations and ~500 oasst2 threads through today's classifier
  offline. If per-turn scores on attack turns aren't separable from
  benign ones, ship the window alone and pull the P2 trajectory model
  forward.
- **Lagging:** once logs exist, track the share of flags confirmed on
  manual review.
- Reports go to `reports/conversation-eval-<date>.md`, following
  `src/trainer/benchmark.py`.

## Open questions

| # | Question | Owner | Blocking? |
|---|---|---|---|
| 1 | Is `conversation_id` global or scoped per tenant? There's no auth yet (`docs/TODO.md:14`), so a guessed id can pollute another client's ledger. | Engineering | Before shared deployment |
| 2 | How does GuardProxy learn each agent's system prompt: per request (`role: system`) or per-app config? This matters for P1 canaries. | Engineering | No |
| 3 | Alert destinations: a log line only, or a webhook too? | Nitesh | No |
| 4 | [MHJ](https://huggingface.co/datasets/ScaleAI/mhj) (Scale AI; 2,912 prompts, 537 human multi-turn jailbreaks) is CC-BY-NC-4.0 and gated. Is eval-only use OK under the commercial-safe data rule? SafeDialBench and CoSafe licenses are unconfirmed. | Legal | No |
| 5 | How should α, β and the threshold be set? Proposal: grid-search on oasst2 plus generated attacks for the best detection at ≤3% false flags, on a held-out split. | Data | For launch defaults |
| 6 | When `conversation_id` is missing, treat the request as single-turn, or mint an id and return it? Proposal: mint one. | Engineering | Yes |
| 7 | Retention: stored text is PII-masked, but the masking is regex-only. What TTL is acceptable, and should storage be opt-out? | Legal / Engineering | Before shared deployment |
| 8 | When client `messages[]` diverge from stored history, which version goes upstream, and does the divergence raise risk? | Engineering | No. v1 forwards the client's version |

## Phasing

| Phase | Scope | Exit criterion |
|---|---|---|
| Spike (2 days) | Offline separability check | Go/no-go on the ledger |
| **v1** | All P0 + eval harness | Targets met on the calibration split; a sample of flags reviewed by hand |
| v1.1 | P1: rephrase detection, canary + leak check, output guardrails | Every planted canary caught |
| v2 | P2: trajectory model, LLM-judge cascade, tool-call rules, agent benchmarks | False-flag data supports promoting high-confidence flags to BLOCK |

P0-1 comes first and P0-2 precedes P0-4/5. There are no external deadlines.
