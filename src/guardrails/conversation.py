"""Conversation monitor v1: flags breaches built up across turns (docs/specs/conversation-monitor.md).

Two signals per conversation, both flag-only (never block):
  - ledger: R_t = CONV_ALPHA * R_{t-1} + CONV_BETA * risk_t, where risk_t is the
    turn's existing per-turn risk (1.0 if a regex check blocked it, else the ML
    classifier's top head). Sustained moderate risk adds up; old turns decay.
  - window: the last CONV_WINDOW_TURNS user turns joined and scored as one
    text, to catch a payload split across turns that no single turn shows.

History lives here, not with the client: a client resending edited or
trimmed messages[] can't reset its own escalation. The store is in-memory,
per process, so run a single uvicorn worker (or swap in Redis later).
"""
import json
import logging
import time
from collections import OrderedDict
from dataclasses import dataclass, field

from src.config import settings
from src.guardrails.input.model_registry import get_registry
from src.schemas.guardrail import GuardrailAction, TotalInputGuardrailResult

log = logging.getLogger("guardproxy.conversation")
HISTORY_IN_RESPONSE = 20
MAX_TURNS_KEPT = 50


@dataclass
class Turn:
    turn: int
    text: str  # PII-masked
    risk: float
    head: str


@dataclass
class Conversation:
    turns: list[Turn] = field(default_factory=list)
    ledger: float = 0.0
    first_flagged_turn: int | None = None
    last_seen: float = field(default_factory=time.monotonic)


class ConversationStore:
    """In-memory, TTL'd, size-capped (least recently used evicted first)."""

    def __init__(self, ttl_seconds: int, max_conversations: int):
        self.ttl = ttl_seconds
        self.max = max_conversations
        self._data: OrderedDict[str, Conversation] = OrderedDict()

    def get(self, conversation_id: str) -> Conversation | None:
        conv = self._data.get(conversation_id)
        if conv is not None and time.monotonic() - conv.last_seen > self.ttl:
            del self._data[conversation_id]
            return None
        return conv

    def get_or_create(self, conversation_id: str) -> Conversation:
        conv = self.get(conversation_id)
        if conv is None:
            conv = self._data[conversation_id] = Conversation()
        conv.last_seen = time.monotonic()
        self._data.move_to_end(conversation_id)
        while len(self._data) > self.max:
            self._data.popitem(last=False)
        return conv

    def __len__(self) -> int:
        return len(self._data)


store = ConversationStore(settings.CONV_TTL_SECONDS, settings.CONV_MAX_CONVERSATIONS)


def turn_risk(result: TotalInputGuardrailResult) -> tuple[float, str]:
    """The turn's risk from checks that already ran: no extra inference."""
    for name, r in (("prompt_injection", result.prompt_injection_result),
                    ("jailbreak", result.jailbreak_result),
                    ("toxic", result.toxic_content_result)):
        if r.action == GuardrailAction.BLOCK:
            return 1.0, name
    ml = result.ml_classifier_result
    heads = ml.details.get("heads") or {}
    return 1.0 - ml.score, (max(heads, key=heads.get) if heads else "ml")


def window_texts(conversation_id: str, current_text: str) -> list[str]:
    conv = store.get(conversation_id)
    prior = [t.text for t in conv.turns[-(settings.CONV_WINDOW_TURNS - 1):]] if conv else []
    return [*prior, current_text]


def score_window(texts: list[str]) -> tuple[float, str]:
    """Whole-text score of the joined turns. Deliberately not per-sentence:
    a per-segment max would just re-surface the strongest single turn."""
    loaded = get_registry().get()
    if len(texts) < 2 or loaded is None:
        return 0.0, ""
    # Split payloads sit in the recent turns; capping length keeps the
    # transformer to ~2 ONNX windows (the 15 ms budget) on long chats.
    text = "\n".join(texts)[-settings.CONV_WINDOW_MAX_CHARS:]
    if hasattr(loaded.model, "score"):
        heads = loaded.model.score(text)
    elif isinstance(loaded.model, dict):
        x = loaded.vectorizer.transform([text])
        heads = {h: float(clf.predict_proba(x)[0, 1]) for h, clf in loaded.model.items()}
    else:
        return 0.0, ""  # legacy single-estimator artifacts aren't windowed
    head = max(heads, key=heads.get)
    return heads[head], head


def record_turn(conversation_id: str, text: str, risk: float, head: str,
                window: tuple[float, str], window_ms: float = 0.0) -> dict:
    start = time.perf_counter()
    conv = store.get_or_create(conversation_id)
    n = conv.turns[-1].turn + 1 if conv.turns else 1
    conv.turns.append(Turn(n, text, risk, head))
    del conv.turns[:-MAX_TURNS_KEPT]
    conv.ledger = settings.CONV_ALPHA * conv.ledger + settings.CONV_BETA * risk

    window_score, window_head = window
    in_window = conv.turns[-settings.CONV_WINDOW_TURNS:]
    # Only a window finding when no single turn in it already crossed the line.
    window_hit = (len(in_window) > 1 and window_score >= settings.ML_CLASSIFIER_THRESHOLD
                  and all(t.risk < settings.ML_CLASSIFIER_THRESHOLD for t in in_window))
    ledger_hit = conv.ledger >= settings.CONV_FLAG_THRESHOLD
    flagged = ledger_hit or window_hit
    if flagged and conv.first_flagged_turn is None:
        conv.first_flagged_turn = n

    reason = None
    if window_hit:
        reason = f"window: '{window_head}' payload split across turns {in_window[0].turn}-{n}"
    elif ledger_hit:
        risky = [t.turn for t in conv.turns if t.risk >= 0.2] or [n]
        top = max(conv.turns[-settings.CONV_WINDOW_TURNS:], key=lambda t: t.risk).head
        reason = f"ledger: sustained '{top}' risk, turns {risky[0]}-{n}"

    out = {
        "conversation_id": conversation_id,
        "turn": n,
        "action": (GuardrailAction.FLAG if flagged else GuardrailAction.ALLOW).value,
        "flagged": flagged,
        "score": round(conv.ledger, 4),
        "threshold": settings.CONV_FLAG_THRESHOLD,
        "window_score": round(window_score, 4),
        "first_flagged_turn": conv.first_flagged_turn,
        "reason": reason,
        "history": [{"turn": t.turn, "risk": round(t.risk, 4)} for t in conv.turns[-HISTORY_IN_RESPONSE:]],
        "latency_ms": round(window_ms + (time.perf_counter() - start) * 1000, 2),
    }
    if flagged:  # ids and scores only: no message content in logs
        log.warning(json.dumps({"event": "conversation_flag", **{k: out[k] for k in (
            "conversation_id", "turn", "score", "window_score", "first_flagged_turn", "reason")}}))
    return out
