import re

from src.config import settings
from src.guardrails.input.model_registry import get_registry
from src.schemas.guardrail import GuardrailAction, GuardrailCheckResult


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")


def segment_probabilities(model: dict, vectorizer, text: str) -> dict[str, float]:
    """Per-head max P(positive) over the whole text and each of its sentences.
    TF-IDF L2-normalizes the whole prompt, so benign filler around an attack
    dilutes its signal; scoring sentences on their own undoes that."""
    segments = [s for s in _SENTENCE_END.split(text) if s.strip()]
    texts = [text] + (segments if len(segments) > 1 else [])
    x = vectorizer.transform(texts)
    return {head: float(clf.predict_proba(x)[:, 1].max()) for head, clf in model.items()}


def _legacy_multiclass(model, vector) -> GuardrailCheckResult:
    # Pre-2026-09 artifacts were a single 4-class estimator; kept so rolling
    # pointer.json back to an old version still works.
    probabilities = model.predict_proba(vector)[0]
    classes = list(model.classes_)
    best_idx = probabilities.argmax()
    label = classes[best_idx]
    confidence = float(probabilities[best_idx])
    safe_score = float(probabilities[classes.index("safe")]) if "safe" in classes else 1.0 - confidence
    blocked = label != "safe" and confidence >= settings.ML_CLASSIFIER_THRESHOLD
    return GuardrailCheckResult(
        passed=not blocked,
        score=safe_score,
        reason=f"ML classifier predicted '{label}' (confidence {confidence:.2f})",
        action=GuardrailAction.BLOCK if blocked else GuardrailAction.ALLOW,
    )


def check_ml_classifier(text: str) -> GuardrailCheckResult:
    loaded = get_registry().get()
    if loaded is None:
        return GuardrailCheckResult(
            passed=True,
            score=1.0,
            reason="ML classifier not loaded",
            action=GuardrailAction.ALLOW,
        )

    if not isinstance(loaded.model, dict):
        return _legacy_multiclass(loaded.model, loaded.vectorizer.transform([text]))

    heads = segment_probabilities(loaded.model, loaded.vectorizer, text)
    top_head = max(heads, key=heads.get)
    top_prob = heads[top_head]
    blocked = top_prob >= settings.ML_CLASSIFIER_THRESHOLD

    return GuardrailCheckResult(
        passed=not blocked,
        score=1.0 - top_prob,
        reason=(
            f"ML classifier flagged '{top_head}' (p={top_prob:.2f})"
            if blocked
            else f"ML classifier: no head above threshold (max '{top_head}' p={top_prob:.2f})"
        ),
        action=GuardrailAction.BLOCK if blocked else GuardrailAction.ALLOW,
        details={"heads": {h: round(p, 4) for h, p in heads.items()}},
    )
