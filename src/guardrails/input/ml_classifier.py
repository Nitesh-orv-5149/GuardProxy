from src.config import settings
from src.guardrails.input.model_registry import get_registry
from src.schemas.guardrail import GuardrailAction, GuardrailCheckResult
from src.trainer.model import head_probabilities


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

    vector = loaded.vectorizer.transform([text])
    if not isinstance(loaded.model, dict):
        return _legacy_multiclass(loaded.model, vector)

    heads = head_probabilities(loaded.model, vector)
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
