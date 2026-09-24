from src.config import settings
from src.guardrails.input.model_registry import get_registry
from src.schemas.guardrail import GuardrailAction, GuardrailCheckResult


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
    probabilities = loaded.model.predict_proba(vector)[0]
    classes = list(loaded.model.classes_)

    best_idx = probabilities.argmax()
    label = classes[best_idx]
    confidence = float(probabilities[best_idx])
    safe_score = float(probabilities[classes.index("safe")]) if "safe" in classes else 1.0 - confidence

    reason = f"ML classifier predicted '{label}' (confidence {confidence:.2f})"

    if label != "safe" and confidence >= settings.ML_CLASSIFIER_THRESHOLD:
        return GuardrailCheckResult(
            passed=False,
            score=safe_score,
            reason=reason,
            action=GuardrailAction.BLOCK,
        )

    return GuardrailCheckResult(
        passed=True,
        score=safe_score,
        reason=reason,
        action=GuardrailAction.ALLOW,
    )
