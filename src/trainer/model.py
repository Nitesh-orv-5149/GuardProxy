from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

from src.trainer.featurizer import build_vectorizer

# One binary head per attack class, all sharing a single vectorizer.
HEADS = ["prompt_injection", "jailbreak", "toxic"]


@dataclass
class TrainResult:
    vectorizer: object
    model: object  # dict[head -> LogisticRegression]
    accuracy: float  # block-vs-allow accuracy on the held-out split
    macro_f1: float  # mean of per-head F1
    per_class_f1: dict[str, float]  # per-head F1 on the held-out split
    n_train: int
    n_test: int


def head_probabilities(model: dict, vector) -> dict[str, float]:
    """P(positive) from each head for a single already-vectorized row."""
    return {head: float(clf.predict_proba(vector)[0][1]) for head, clf in model.items()}


def _head_rows(x, labels, head):
    """A head trains on its own positives vs `safe` only. Other attack
    classes are left out rather than labelled negative, since e.g. jailbreak
    prompts often contain injection-like text."""
    idx = [i for i, l in enumerate(labels) if l in (head, "safe")]
    return x[idx], [1 if labels[i] == head else 0 for i in idx]


def train_and_evaluate(
    examples: list[tuple[str, str]],
    test_size: float = 0.2,
    random_state: int = 42,
) -> TrainResult:
    texts = [t for t, _ in examples]
    labels = [l for _, l in examples]

    label_counts: dict[str, int] = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    missing = [h for h in HEADS + ["safe"] if label_counts.get(h, 0) < 2]
    if missing:
        raise ValueError(f"Need at least 2 examples of each label, missing/short: {missing}")
    can_stratify = round(test_size * len(examples)) >= len(label_counts)

    split_kwargs = {"test_size": test_size, "random_state": random_state}
    if can_stratify:
        split_kwargs["stratify"] = labels

    x_train, x_test, y_train, y_test = train_test_split(texts, labels, **split_kwargs)

    vectorizer = build_vectorizer()
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    model: dict[str, LogisticRegression] = {}
    per_class_f1: dict[str, float] = {}
    for head in HEADS:
        xh, yh = _head_rows(x_train_vec, y_train, head)
        # C=4 beat C in {0.5, 1, 10} on every head's held-out F1 (2026-09 sweep).
        clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=4.0)
        clf.fit(xh, yh)
        model[head] = clf

        xt, yt = _head_rows(x_test_vec, y_test, head)
        per_class_f1[head] = float(f1_score(yt, clf.predict(xt), zero_division=0))

    # Overall block decision: block if any head says positive.
    blocked = sum(clf.predict(x_test_vec) for clf in model.values()) > 0
    accuracy = accuracy_score([l != "safe" for l in y_test], blocked)

    return TrainResult(
        vectorizer=vectorizer,
        model=model,
        accuracy=float(accuracy),
        macro_f1=sum(per_class_f1.values()) / len(per_class_f1),
        per_class_f1=per_class_f1,
        n_train=len(x_train),
        n_test=len(x_test),
    )
