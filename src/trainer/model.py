from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import train_test_split

from src.trainer.data.loader import LABEL_CLASSES
from src.trainer.featurizer import build_vectorizer


@dataclass
class TrainResult:
    vectorizer: object
    model: object
    accuracy: float
    macro_f1: float
    per_class_f1: dict[str, float]
    n_train: int
    n_test: int


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
    can_stratify = (
        len(examples) >= 5
        and min(label_counts.values()) >= 2
        and round(test_size * len(examples)) >= len(label_counts)
    )

    split_kwargs = {"test_size": test_size, "random_state": random_state}
    if can_stratify:
        split_kwargs["stratify"] = labels

    x_train, x_test, y_train, y_test = train_test_split(texts, labels, **split_kwargs)

    vectorizer = build_vectorizer()
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(x_train_vec, y_train)

    y_pred = model.predict(x_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    present_labels = sorted(set(y_test) | set(y_pred))
    macro_f1 = f1_score(y_test, y_pred, labels=present_labels, average="macro", zero_division=0)
    per_class = f1_score(y_test, y_pred, labels=present_labels, average=None, zero_division=0)
    per_class_f1 = dict(zip(present_labels, (float(v) for v in per_class)))
    for label in LABEL_CLASSES:
        per_class_f1.setdefault(label, 0.0)

    return TrainResult(
        vectorizer=vectorizer,
        model=model,
        accuracy=float(accuracy),
        macro_f1=float(macro_f1),
        per_class_f1=per_class_f1,
        n_train=len(x_train),
        n_test=len(x_test),
    )
