"""CLI entrypoint for training the input guardrail ML classifier.

Usage:
    python -m src.trainer.train --source csv [--data-dir data/input_classifier]
    python -m src.trainer.train --source huggingface
                                 [--min-macro-f1 0.75] [--min-eval-accuracy 0.8]
                                 [--keep-versions 5]

Trains one binary TF-IDF + LogisticRegression head per attack class
(prompt_injection, jailbreak, toxic) on a shared vectorizer and — only if
every head clears the minimum F1 gate AND the hand-written eval set
(eval_prompts.csv) clears the block-accuracy gate — writes a new versioned
artifact under ML_MODEL_DIR and atomically repoints pointer.json at it. A
run that fails a gate exits non-zero without touching the pointer, so a bad
retrain never silently replaces a good model.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
import sklearn

from src.config import settings
from src.trainer.data.loader import LABEL_CLASSES, DatasetSource, HFDatasetSource, LocalCSVSource
from src.trainer.model import head_probabilities, train_and_evaluate

EVAL_PROMPTS_PATH = Path(__file__).parent / "eval_prompts.csv"


def _build_source(source: str, data_dir: str) -> DatasetSource:
    if source == "csv":
        return LocalCSVSource(data_dir)
    if source == "huggingface":
        return HFDatasetSource()
    raise ValueError(f"Unknown source '{source}', expected 'csv' or 'huggingface'")


def _write_pointer(model_dir: Path, version: str) -> None:
    pointer_path = model_dir / "pointer.json"
    tmp_path = model_dir / "pointer.json.tmp"
    tmp_path.write_text(json.dumps({"current_version": version}, indent=2))
    tmp_path.replace(pointer_path)


def _prune_old_versions(model_dir: Path, keep_versions: int) -> None:
    version_dirs = sorted(
        (p for p in model_dir.iterdir() if p.is_dir()),
        key=lambda p: p.name,
    )
    for old_dir in version_dirs[:-keep_versions]:
        for f in old_dir.iterdir():
            f.unlink()
        old_dir.rmdir()


def evaluate_eval_set(vectorizer, model: dict, threshold: float) -> tuple[float, list[str]]:
    """Block-decision accuracy on the hand-written eval set. Only block vs
    allow is scored: an injection caught by the jailbreak head still counts."""
    df = pd.read_csv(EVAL_PROMPTS_PATH)
    misses = []
    for text, label in zip(df["text"], df["label"]):
        heads = head_probabilities(model, vectorizer.transform([text]))
        top = max(heads, key=heads.get)
        blocked = heads[top] >= threshold
        if blocked != (label != "safe"):
            misses.append(f"expected {label:<16} got {'BLOCK ' + top if blocked else 'allow'} (p={heads[top]:.2f}): {text}")
    return 1 - len(misses) / len(df), misses


def train(
    data_dir: str,
    min_macro_f1: float,
    keep_versions: int,
    source: str = "csv",
    min_eval_accuracy: float = 0.0,
) -> int:
    model_dir = Path(settings.ML_MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)

    dataset_source = _build_source(source, data_dir)
    examples = dataset_source.load()
    label_counts = {l: sum(1 for _, x in examples if x == l) for l in LABEL_CLASSES}
    print(f"Loaded {len(examples)} labeled examples from source={source}: {label_counts}")

    result = train_and_evaluate(examples)
    print(f"held-out block accuracy={result.accuracy:.3f} macro_f1={result.macro_f1:.3f}")
    for head, f1 in result.per_class_f1.items():
        print(f"  {head}: f1={f1:.3f}")

    weakest_head = min(result.per_class_f1, key=result.per_class_f1.get)
    if result.per_class_f1[weakest_head] < min_macro_f1:
        print(
            f"REJECTED: head '{weakest_head}' f1 {result.per_class_f1[weakest_head]:.3f} is below "
            f"the minimum {min_macro_f1:.3f} quality gate. Model was NOT shipped."
        )
        return 1

    eval_accuracy, misses = evaluate_eval_set(
        result.vectorizer, result.model, settings.ML_CLASSIFIER_THRESHOLD
    )
    print(f"eval set block accuracy={eval_accuracy:.3f} ({len(misses)} misses)")
    for miss in misses:
        print(f"  {miss}")

    if eval_accuracy < min_eval_accuracy:
        print(
            f"REJECTED: eval set block accuracy {eval_accuracy:.3f} is below the minimum "
            f"{min_eval_accuracy:.3f} quality gate. Model was NOT shipped."
        )
        return 1

    version = "v" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    version_dir = model_dir / version
    version_dir.mkdir(parents=True, exist_ok=False)

    joblib.dump(result.model, version_dir / "model.joblib")
    joblib.dump(result.vectorizer, version_dir / "vectorizer.joblib")

    metadata = {
        "version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "data_dir": str(Path(data_dir).resolve()) if source == "csv" else None,
        "n_examples_loaded": len(examples),
        "label_counts": label_counts,
        "label_classes": LABEL_CLASSES,
        "heads": list(result.model),
        "accuracy": result.accuracy,
        "macro_f1": result.macro_f1,
        "per_class_f1": result.per_class_f1,
        "eval_block_accuracy": eval_accuracy,
        "eval_misses": misses,
        "n_train": result.n_train,
        "n_test": result.n_test,
        "sklearn_version": sklearn.__version__,
    }
    (version_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    _write_pointer(model_dir, version)
    _prune_old_versions(model_dir, keep_versions)

    print(f"SHIPPED: {version} is now the active input classifier ({model_dir}).")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the input guardrail ML classifier.")
    parser.add_argument("--source", choices=["csv", "huggingface"], default="csv")
    parser.add_argument("--data-dir", default="data/input_classifier")
    parser.add_argument(
        "--min-macro-f1", type=float, default=settings.ML_MIN_MACRO_F1,
        help="Every head's held-out F1 must reach this.",
    )
    parser.add_argument("--min-eval-accuracy", type=float, default=settings.ML_MIN_EVAL_ACCURACY)
    parser.add_argument("--keep-versions", type=int, default=5)
    args = parser.parse_args()

    exit_code = train(
        args.data_dir,
        args.min_macro_f1,
        args.keep_versions,
        source=args.source,
        min_eval_accuracy=args.min_eval_accuracy,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
