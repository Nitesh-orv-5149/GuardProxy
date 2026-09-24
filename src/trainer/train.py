"""CLI entrypoint for training the input guardrail ML classifier.

Usage:
    python -m src.trainer.train --source csv [--data-dir data/input_classifier]
    python -m src.trainer.train --source huggingface [--max-samples 10000]
                                 [--min-macro-f1 0.75] [--keep-versions 5]

Trains a TF-IDF + LogisticRegression classifier over our four label classes
(safe, jailbreak, prompt_injection, toxic) and — only if it clears the
minimum macro-F1 quality gate — writes a new versioned artifact under
ML_MODEL_DIR and atomically repoints pointer.json at it. A run that fails
the quality gate exits non-zero without touching the pointer, so a bad
retrain never silently replaces a good model.
"""
import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn

from src.config import settings
from src.trainer.data.loader import LABEL_CLASSES, DatasetSource, HFDatasetSource, LocalCSVSource
from src.trainer.model import train_and_evaluate


def _build_source(source: str, data_dir: str, max_samples: int) -> DatasetSource:
    if source == "csv":
        return LocalCSVSource(data_dir)
    if source == "huggingface":
        return HFDatasetSource(max_samples=max_samples)
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


def train(
    data_dir: str,
    min_macro_f1: float,
    keep_versions: int,
    source: str = "csv",
    max_samples: int = 10_000,
) -> int:
    model_dir = Path(settings.ML_MODEL_DIR)
    model_dir.mkdir(parents=True, exist_ok=True)

    dataset_source = _build_source(source, data_dir, max_samples)
    examples = dataset_source.load()
    print(f"Loaded {len(examples)} labeled examples from source={source}")

    result = train_and_evaluate(examples)
    print(f"accuracy={result.accuracy:.3f} macro_f1={result.macro_f1:.3f}")
    for label, f1 in result.per_class_f1.items():
        print(f"  {label}: f1={f1:.3f}")

    if result.macro_f1 < min_macro_f1:
        print(
            f"REJECTED: macro_f1 {result.macro_f1:.3f} is below the minimum "
            f"{min_macro_f1:.3f} quality gate. Model was NOT shipped."
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
        "label_classes": LABEL_CLASSES,
        "accuracy": result.accuracy,
        "macro_f1": result.macro_f1,
        "per_class_f1": result.per_class_f1,
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
        "--max-samples",
        type=int,
        default=10_000,
        help="Cap on rows streamed from the huggingface source (prototype scale, not the full ~4M-row dataset).",
    )
    parser.add_argument("--min-macro-f1", type=float, default=settings.ML_MIN_MACRO_F1)
    parser.add_argument("--keep-versions", type=int, default=5)
    args = parser.parse_args()

    exit_code = train(
        args.data_dir,
        args.min_macro_f1,
        args.keep_versions,
        source=args.source,
        max_samples=args.max_samples,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
