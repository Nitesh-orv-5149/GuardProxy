import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib

from src.config import settings


@dataclass
class LoadedModel:
    version: str
    vectorizer: object
    model: object
    metadata: dict


class ModelRegistry:
    """Lazily loads the trained input classifier and hot-reloads it whenever
    pointer.json (written by src.trainer.train) points at a new version —
    no process restart or docker rebuild required."""

    def __init__(self, model_dir: Optional[str | Path] = None):
        self.model_dir = Path(model_dir) if model_dir is not None else Path(settings.ML_MODEL_DIR)
        self._loaded: Optional[LoadedModel] = None

    def _read_pointer_version(self) -> Optional[str]:
        pointer_path = self.model_dir / "pointer.json"
        if not pointer_path.exists():
            return None
        try:
            data = json.loads(pointer_path.read_text())
        except (json.JSONDecodeError, OSError):
            return None
        return data.get("current_version")

    def get(self, version: Optional[str] = None) -> Optional[LoadedModel]:
        """The pointer's version, or a pinned one (benchmarking a candidate before shipping it)."""
        version = version or self._read_pointer_version()
        if version is None:
            self._loaded = None
            return None
        if self._loaded is not None and self._loaded.version == version:
            return self._loaded

        version_dir = self.model_dir / version
        try:
            metadata = json.loads((version_dir / "metadata.json").read_text())
            if metadata.get("kind") == "transformer":
                from src.guardrails.input.transformer_model import TransformerHeads

                model, vectorizer = TransformerHeads(version_dir, metadata["heads"], metadata["max_len"]), None
            else:
                model = joblib.load(version_dir / "model.joblib")
                vectorizer = joblib.load(version_dir / "vectorizer.joblib")
        except Exception:  # noqa: BLE001 - onnxruntime raises its own error types
            # Keep serving the previously loaded model if the new version
            # can't be read (e.g. still being written to disk).
            return self._loaded

        self._loaded = LoadedModel(version=version, vectorizer=vectorizer, model=model, metadata=metadata)
        return self._loaded


_registry: Optional[ModelRegistry] = None


def get_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


def reset_registry(model_dir: Optional[str | Path] = None) -> ModelRegistry:
    """Test helper: force a fresh registry, optionally pointed at a fixture dir."""
    global _registry
    _registry = ModelRegistry(model_dir)
    return _registry
