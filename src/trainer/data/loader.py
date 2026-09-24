from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

LABEL_CLASSES = ["safe", "jailbreak", "prompt_injection", "toxic"]


class DatasetSource(ABC):
    """A source of labeled (text, label) training examples."""

    @abstractmethod
    def load(self) -> list[tuple[str, str]]:
        """Return a list of (text, label) pairs. label must be one of LABEL_CLASSES."""


class LocalCSVSource(DatasetSource):
    """Reads every *.csv file in a directory. Each CSV must have `text` and `label` columns."""

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)

    def load(self) -> list[tuple[str, str]]:
        csv_paths = sorted(self.data_dir.glob("*.csv"))
        if not csv_paths:
            raise FileNotFoundError(
                f"No CSV files found in {self.data_dir}. "
                f"Each CSV needs `text` and `label` columns "
                f"(label one of {LABEL_CLASSES})."
            )

        examples: list[tuple[str, str]] = []
        for path in csv_paths:
            df = pd.read_csv(path)
            if "text" not in df.columns or "label" not in df.columns:
                raise ValueError(f"{path} must have `text` and `label` columns.")
            for text, label in zip(df["text"], df["label"]):
                if label not in LABEL_CLASSES:
                    raise ValueError(
                        f"{path}: unknown label '{label}', expected one of {LABEL_CLASSES}"
                    )
                examples.append((str(text), str(label)))
        return examples


class HFDatasetSource(DatasetSource):
    """Streams a bounded sample from BudEcosystem/guardrail-training-data on
    the HuggingFace Hub (~4M rows across 26 harm categories) and maps it onto
    our four label classes.

    The dataset only has one merged category for "Jailbreak & prompt
    injection" — it doesn't distinguish the two the way our regex checks do
    — so rows in that category are sub-classified by running them through
    our own `check_prompt_injection`/`check_jailbreak` regex checks. Every
    other unsafe category (hate speech, self-harm, violence, drugs, fraud,
    etc.) collapses into `toxic`, matching the single broad `toxic_content`
    regex check we already have. This keeps the ML classifier's label space
    identical to the existing guardrail categories instead of introducing a
    new taxonomy.
    """

    DATASET_NAME = "BudEcosystem/guardrail-training-data"

    def __init__(self, split: str = "train", max_samples: int = 10_000, seed: int = 42):
        self.split = split
        self.max_samples = max_samples
        self.seed = seed

    def _map_label(self, text: str, row: dict) -> str:
        if row.get("is_safe"):
            return "safe"

        category = (row.get("category") or "").lower()
        if "jailbreak" in category or "prompt injection" in category:
            from src.guardrails.input.jailbreak import check_jailbreak
            from src.guardrails.input.prompt_injection import check_prompt_injection
            from src.schemas.guardrail import GuardrailAction

            if check_prompt_injection(text).action == GuardrailAction.BLOCK:
                return "prompt_injection"
            if check_jailbreak(text).action == GuardrailAction.BLOCK:
                return "jailbreak"
            return "jailbreak"  # category name leads with "jailbreak"; default there

        return "toxic"

    def load(self) -> list[tuple[str, str]]:
        from datasets import load_dataset  # imported lazily; optional heavy dep

        stream = load_dataset(self.DATASET_NAME, split=self.split, streaming=True)
        stream = stream.shuffle(seed=self.seed, buffer_size=10_000)

        # `safe` and `prompt_injection` (a narrow regex-matched slice of one
        # merged dataset category) are rare next to the other 25 harm
        # categories, so a pure random sample starves them. Bucket by our
        # mapped label instead and cap each bucket at an even share of
        # max_samples, scanning a bounded multiple of rows so a very rare
        # label can't spin the stream forever.
        per_label_cap = max(1, self.max_samples // len(LABEL_CLASSES))
        max_rows_to_scan = self.max_samples * 50

        buckets: dict[str, list[tuple[str, str]]] = {label: [] for label in LABEL_CLASSES}
        rows_scanned = 0
        for row in stream:
            rows_scanned += 1
            text = row.get("text")
            if text:
                label = self._map_label(str(text), row)
                if len(buckets[label]) < per_label_cap:
                    buckets[label].append((str(text), label))

            if rows_scanned >= max_rows_to_scan:
                break
            if all(len(bucket) >= per_label_cap for bucket in buckets.values()):
                break

        examples = [example for bucket in buckets.values() for example in bucket]
        if not examples:
            raise RuntimeError(f"No examples loaded from {self.DATASET_NAME} split={self.split}")
        return examples


# TODO: TrafficLogSource — build training examples from logged /chat prompts
# once request logging + a labeling workflow exist, to support retraining on
# real production traffic rather than only public datasets.
