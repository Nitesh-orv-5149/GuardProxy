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


def _map_deepset(row: dict) -> tuple[str, str]:
    return row["text"], "prompt_injection" if row["label"] == 1 else "safe"


def _map_jackhhao(row: dict) -> tuple[str, str]:
    return row["prompt"], "jailbreak" if row["type"] == "jailbreak" else "safe"


def _map_toxic_chat(row: dict) -> tuple[str, str]:
    if row["jailbreaking"] == 1:
        return row["user_input"], "jailbreak"
    if row["toxicity"] == 1:
        return row["user_input"], "toxic"
    return row["user_input"], "safe"


class HFDatasetSource(DatasetSource):
    """Loads three purpose-built public datasets from the HuggingFace Hub,
    each of which labels exactly one of our attack classes against benign:

    - deepset/prompt-injections          -> prompt_injection vs safe
    - jackhhao/jailbreak-classification  -> jailbreak vs safe
    - lmsys/toxic-chat (toxicchat0124)   -> jailbreak / toxic vs safe
      (real user prompts; CC-BY-NC-4.0, non-commercial)

    Both train and test splits are used; train.py does its own split.
    Duplicate texts are dropped (first label wins).
    """

    SOURCES = [
        ("deepset/prompt-injections", None, _map_deepset),
        ("jackhhao/jailbreak-classification", None, _map_jackhhao),
        ("lmsys/toxic-chat", "toxicchat0124", _map_toxic_chat),
    ]

    def load(self) -> list[tuple[str, str]]:
        from datasets import load_dataset  # imported lazily; optional heavy dep

        seen: set[str] = set()
        examples: list[tuple[str, str]] = []
        for name, config, map_row in self.SOURCES:
            for split in load_dataset(name, config).values():
                for row in split:
                    text, label = map_row(row)
                    text = str(text or "").strip()
                    if text and text not in seen:
                        seen.add(text)
                        examples.append((text, label))
        return examples


# TODO: TrafficLogSource — build training examples from logged /chat prompts
# once request logging + a labeling workflow exist, to support retraining on
# real production traffic rather than only public datasets.
