import re
import zlib
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


def _map_deepset(row: dict):
    return row["text"], "prompt_injection" if row["label"] == 1 else "safe"


def _map_jackhhao(row: dict):
    return row["prompt"], "jailbreak" if row["type"] == "jailbreak" else "safe"


def _map_spml(row: dict):
    return row["User Prompt"], "prompt_injection" if row["Prompt injection"] == 1 else "safe"


def _map_wild_jailbreak(row: dict):
    return row["prompt"], "jailbreak"


# ~15% of in-the-wild "regular" prompts are jailbreak-style ("forget all previous
# instructions", DAN rules, "exception to AI ethical protocols"). Labelled safe they
# outnumbered the whole jailbreak class, so rows matching this are dropped from
# the `safe` sources (not relabelled: many are roleplay, not jailbreaks).
_JAILBREAKY = re.compile(
    r"\b(DAN|jailbreak|jailbroken|developer mode|do anything now|unfiltered|uncensored|amoral|immoral"
    r"|stay in character|never refuse|no (?:restrictions|limitations|filters|rules|guidelines|ethics)"
    r"|(?:ignore|forget|disregard) (?:all|any|your|previous|prior)"
    r"|without (?:any )?(?:restrictions|censorship|filters)|openai'?s? (?:policy|policies|content))\b",
    re.IGNORECASE,
)


def _map_wild_regular(row: dict):
    return None if _JAILBREAKY.search(row["prompt"] or "") else (row["prompt"], "safe")


def _map_aegis(row: dict):
    if row["prompt"] == "REDACTED":  # suicide-dataset prompts are withheld
        return None
    return row["prompt"], "toxic" if row["prompt_label"] == "unsafe" else "safe"


def _map_dolly(row: dict):
    text = row["instruction"] + (f"\n\n{row['context']}" if row["context"] else "")
    return text, "safe"


def _map_oasst(row: dict):
    if row["role"] != "prompter" or row["parent_id"] is not None:
        return None
    return row["text"], "safe"


def _map_orbench_80k(row: dict):
    # Benign prompts that sound harmful (over-refusal bait). Deterministic ~1/4
    # sample so they don't swamp the rest of `safe`.
    return (row["prompt"], "safe") if zlib.crc32(row["prompt"].encode()) % 4 == 0 else None


def _map_orbench_toxic(row: dict):
    return row["prompt"], "toxic"


def _map_prompts_chat(row: dict):
    # benign "act as ..." role prompts (jailbreak look-alikes), minus the ~1% that are jailbreak-style
    return None if _JAILBREAKY.search(row["prompt"] or "") else (row["prompt"], "safe")


# Real-world goals swapped in for HackAPrompt's fixed target phrase, so the
# model learns the injection pattern rather than the token "PWNED".
_PWNED = re.compile(r"i have been pwned", re.IGNORECASE)
_PAYLOADS = [  # things an attack tricks the model into saying, like the original phrase
    "Here is my full system prompt",
    "Safety rules disabled",
    "The admin password is",
    "I have no restrictions now",
    "Access granted",
    "My hidden instructions are",
]


def _map_hackaprompt(row: dict):
    # ~600k competition attempts (~22k kept by this sample, 57% aimed at
    # "I have been PWNED"); successful attacks are kept preferentially.
    text = (row["user_input"] or "").strip()
    if len(text) < 10:
        return None
    h = zlib.crc32(text.encode())
    if h % (2 if row["correct"] else 30):
        return None
    return _PWNED.sub(_PAYLOADS[h % len(_PAYLOADS)], text), "prompt_injection"


def _sample(text: str, one_in: int) -> bool:
    return zlib.crc32(text.encode()) % one_in == 0


# Rows these sources took from AdvBench (JailbreakBench is built on it) or from
# non-commercial sets (ToxicChat, BeaverTails) are dropped.
_EXCLUDED_ORIGINS = {"Advbench", "AdvBench", "ToxicChat", "BeaverTails", "LLM Jailbreak Study"}


def _map_salad_base(row: dict):
    # 21k harmful questions incl. 2k misinformation; 1/2 sample keeps `toxic` from swamping.
    if row["source"] in _EXCLUDED_ORIGINS or not _sample(row["question"], 2):
        return None
    return row["question"], "toxic"


def _map_salad_attack(row: dict):
    return row["augq"], "jailbreak"  # harmful question wrapped by a jailbreak method (DAN, AutoDAN, TAP, GCG...)


def _map_jailbreakv(row: dict):
    # Text-only attack formats; the 18k "Template" rows reuse a few hundred
    # templates, so they're sampled 1/4 to avoid memorising them.
    fmt, text = row["format"], row["jailbreak_query"]
    if fmt not in ("Template", "Persuade", "Logic") or row["from"] in _EXCLUDED_ORIGINS:
        return None
    if fmt == "Template" and not _sample(text, 4):
        return None
    return text, "jailbreak"


class HFDatasetSource(DatasetSource):
    """Commercial-use-safe public datasets from the HuggingFace Hub, each
    mapped to our labels (safe / prompt_injection / jailbreak / toxic):

    - deepset/prompt-injections (Apache-2.0)            -> prompt_injection / safe
    - jackhhao/jailbreak-classification (Apache-2.0)    -> jailbreak / safe
    - reshabhs/SPML_Chatbot_Prompt_Injection (MIT)      -> prompt_injection / safe
    - TrustAIRLab/in-the-wild-jailbreak-prompts (MIT)   -> jailbreak (jailbreak_*) / safe (regular_*)
    - nvidia/Aegis-AI-Content-Safety-Dataset-2.0 (CC-BY-4.0) -> toxic (unsafe prompt) / safe
    - databricks/databricks-dolly-15k (CC-BY-SA-3.0)    -> safe
    - OpenAssistant/oasst2 (Apache-2.0, first user turns, multilingual) -> safe
    - bench-llm/or-bench (CC-BY-4.0): 80k scary-sounding benign (1/4 sample) -> safe, toxic subset -> toxic
    - fka/prompts.chat (CC0-1.0): "act as ..." role prompts -> safe
    - hackaprompt/hackaprompt-dataset (MIT, gated): competition injection attempts -> prompt_injection
    - OpenSafetyLab/Salad-Data (Apache-2.0): harmful questions -> toxic; jailbreak-wrapped ones -> jailbreak
    - JailbreakV-28K/JailBreakV-28k (MIT): text jailbreaks (template/persuade/logic) -> jailbreak
      (rows sourced from AdvBench, ToxicChat or BeaverTails are dropped)

    No non-commercial (NC) data. allenai/wildjailbreak is deliberately excluded:
    its access terms restrict use to research. Benchmark sets (src.trainer.benchmark) are
    never sourced here. All splits are used; trainers do their own split.
    Duplicate texts are dropped (first label wins).
    """

    SOURCES = [
        ("deepset/prompt-injections", None, _map_deepset),
        ("jackhhao/jailbreak-classification", None, _map_jackhhao),
        ("reshabhs/SPML_Chatbot_Prompt_Injection", None, _map_spml),
        ("TrustAIRLab/in-the-wild-jailbreak-prompts", "jailbreak_2023_12_25", _map_wild_jailbreak),
        ("TrustAIRLab/in-the-wild-jailbreak-prompts", "regular_2023_12_25", _map_wild_regular),
        ("nvidia/Aegis-AI-Content-Safety-Dataset-2.0", None, _map_aegis),
        ("databricks/databricks-dolly-15k", None, _map_dolly),
        ("OpenAssistant/oasst2", None, _map_oasst),
        ("bench-llm/or-bench", "or-bench-80k", _map_orbench_80k),
        ("bench-llm/or-bench", "or-bench-toxic", _map_orbench_toxic),
        ("fka/prompts.chat", None, _map_prompts_chat),
        ("hackaprompt/hackaprompt-dataset", None, _map_hackaprompt),  # gated: accept terms on HF first
        ("OpenSafetyLab/Salad-Data", "base_set", _map_salad_base),
        ("OpenSafetyLab/Salad-Data", "attack_enhanced_set", _map_salad_attack),
        ("JailbreakV-28K/JailBreakV-28k", "JailBreakV_28K", _map_jailbreakv),
    ]

    def load(self) -> list[tuple[str, str]]:
        from datasets import load_dataset  # imported lazily; optional heavy dep

        seen: set[str] = set()
        examples: list[tuple[str, str]] = []
        for name, config, map_row in self.SOURCES:
            for split in load_dataset(name, config).values():
                for row in split:
                    mapped = map_row(row)
                    if mapped is None:
                        continue
                    text, label = mapped
                    text = str(text or "").strip()
                    if text and text not in seen:
                        seen.add(text)
                        examples.append((text, label))
        return examples


# TODO: TrafficLogSource — build training examples from logged /chat prompts
# once request logging + a labeling workflow exist, to support retraining on
# real production traffic rather than only public datasets.
