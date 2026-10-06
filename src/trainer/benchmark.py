"""Score the active input guard on external, public benchmarks it was never trained on.

Usage:
    python -m src.trainer.benchmark [--pint-yaml PATH] [--out reports]

Benchmarks (none overlap the training sources in src.trainer.data.loader):
  - pint:      Lakera / Check Point PINT benchmark format. Defaults to the public
               example set from github.com/CheckPointSW/pint-benchmark; the full
               4,314-prompt PINT set is access-on-request, pass it via --pint-yaml.
  - jbb:       JailbreakBench JBB-Behaviors (100 harmful vs 100 benign requests).
  - safeguard: xTRam1/safe-guard-prompt-injection test split (~2.1k prompts).
  - gandalf:   Lakera gandalf_ignore_instructions (1k real injections, recall only).
  - xstest:    XSTest (250 safe-but-scary-sounding prompts vs 200 unsafe): over-blocking.

Every benchmark is scored the way PINT scores its leaderboard: accuracy per
label (attack / benign), averaged, so the published PINT numbers for Lakera
Guard, Prompt Guard, etc. are directly comparable on the PINT set. Two
detectors are scored: the ML classifier alone and the full input guard
(regex checks + ML), which is what /chat actually runs.

Writes reports/benchmark-<version>.json and .md; `src.trainer.hub push`
attaches them to the model on the Hugging Face Hub.
"""
import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np
import yaml

from src.config import settings
from src.guardrails.input.model_registry import get_registry
from src.trainer.evaluate import _ml_blocks, _regex_blocks

PINT_EXAMPLE_URL = "https://raw.githubusercontent.com/CheckPointSW/pint-benchmark/main/benchmark/data/example-dataset.yaml"
CACHE_DIR = Path("data/benchmarks")


def load_pint(path: str | None) -> tuple[list[str], np.ndarray, list[str]]:
    if path is None:
        cached = CACHE_DIR / "pint-example-dataset.yaml"
        if not cached.exists():
            cached.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(PINT_EXAMPLE_URL, cached)
        path = cached
    rows = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return [str(r["text"]) for r in rows], np.array([bool(r["label"]) for r in rows]), [r["category"] for r in rows]


def load_jbb() -> tuple[list[str], np.ndarray, list[str]]:
    from datasets import load_dataset

    ds = load_dataset("JailbreakBench/JBB-Behaviors", "behaviors")
    texts, labels, cats = [], [], []
    for split, is_attack in (("harmful", True), ("benign", False)):
        for row in ds[split]:
            texts.append(row["Goal"])
            labels.append(is_attack)
            cats.append(f"{split}/{row['Category']}")
    return texts, np.array(labels), cats


def load_safeguard() -> tuple[list[str], np.ndarray, list[str]]:
    from datasets import load_dataset

    ds = load_dataset("xTRam1/safe-guard-prompt-injection", split="test")
    labels = np.array([r == 1 for r in ds["label"]])
    return list(ds["text"]), labels, ["prompt_injection" if l else "benign" for l in labels]


def load_gandalf() -> tuple[list[str], np.ndarray, list[str]]:
    from datasets import load_dataset

    texts = [r["text"] for split in load_dataset("Lakera/gandalf_ignore_instructions").values() for r in split]
    return texts, np.ones(len(texts), dtype=bool), ["prompt_injection"] * len(texts)


def load_xstest() -> tuple[list[str], np.ndarray, list[str]]:
    from datasets import load_dataset

    ds = load_dataset("Paul/XSTest", split="train")
    return list(ds["prompt"]), np.array([l == "unsafe" for l in ds["label"]]), [f"{l}/{t}" for l, t in zip(ds["label"], ds["type"])]


LOADERS = {"jbb": load_jbb, "safeguard": load_safeguard, "gandalf": load_gandalf, "xstest": load_xstest}


def _norm(text: str) -> str:
    return " ".join(text.lower().split())


def benchmark_texts() -> set[str]:
    """Normalized text of every benchmark prompt, so trainers can keep them out of training data."""
    texts = load_pint(None)[0]
    for load in LOADERS.values():
        texts += load()[0]
    return {_norm(t) for t in texts}


def score(pred: np.ndarray, labels: np.ndarray, cats: list[str]) -> dict:
    cats = np.array(cats)
    attack_acc = float(pred[labels].mean())
    # Attack-only sets (gandalf) have no benign rows: score is plain recall.
    benign_acc = float((~pred[~labels]).mean()) if (~labels).any() else None
    return {
        "score": round((attack_acc + benign_acc) / 2 if benign_acc is not None else attack_acc, 4),  # PINT "balanced" score
        "attack_recall": round(attack_acc, 4),
        "false_positive_rate": round(1 - benign_acc, 4) if benign_acc is not None else None,
        "by_category": {str(c): round(float((pred[cats == c] == labels[cats == c]).mean()), 4) for c in sorted(set(cats))},
    }


def run(pint_yaml: str | None, version: str | None = None) -> dict:
    loaded = get_registry().get(version)
    if loaded is None or not (isinstance(loaded.model, dict) or hasattr(loaded.model, "score")):
        raise SystemExit("No active 3-head model found via pointer.json.")
    threshold = settings.ML_CLASSIFIER_THRESHOLD
    benchmarks = {
        "pint" if pint_yaml else "pint_example": load_pint(pint_yaml),
        **{name: load() for name, load in LOADERS.items()},
    }
    results = {}
    for name, (texts, labels, cats) in benchmarks.items():
        ml = _ml_blocks(loaded, texts, threshold)
        results[name] = {
            "n": len(texts),
            "n_attack": int(labels.sum()),
            "ml_only": score(ml, labels, cats),
            "guard": score(ml | _regex_blocks(texts), labels, cats),
        }
    return {"version": loaded.version, "threshold": threshold, "benchmarks": results}


def markdown(r: dict) -> str:
    lines = [
        f"# External benchmarks — {r['version']}",
        "",
        f"Threshold {r['threshold']}. Score = PINT balanced accuracy (mean of attack recall and benign accuracy).",
        "Generated by `python -m src.trainer.benchmark`.",
        "",
        "| Benchmark | Prompts (attacks) | Detector | Score | Attack recall | False positive rate |",
        "|---|---|---|---|---|---|",
    ]
    for name, b in r["benchmarks"].items():
        for det in ("ml_only", "guard"):
            s = b[det]
            lines.append(f"| {name} | {b['n']} ({b['n_attack']}) | {det} | {s['score']} | {s['attack_recall']} | {s['false_positive_rate']} |")
    lines += ["", "Per-category accuracy (guard):", ""]
    for name, b in r["benchmarks"].items():
        lines.append(f"- **{name}**: {b['guard']['by_category']}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Score the active input guard on external benchmarks.")
    parser.add_argument("--pint-yaml", help="Full PINT dataset YAML (access on request from Lakera/Check Point).")
    parser.add_argument("--version", help="Benchmark this version instead of the active one.")
    parser.add_argument("--out", default="reports")
    args = parser.parse_args()

    report = run(args.pint_yaml, args.version)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"benchmark-{report['version']}.json").write_text(json.dumps(report, indent=2))
    (out / f"benchmark-{report['version']}.md").write_text(markdown(report), encoding="utf-8")
    print(markdown(report))


if __name__ == "__main__":
    main()
