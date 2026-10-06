"""Hugging Face Hub as the model registry for the input classifier.

Usage:
    python -m src.trainer.hub push [--version vX] [--candidate]   # upload a local version (default: the active one);
                                                                 # --candidate keeps the Hub's latest unchanged
    python -m src.trainer.hub pull [--version vX]   # download a version (default: hub latest) and activate it
    python -m src.trainer.hub list

Repo (private, HF_MODEL_REPO) layout mirrors ML_MODEL_DIR:
    <version>/model.joblib, vectorizer.joblib, metadata.json
    <version>/eval.json, benchmark.json        # if reports/ has them
    pointer.json                               # latest pushed version
    README.md                                  # model card, regenerated on push
Every push is also a git commit on the Hub, so history is kept there too.
Needs HF_TOKEN (write scope) in .env.
"""
import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from src.config import settings
from src.trainer.train import _write_pointer

MODEL_DIR = Path(settings.ML_MODEL_DIR)
REPORTS_DIR = Path("reports")


def _api() -> HfApi:
    if not settings.HF_TOKEN:
        raise SystemExit("HF_TOKEN missing from .env")
    return HfApi(token=settings.HF_TOKEN)


def _active_version() -> str:
    return json.loads((MODEL_DIR / "pointer.json").read_text())["current_version"]


def _card(version: str, metadata: dict, bench: dict | None) -> str:
    lines = [
        "---", "library_name: sklearn", "tags: [prompt-injection, jailbreak, toxicity, guardrails]", "---", "",
        "# GuardProxy input classifier", "",
        "TF-IDF + logistic regression, one binary head per attack class (prompt_injection, jailbreak, toxic).",
        "Served by GuardProxy's input guardrails; pulled with `python -m src.trainer.hub pull`.", "",
        f"Latest version: **{version}** (trained {metadata['trained_at'][:10]}, sklearn {metadata['sklearn_version']}).", "",
        "## Held-out F1 per head", "",
        *[f"- {h}: {f1:.3f}" for h, f1 in metadata["per_class_f1"].items()], "",
    ]
    if bench:
        lines += [
            "## External benchmarks (never trained on)", "",
            "Score = PINT balanced accuracy. `guard` = regex checks + ML, as served.", "",
            "| Benchmark | Prompts | Detector | Score | Attack recall | FPR |", "|---|---|---|---|---|---|",
        ]
        for name, b in bench["benchmarks"].items():
            for det in ("ml_only", "guard"):
                s = b[det]
                lines.append(f"| {name} | {b['n']} | {det} | {s['score']} | {s['attack_recall']} | {s['false_positive_rate']} |")
    return "\n".join(lines) + "\n"


def push(version: str | None, candidate: bool = False) -> None:
    api = _api()
    version = version or _active_version()
    vdir = MODEL_DIR / version
    metadata = json.loads((vdir / "metadata.json").read_text())
    api.create_repo(settings.HF_MODEL_REPO, private=True, exist_ok=True)

    api.upload_folder(repo_id=settings.HF_MODEL_REPO, folder_path=vdir, path_in_repo=version,
                      commit_message=f"Add {version}")
    bench = None
    for kind in ("eval", "benchmark"):
        report = REPORTS_DIR / f"{kind}-{version}.json"
        if report.exists():
            api.upload_file(repo_id=settings.HF_MODEL_REPO, path_or_fileobj=report,
                            path_in_repo=f"{version}/{kind}.json", commit_message=f"{kind} report for {version}")
            if kind == "benchmark":
                bench = json.loads(report.read_text())
    if candidate:  # stored for comparison/rollback, but "latest" and the model card stay put
        print(f"Pushed candidate {version} -> https://huggingface.co/{settings.HF_MODEL_REPO}")
        return
    api.upload_file(repo_id=settings.HF_MODEL_REPO, path_in_repo="pointer.json",
                    path_or_fileobj=json.dumps({"current_version": version}, indent=2).encode(),
                    commit_message=f"Point latest at {version}")
    api.upload_file(repo_id=settings.HF_MODEL_REPO, path_in_repo="README.md",
                    path_or_fileobj=_card(version, metadata, bench).encode(),
                    commit_message=f"Model card for {version}")
    print(f"Pushed {version} -> https://huggingface.co/{settings.HF_MODEL_REPO}")


def pull(version: str | None) -> None:
    api = _api()
    if version is None:
        path = api.hf_hub_download(settings.HF_MODEL_REPO, "pointer.json")
        version = json.loads(Path(path).read_text())["current_version"]
    # Whole version folder: joblib files for TF-IDF versions, model.onnx + tokenizer.json for transformers.
    snapshot_download(settings.HF_MODEL_REPO, token=settings.HF_TOKEN, local_dir=MODEL_DIR,
                      allow_patterns=[f"{version}/*"])
    _write_pointer(MODEL_DIR, version)
    print(f"Pulled {version}; it is now the active input classifier.")


def list_versions() -> None:
    files = _api().list_repo_files(settings.HF_MODEL_REPO)
    for v in sorted({f.split("/")[0] for f in files if f.startswith("v") and "/" in f}):
        print(v)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync input classifier versions with the Hugging Face Hub.")
    parser.add_argument("command", choices=["push", "pull", "list"])
    parser.add_argument("--version")
    parser.add_argument("--candidate", action="store_true",
                        help="push: upload without making it the Hub's latest or rewriting the model card")
    args = parser.parse_args()
    {"push": lambda: push(args.version, args.candidate), "pull": lambda: pull(args.version), "list": list_versions}[args.command]()


if __name__ == "__main__":
    main()
