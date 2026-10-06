"""Fine-tune a small transformer as the input classifier and ship it as ONNX.

Usage:
    python -m src.trainer.transformer [--base microsoft/deberta-v3-small] [--epochs 2]
                                      [--max-len 256] [--batch-size 16] [--lr 3e-5]

Same contract as src.trainer.train: three sigmoid heads (prompt_injection,
jailbreak, toxic) on one encoder, trained on the commercial-safe
huggingface source minus every benchmark prompt, with the same 80/20
split (so evaluate.py's held-out section stays valid). It ships only if
every head clears the F1 gate and the hand-written eval set clears the
block-accuracy gate; then pointer.json is repointed and the running proxy
hot-swaps to it. Needs a CUDA GPU for reasonable speed; serving is CPU-only.
"""
import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from src.config import settings
from src.guardrails.input.transformer_model import TransformerHeads
from src.trainer.model import HEADS
from src.trainer.train import EVAL_PROMPTS_PATH, _prune_old_versions, _write_pointer, load_training_examples


def _targets(labels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Multi-hot targets plus a loss mask: like the TF-IDF heads, a head learns
    its own class vs safe only, so other attack classes are masked out."""
    y = np.array([[l == h for h in HEADS] for l in labels], dtype=np.float32)
    mask = np.array([[l in (h, "safe") for h in HEADS] for l in labels], dtype=np.float32)
    return y, mask


def _batches(lengths: list[int], batch_size: int, rng: random.Random) -> list[list[int]]:
    """Length-bucketed shuffled batches: cuts padding, ~3x faster than random batches."""
    idx = list(range(len(lengths)))
    rng.shuffle(idx)
    chunk = batch_size * 50
    batches = []
    for c in range(0, len(idx), chunk):
        part = sorted(idx[c:c + chunk], key=lambda i: lengths[i])
        batches += [part[b:b + batch_size] for b in range(0, len(part), batch_size)]
    rng.shuffle(batches)
    return batches


def _collate(enc_ids: list[list[int]], rows: list[int], pad_id: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    n = max(len(enc_ids[i]) for i in rows)
    ids = torch.full((len(rows), n), pad_id, dtype=torch.long)
    att = torch.zeros((len(rows), n), dtype=torch.long)
    for r, i in enumerate(rows):
        ids[r, :len(enc_ids[i])] = torch.tensor(enc_ids[i])
        att[r, :len(enc_ids[i])] = 1
    return ids.to(device), att.to(device)


@torch.no_grad()
def _predict(model, enc_ids, pad_id, device, batch_size=64) -> np.ndarray:
    model.eval()
    order = sorted(range(len(enc_ids)), key=lambda i: len(enc_ids[i]))
    out = np.zeros((len(enc_ids), len(HEADS)), dtype=np.float32)
    for b in range(0, len(order), batch_size):
        rows = order[b:b + batch_size]
        ids, att = _collate(enc_ids, rows, pad_id, device)
        with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            logits = model(input_ids=ids, attention_mask=att).logits
        out[rows] = torch.sigmoid(logits.float()).cpu().numpy()
    return out


def _tune_thresholds(probs: np.ndarray, labels: list[str]) -> dict[str, float]:
    """Per-head threshold maximising F1 (head vs safe) on the validation slice.
    Class weighting in the loss inflates probabilities, so 0.5 over-blocks."""
    labels = np.array(labels)
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2)
    tuned = {}
    for i, h in enumerate(HEADS):
        keep = np.isin(labels, [h, "safe"])
        y, p = labels[keep] == h, probs[keep, i]
        tuned[h] = float(max(grid, key=lambda t: f1_score(y, p >= t, zero_division=0)))
    return tuned


class _LogitsOnly(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).logits


def _export_onnx(model, path: Path) -> None:
    model = _LogitsOnly(model.float().cpu().eval())
    dummy = (torch.ones((2, 16), dtype=torch.long), torch.ones((2, 16), dtype=torch.long))
    torch.onnx.export(
        model, dummy, str(path), input_names=["input_ids", "attention_mask"], output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"}, "logits": {0: "batch"}},
        opset_version=17, dynamo=False,
    )


def _eval_set_accuracy(heads: TransformerHeads, threshold: float) -> tuple[float, list[str]]:
    df = pd.read_csv(EVAL_PROMPTS_PATH)
    misses = []
    for text, label in zip(df["text"], df["label"]):
        p = heads.score(text)
        top = max(p, key=p.get)
        if (p[top] >= threshold) != (label != "safe"):
            misses.append(f"expected {label:<16} got {'BLOCK ' + top if p[top] >= threshold else 'allow'} (p={p[top]:.2f}): {text}")
    return 1 - len(misses) / len(df), misses


def train(args) -> int:
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    examples = load_training_examples("huggingface", "")
    texts = [t for t, _ in examples]
    labels = [l for _, l in examples]
    counts = {l: labels.count(l) for l in sorted(set(labels))}
    print(f"{len(examples)} examples: {counts}")
    # Must match evaluate.heldout_metrics's split.
    x_tr, x_te, y_tr, y_te = train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels)
    # Validation slice for threshold tuning, so the held-out test stays untouched.
    x_tr, x_val, y_tr, y_val = train_test_split(x_tr, y_tr, test_size=0.1, random_state=42, stratify=y_tr)

    tok = AutoTokenizer.from_pretrained(args.base)
    enc_tr = tok(x_tr, truncation=True, max_length=args.max_len)["input_ids"]
    enc_val = tok(x_val, truncation=True, max_length=args.max_len)["input_ids"]
    enc_te = tok(x_te, truncation=True, max_length=args.max_len)["input_ids"]
    t_tr, m_tr = _targets(y_tr)
    t_te, _ = _targets(y_te)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base, num_labels=len(HEADS), problem_type="multi_label_classification", dtype=torch.float32
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * ((len(enc_tr) + args.batch_size - 1) // args.batch_size)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")
    # Up-weight rare heads (jailbreak is ~2% of rows): sqrt(neg/pos), capped.
    pos = (t_tr * m_tr).sum(0)
    pos_weight = torch.tensor(np.clip(np.sqrt((m_tr.sum(0) - pos) / np.maximum(pos, 1)), 1, 10), dtype=torch.float32, device=device)
    print(f"pos_weight per head: {dict(zip(HEADS, pos_weight.tolist()))}")
    loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)
    lengths = [len(e) for e in enc_tr]

    step, start = 0, time.time()
    for epoch in range(args.epochs):
        model.train()
        for rows in _batches(lengths, args.batch_size, rng):
            ids, att = _collate(enc_tr, rows, tok.pad_token_id, device)
            y = torch.tensor(t_tr[rows], device=device)
            m = torch.tensor(m_tr[rows], device=device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
                logits = model(input_ids=ids, attention_mask=att).logits
            loss = (loss_fn(logits.float(), y) * m).sum() / m.sum().clamp(min=1)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            if step % 200 == 0:
                print(f"epoch {epoch} step {step}/{steps} loss {loss.item():.4f} ({time.time() - start:.0f}s)", flush=True)

    thresholds = _tune_thresholds(_predict(model, enc_val, tok.pad_token_id, device), y_val)
    logit_bias = {h: float(np.log(t / (1 - t))) for h, t in thresholds.items()}
    print(f"tuned thresholds (validation slice): {thresholds}")

    probs = _predict(model, enc_te, tok.pad_token_id, device)
    y_te_arr = np.array(y_te)
    hit = np.column_stack([probs[:, i] >= thresholds[h] for i, h in enumerate(HEADS)])
    per_head_f1 = {}
    for i, h in enumerate(HEADS):
        keep = np.isin(y_te_arr, [h, "safe"])
        per_head_f1[h] = float(f1_score(t_te[keep, i], hit[keep, i], zero_division=0))
    blocked = hit.any(axis=1)
    accuracy = float((blocked == (y_te_arr != "safe")).mean())
    print(f"held-out block accuracy={accuracy:.4f} per-head F1={per_head_f1}")

    version = "v" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    model_dir = Path(settings.ML_MODEL_DIR)
    version_dir = model_dir / version
    version_dir.mkdir(parents=True)
    _export_onnx(model, version_dir / "model.onnx")
    tok.backend_tokenizer.save(str(version_dir / "tokenizer.json"))

    heads = TransformerHeads(version_dir, HEADS, args.max_len, logit_bias)
    eval_acc, misses = _eval_set_accuracy(heads, settings.ML_CLASSIFIER_THRESHOLD)
    print(f"eval set block accuracy={eval_acc:.3f}")
    for miss in misses:
        print(f"  {miss}")

    metadata = {
        "version": version, "kind": "transformer", "base_model": args.base, "heads": HEADS, "max_len": args.max_len,
        "thresholds": thresholds, "logit_bias": logit_bias,
        "trained_at": datetime.now(timezone.utc).isoformat(), "source": "huggingface",
        "n_examples_loaded": len(examples), "label_counts": counts,
        "hyperparams": {"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr, "seed": args.seed},
        "accuracy": accuracy, "macro_f1": sum(per_head_f1.values()) / len(per_head_f1), "per_class_f1": per_head_f1,
        "eval_block_accuracy": eval_acc, "eval_misses": misses,
        "n_train": len(x_tr), "n_test": len(x_te), "sklearn_version": "n/a (transformer)",
    }
    (version_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    weakest = min(per_head_f1, key=per_head_f1.get)
    if per_head_f1[weakest] < args.min_f1 or eval_acc < args.min_eval_accuracy:
        print(f"REJECTED: {weakest} F1 {per_head_f1[weakest]:.3f} (min {args.min_f1}) / eval acc {eval_acc:.3f} "
              f"(min {args.min_eval_accuracy}). Saved at {version_dir} but NOT activated.")
        return 1
    _write_pointer(model_dir, version)
    _prune_old_versions(model_dir, args.keep_versions)
    print(f"SHIPPED: {version} is now the active input classifier.")
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="Fine-tune the transformer input classifier.")
    p.add_argument("--base", default="microsoft/deberta-v3-small")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--seed", type=int, default=0, help="Init/shuffle seed (data split is fixed); vary it to measure run-to-run noise.")
    p.add_argument("--min-f1", type=float, default=settings.ML_MIN_MACRO_F1)
    p.add_argument("--min-eval-accuracy", type=float, default=settings.ML_MIN_EVAL_ACCURACY)
    p.add_argument("--keep-versions", type=int, default=8)
    sys.exit(train(p.parse_args()))


if __name__ == "__main__":
    main()
