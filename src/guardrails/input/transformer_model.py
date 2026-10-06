"""CPU inference for the fine-tuned transformer input classifier (ONNX).

Artifact (written by src.trainer.transformer): model.onnx, tokenizer.json,
metadata.json with kind="transformer", heads, max_len and optional per-head
logit_bias (calibration: subtracting it makes 0.5 the tuned threshold). Only onnxruntime
and tokenizers are needed at serve time, no torch.
"""
from pathlib import Path

import numpy as np

MAX_WINDOWS = 8  # ponytail: caps cost on huge prompts; text past 8 windows (~1.5k tokens) is unscored


class TransformerHeads:
    def __init__(self, version_dir: Path, heads: list[str], max_len: int, logit_bias: dict | None = None):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4
        self.session = ort.InferenceSession(str(version_dir / "model.onnx"), opts, providers=["CPUExecutionProvider"])
        self.tokenizer = Tokenizer.from_file(str(version_dir / "tokenizer.json"))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        self.heads = heads
        self.max_len = max_len
        self.bias = np.array([(logit_bias or {}).get(h, 0.0) for h in heads], dtype=np.float32)

    def _windows(self, text: str) -> list[list[int]]:
        """Long prompts are split into overlapping windows so an injection
        buried at the end of a document is still seen."""
        ids = self.tokenizer.encode(text).ids
        cls, body, sep = ids[0], ids[1:-1], ids[-1]
        w = self.max_len - 2
        starts = range(0, max(len(body) - w, 0) + 1, w // 2)
        windows = [[cls, *body[s:s + w], sep] for s in starts][:MAX_WINDOWS]
        if len(body) > w and starts[-1] + w < len(body) and len(windows) < MAX_WINDOWS:
            windows.append([cls, *body[-w:], sep])
        return windows

    def score(self, text: str) -> dict[str, float]:
        windows = self._windows(text)
        n = max(len(w) for w in windows)
        ids = np.zeros((len(windows), n), dtype=np.int64)
        mask = np.zeros_like(ids)
        for i, w in enumerate(windows):
            ids[i, :len(w)] = w
            mask[i, :len(w)] = 1
        logits = self.session.run(None, {"input_ids": ids, "attention_mask": mask})[0]
        probs = 1 / (1 + np.exp(-(logits - self.bias)))
        return {h: float(probs[:, i].max()) for i, h in enumerate(self.heads)}
