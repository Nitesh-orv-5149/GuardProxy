from types import SimpleNamespace

from src.guardrails.input.transformer_model import MAX_WINDOWS, TransformerHeads


def _heads(n_tokens: int, max_len: int = 10) -> TransformerHeads:
    h = TransformerHeads.__new__(TransformerHeads)  # skip ONNX/tokenizer loading
    h.max_len = max_len
    ids = [0, *range(1, n_tokens + 1), 99]  # CLS, body, SEP
    h.tokenizer = SimpleNamespace(encode=lambda text: SimpleNamespace(ids=ids))
    return h


def test_short_prompt_is_one_window():
    assert _heads(5)._windows("x") == [[0, 1, 2, 3, 4, 5, 99]]


def test_long_prompt_windows_cover_the_end():
    windows = _heads(30)._windows("x")
    assert all(w[0] == 0 and w[-1] == 99 and len(w) <= 10 for w in windows)
    assert windows[-1][-2] == 30  # last body token is scored
    assert len(windows) <= MAX_WINDOWS
