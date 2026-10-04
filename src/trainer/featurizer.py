from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion

from src.guardrails.input.normalize import normalize


def build_vectorizer() -> FeatureUnion:
    # Word n-grams capture phrasing; char n-grams catch partial-word
    # paraphrases. normalize() (NFKC, invisible chars, leetspeak, lowercase)
    # runs as the preprocessor, so it's pickled with the model and training
    # and inference can't drift apart.
    return FeatureUnion([
        ("word", TfidfVectorizer(
            preprocessor=normalize,
            ngram_range=(1, 2),
            max_features=20_000,
            sublinear_tf=True,
        )),
        ("char", TfidfVectorizer(
            preprocessor=normalize,
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=50_000,
            sublinear_tf=True,
        )),
    ])
