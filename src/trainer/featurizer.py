from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion


def build_vectorizer() -> FeatureUnion:
    # Word n-grams capture phrasing; char n-grams catch obfuscation
    # ("ign0re prev1ous") and partial-word paraphrases.
    return FeatureUnion([
        ("word", TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            max_features=20_000,
            sublinear_tf=True,
        )),
        ("char", TfidfVectorizer(
            lowercase=True,
            analyzer="char_wb",
            ngram_range=(3, 5),
            max_features=50_000,
            sublinear_tf=True,
        )),
    ])
