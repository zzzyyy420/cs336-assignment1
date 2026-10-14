from functools import lru_cache
from pathlib import Path

import fasttext


MODEL_PATH = Path(__file__).resolve().parent / "quality_classifier.bin"
LABELS = {"hq": "wiki", "lq": "cc"}


@lru_cache(maxsize=1)
def _model():
    return fasttext.load_model(str(MODEL_PATH))


def run_classify_quality(text: str) -> tuple[str, float]:
    labels, scores = _model().predict(" ".join(text.split()))
    label = labels[0].removeprefix("__label__")
    return LABELS.get(label, label), float(scores[0])
