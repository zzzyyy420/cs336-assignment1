from functools import lru_cache
from pathlib import Path

import fasttext


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "jigsaw_fasttext_bigrams_hatespeech_final.bin"


@lru_cache(maxsize=1)
def _model():
    return fasttext.load_model(str(MODEL_PATH))


def run_classify_toxic_speech(text: str) -> tuple[str, float]:
    labels, scores = _model().predict(" ".join(text.split()))
    return labels[0].removeprefix("__label__"), float(scores[0])
