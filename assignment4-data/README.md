# Assignment 4: Data

The data package contains text extraction, filtering, deduplication, and fastText classification interfaces. From this directory:

```sh
uv sync
uv run pytest tests -q
```

The classifiers read their model files from `models/`; the quality training script writes `cs336_basics/quality_classify/quality_classifier.bin`.
