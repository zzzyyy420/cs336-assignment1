# Assignment 3: Scaling

The scaling package contains the experiment API, training infrastructure, and IsoFLOPs analysis. Each command runs from this directory.

```sh
uv sync --extra analysis
uv run python -m cs336_scaling.isoflop
```

The API dependencies are installed with `uv sync --extra server`. The API uses the database settings read by `cs336_scaling.config`.
