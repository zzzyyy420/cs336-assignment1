"""Token batches shared by the training profiling scripts."""

import pickle
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch
from cs336_basics.data import get_batch


def load_token_ids(path: str | Path) -> np.ndarray:
    with Path(path).open("rb") as file:
        token_ids = np.asarray(pickle.load(file), dtype=np.int64)
    if token_ids.ndim != 1:
        raise ValueError("token IDs must be a one-dimensional sequence")
    return token_ids


def training_batch(
    token_ids: np.ndarray, batch_size: int, context_length: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(token_ids) <= context_length:
        raise ValueError("token sequence must be longer than context_length")
    return get_batch(token_ids, batch_size, context_length, str(device))


def validation_batches(
    token_ids: np.ndarray, batch_size: int, context_length: int, device: torch.device
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    if len(token_ids) <= context_length:
        raise ValueError("token sequence must be longer than context_length")
    last_start = len(token_ids) - context_length - 1
    for offset in range(0, last_start + 1, batch_size * context_length):
        starts = range(offset, min(offset + batch_size * context_length, last_start + 1), context_length)
        inputs = np.stack([token_ids[start : start + context_length] for start in starts])
        targets = np.stack([token_ids[start + 1 : start + context_length + 1] for start in starts])
        yield torch.from_numpy(inputs).to(device), torch.from_numpy(targets).to(device)
