"""Sampling helper shared with the Assignment 1 generator."""

import torch

from cs336_basics.generate import generate_tokens


def decode_token(input_tokens, model, max_tokens_to_generate, top_p=0.9, temperature=1.0):
    input_ids = torch.as_tensor(input_tokens, dtype=torch.long, device=next(model.parameters()).device)
    if input_ids.ndim == 1:
        input_ids = input_ids.unsqueeze(0)
    return generate_tokens(
        model, input_ids, max_tokens_to_generate,
        top_p=top_p, temperature=temperature,
    )
