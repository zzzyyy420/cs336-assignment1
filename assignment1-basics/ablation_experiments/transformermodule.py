import torch
from torch import nn

from cs336_basics.RMSnorm import RMSNorm
from cs336_basics.embedding import EmbeddingModule
from cs336_basics.linear import LinearModule

from .transformer_no_weight_block import TransformerBlock


class TransformerModule(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, max_seq_len: int,
                 theta: float, n_layers: int, vocab_size: int, device=None):
        super().__init__()
        self.context_length = max_seq_len
        self.vocab_size = vocab_size
        self.embedding_module = EmbeddingModule(vocab_size, d_model, device=device)
        self.transformer_blocks = nn.ModuleList(
            TransformerBlock(d_model, n_heads, d_ff, max_seq_len, theta, device=device)
            for _ in range(n_layers)
        )
        self.final_norm = RMSNorm(d_model, device=device)
        self.linear_module = LinearModule(d_model, vocab_size, device=device)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding_module(token_ids)
        for block in self.transformer_blocks:
            x = block(x)
        return self.linear_module(self.final_norm(x))
