import torch
from torch import nn

from cs336_basics.linear import LinearModule
from cs336_basics.rope import RoPE
from cs336_basics.scaled_dot_product_attention import ScaledDotProductAttention


class CausalMultiHeadAttentionNoWeight(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len: int, theta: float, device=None):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.rope = RoPE(theta, self.head_dim, max_seq_len, device=device)
        self.wq = LinearModule(d_model, d_model, device=device)
        self.wk = LinearModule(d_model, d_model, device=device)
        self.wv = LinearModule(d_model, d_model, device=device)
        self.wo = LinearModule(d_model, d_model, device=device)
        self.attention = ScaledDotProductAttention()

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        *leading, seq_len, d_model = x.shape

        def split_heads(projection):
            return projection(x).reshape(*leading, seq_len, self.n_heads, self.head_dim).transpose(-3, -2)

        q = self.rope(split_heads(self.wq), token_positions)
        k = self.rope(split_heads(self.wk), token_positions)
        v = split_heads(self.wv)
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device).tril()
        out = self.attention(q, k, v, mask)
        return self.wo(out.transpose(-3, -2).contiguous().reshape(*leading, seq_len, d_model))
